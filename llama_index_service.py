"""Standalone LlamaIndex RAG service for semantic search and cross-paper queries"""

import asyncio
import os
from typing import List, Optional, Dict, Any

from config import settings
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import make_url


class LlamaIndexService:
    """Service for LlamaIndex-powered RAG operations"""

    _instance = None
    _index = None

    def __new__(cls):
        """Singleton pattern to reuse index across requests"""
        if cls._instance is None:
            cls._instance = super(LlamaIndexService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize LlamaIndex with PostgreSQL vector store"""
        if self._initialized:
            return

        # Set LlamaIndex API key if provided
        if settings.LLAMAINDEX_API_KEY:
            os.environ["LLAMA_CLOUD_API_KEY"] = settings.LLAMAINDEX_API_KEY

        # Configure global LlamaIndex settings with retry logic
        # max_retries enables exponential backoff for rate limit errors
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
            max_retries=5,  # Exponential backoff on rate limit
            timeout=120.0,  # 2 minute timeout for long operations
        )

        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY,
            max_retries=5,  # Exponential backoff on rate limit
            timeout=60.0,  # 1 minute timeout for embeddings
        )

        # Initialize PGVectorStore with PostgreSQL connection
        if LlamaIndexService._index is None:
            # Parse DATABASE_URL
            db_url = make_url(settings.DATABASE_URL)

            # Build connection params, only including credentials if they exist
            connection_params = {
                "database": db_url.database,
                "host": db_url.host or 'localhost',
                "port": db_url.port or 5432,
                "table_name": "llamaindex",  # PGVectorStore will prefix with "data_", becoming "data_llamaindex"
                "embed_dim": 1536,  # text-embedding-3-small dimension
                "hnsw_kwargs": {"hnsw_m": 16, "hnsw_ef_construction": 64, "hnsw_ef_search": 40},  # HNSW index params
            }

            # Only add user/password if they exist in the URL
            if db_url.username:
                connection_params["user"] = db_url.username
            if db_url.password:
                connection_params["password"] = db_url.password

            # Create PGVectorStore
            vector_store = PGVectorStore.from_params(**connection_params)

            # Create storage context with vector store
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # Create index with storage context
            LlamaIndexService._index = VectorStoreIndex.from_documents(
                [],
                storage_context=storage_context
            )

        self.index = LlamaIndexService._index
        self._initialized = True

    def _batch_insert_nodes(self, nodes: List, batch_size: int = None):
        """
        Insert nodes in batches to avoid PostgreSQL message size limits.
        Uses index.insert_nodes() to properly generate embeddings.

        Args:
            nodes: List of nodes to insert
            batch_size: Size of each batch (defaults to settings.VECTOR_INSERT_BATCH_SIZE)
        """
        if batch_size is None:
            batch_size = settings.VECTOR_INSERT_BATCH_SIZE

        # Split nodes into batches and insert using proper index API (generates embeddings)
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            self.index.insert_nodes(batch)

    def _insert_document_batched(self, doc: Document):
        """
        Convert a document to nodes and insert them in batches

        Args:
            doc: Document to index
        """
        from llama_index.core.node_parser import SentenceSplitter

        # Parse document into nodes using the configured node parser
        node_parser = Settings.node_parser or SentenceSplitter()
        nodes = node_parser.get_nodes_from_documents([doc])

        # Insert nodes in batches
        self._batch_insert_nodes(nodes)

    def _insert_documents_batched(self, docs: List[Document]):
        """
        Convert multiple documents to nodes and insert them in batches

        Args:
            docs: List of documents to index
        """
        from llama_index.core.node_parser import SentenceSplitter

        # Parse documents into nodes using the configured node parser
        node_parser = Settings.node_parser or SentenceSplitter()
        nodes = node_parser.get_nodes_from_documents(docs)

        # Insert nodes in batches
        self._batch_insert_nodes(nodes)

    async def index_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        full_text: str,
        authors: List[str] = None,
        github_url: Optional[str] = None,
        user_id: str = None,
        figure_analyses: List[Dict[str, Any]] = None
    ) -> None:
        """
        Index a paper for semantic search and RAG

        Args:
            paper_id: Unique paper identifier
            title: Paper title
            abstract: Paper abstract
            full_text: Full paper text
            authors: List of author names
            github_url: GitHub repository URL if available
            user_id: User who owns this paper
            figure_analyses: List of figure analysis dicts (description, insights, type, etc.)
        """
        # Create main document with rich metadata
        doc = Document(
            text=full_text,
            metadata={
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors or [],
                "github_url": github_url or "",
                "user_id": user_id or "",
                "doc_type": "paper_text"
            },
            excluded_llm_metadata_keys=["paper_id", "user_id"],  # Don't send to LLM
            excluded_embed_metadata_keys=["paper_id", "user_id"],  # Don't embed these
        )

        # Convert document to nodes and insert in batches (run in thread pool to avoid asyncio.run() error)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._insert_document_batched, doc)

        # Index each figure as a separate document for better retrieval
        if figure_analyses:
            figure_docs = []
            for idx, fig in enumerate(figure_analyses):
                # Create rich text description of the figure
                fig_text = f"""
Figure {idx + 1} from '{title}' (Page {fig.get('page', 'unknown')}):

Type: {fig.get('type', 'unknown')}
Description: {fig.get('description', 'No description available')}

Key Insights:
{chr(10).join(f'- {insight}' for insight in fig.get('key_insights', []))}
"""

                fig_doc = Document(
                    text=fig_text.strip(),
                    metadata={
                        "paper_id": paper_id,
                        "title": title,
                        "doc_type": "figure",
                        "figure_index": idx,
                        "figure_type": fig.get('type', 'unknown'),
                        "page": fig.get('page', 0),
                        "user_id": user_id or "",
                    },
                    excluded_llm_metadata_keys=["paper_id", "user_id", "figure_index"],
                    excluded_embed_metadata_keys=["paper_id", "user_id"],
                )
                figure_docs.append(fig_doc)

            # Insert all figures in batches
            if figure_docs:
                await loop.run_in_executor(None, self._insert_documents_batched, figure_docs)

    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        paper_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across papers

        Args:
            query: Search query
            top_k: Number of unique papers to return
            user_id: Filter by user (for privacy)
            paper_ids: Filter by specific paper IDs
            filters: Additional metadata filters

        Returns:
            List of matching papers with scores
        """
        # Build metadata filters for retrieval
        filter_list = []
        if user_id:
            filter_list.append(MetadataFilter(key="user_id", value=str(user_id), operator=FilterOperator.EQ))
        if paper_ids:
            filter_list.append(MetadataFilter(key="paper_id", value=[str(p) for p in paper_ids], operator=FilterOperator.IN))

        # Apply additional custom filters
        if filters:
            for key, value in filters.items():
                filter_list.append(MetadataFilter(key=key, value=value, operator=FilterOperator.EQ))

        metadata_filters = MetadataFilters(filters=filter_list) if filter_list else None

        # Create retriever with metadata filters applied at retrieval time
        retriever = self.index.as_retriever(
            similarity_top_k=top_k * 10,  # Get 10x more to account for duplicate papers
            filters=metadata_filters
        )

        # Retrieve nodes - already filtered by metadata
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(None, retriever.retrieve, query)

        # Format results and deduplicate by paper_id
        # Keep the highest scoring node for each paper
        results = []
        set()
        paper_best_nodes = {}  # Track best node per paper

        # Find best node for each paper
        for node in nodes:
            paper_id = node.metadata.get('paper_id')
            if not paper_id:
                continue

            # Keep the highest scoring node for each paper
            if paper_id not in paper_best_nodes or node.score > paper_best_nodes[paper_id].score:
                paper_best_nodes[paper_id] = node

        # Sort papers by their best node's score and take top_k
        sorted_papers = sorted(
            paper_best_nodes.items(),
            key=lambda x: x[1].score,
            reverse=True
        )[:top_k]

        # Format results
        for paper_id, node in sorted_papers:
            results.append({
                "paper_id": paper_id,
                "title": node.metadata.get('title', 'Unknown'),
                "abstract": node.metadata.get('abstract', ''),
                "score": node.score,
                "excerpt": node.text[:300] + "..." if len(node.text) > 300 else node.text,
                "github_url": node.metadata.get('github_url'),
            })

        return results

    async def ask_question(
        self,
        question: str,
        user_id: Optional[str] = None,
        paper_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Ask a question across all papers using RAG

        Args:
            question: Question to ask
            user_id: Filter papers by user
            paper_ids: Filter papers by specific paper IDs
            top_k: Number of unique papers to use as sources

        Returns:
            Answer with source citations
        """
        # Build metadata filters for retrieval
        filters = []
        if user_id:
            filters.append(MetadataFilter(key="user_id", value=str(user_id), operator=FilterOperator.EQ))
        if paper_ids:
            filters.append(MetadataFilter(key="paper_id", value=[str(p) for p in paper_ids], operator=FilterOperator.IN))

        metadata_filters = MetadataFilters(filters=filters) if filters else None

        # Create query engine with metadata filters applied at retrieval time
        # Use tree_summarize for better multi-document synthesis
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k * 10,  # Get 10x more nodes to ensure multiple papers
            response_mode="tree_summarize",  # Better for synthesizing across multiple documents
            filters=metadata_filters  # Apply filters at retrieval time
        )

        # Execute query - run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, query_engine.query, question)

        # Source nodes are already filtered by metadata
        source_nodes = response.source_nodes

        # Deduplicate sources by paper_id, keeping the best node from each paper
        paper_best_nodes = {}
        for node in source_nodes:
            paper_id = node.metadata.get('paper_id')
            if not paper_id:
                continue

            # Keep the highest scoring node for each paper
            if paper_id not in paper_best_nodes or node.score > paper_best_nodes[paper_id].score:
                paper_best_nodes[paper_id] = node

        # Sort by score and take top papers
        sorted_sources = sorted(
            paper_best_nodes.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]

        # Format response
        return {
            "question": question,
            "answer": response.response,
            "sources": [
                {
                    "paper_id": node.metadata.get('paper_id'),
                    "title": node.metadata.get('title'),
                    "excerpt": node.text[:200] + "...",
                    "score": node.score
                }
                for node in sorted_sources
            ],
            "metadata": {
                "num_sources": len(sorted_sources),
                "num_unique_papers": len(paper_best_nodes),
                "top_k_used": top_k
            }
        }

    async def compare_papers(
        self,
        paper_ids: List[str],
        aspect: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare specific papers on a dimension

        Args:
            paper_ids: List of paper IDs to compare
            aspect: What to compare (e.g., "methodology", "findings", "approach")
            user_id: User ID for verification

        Returns:
            Comparison analysis
        """
        # Build metadata filters for retrieval
        filter_list = [MetadataFilter(key="paper_id", value=[str(p) for p in paper_ids], operator=FilterOperator.IN)]
        if user_id:
            filter_list.append(MetadataFilter(key="user_id", value=str(user_id), operator=FilterOperator.EQ))

        metadata_filters = MetadataFilters(filters=filter_list)

        # Retrieve specific papers from index with filters
        retriever = self.index.as_retriever(
            similarity_top_k=100,
            filters=metadata_filters
        )

        # Get all nodes - already filtered to requested papers
        query = f"Compare {aspect}"
        loop = asyncio.get_event_loop()
        selected_nodes = await loop.run_in_executor(None, retriever.retrieve, query)

        if not selected_nodes:
            return {
                "aspect": aspect,
                "comparison": "No papers found to compare.",
                "papers_compared": []
            }

        # Create a sub-index from selected documents
        sub_docs = [
            Document(
                text=node.text,
                metadata=node.metadata
            )
            for node in selected_nodes
        ]

        sub_index = VectorStoreIndex.from_documents(sub_docs)

        # Query for comparison - run in executor to avoid blocking
        comparison_query = f"Compare these papers in terms of their {aspect}. Provide a detailed comparison highlighting similarities and differences."
        query_engine = sub_index.as_query_engine()
        response = await loop.run_in_executor(None, query_engine.query, comparison_query)

        return {
            "aspect": aspect,
            "comparison": response.response,
            "papers_compared": [
                {
                    "paper_id": node.metadata.get('paper_id'),
                    "title": node.metadata.get('title')
                }
                for node in selected_nodes
            ]
        }

    async def get_similar_papers(
        self,
        paper_id: str,
        top_k: int = 5,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find papers similar to a given paper

        Args:
            paper_id: Reference paper ID
            top_k: Number of similar papers to return
            user_id: Filter by user

        Returns:
            List of similar papers
        """
        # Get the reference paper's content
        retriever = self.index.as_retriever(similarity_top_k=100)
        loop = asyncio.get_event_loop()
        all_nodes = await loop.run_in_executor(None, retriever.retrieve, "similarity search")

        ref_node = None
        for node in all_nodes:
            if node.metadata.get('paper_id') == paper_id:
                ref_node = node
                break

        if not ref_node:
            return []

        # Use the paper's abstract as query
        query = ref_node.metadata.get('abstract', ref_node.text[:500])

        # Search for similar papers
        results = await self.semantic_search(
            query=query,
            top_k=top_k + 1,  # +1 because the paper itself might be included
            user_id=user_id
        )

        # Remove the reference paper from results
        results = [r for r in results if r['paper_id'] != paper_id]

        return results[:top_k]

    async def search_figures(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        figure_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for figures across papers

        Args:
            query: Search query (e.g., "plots showing accuracy", "network diagrams")
            top_k: Number of figures to return
            user_id: Filter by user
            figure_type: Filter by figure type (plot, diagram, table, etc.)

        Returns:
            List of matching figures with metadata
        """
        # Build metadata filters for retrieval
        filter_list = [MetadataFilter(key="doc_type", value="figure", operator=FilterOperator.EQ)]
        if user_id:
            filter_list.append(MetadataFilter(key="user_id", value=str(user_id), operator=FilterOperator.EQ))
        if figure_type:
            filter_list.append(MetadataFilter(key="figure_type", value=figure_type.lower(), operator=FilterOperator.EQ))

        metadata_filters = MetadataFilters(filters=filter_list)

        # Create retriever with metadata filters
        retriever = self.index.as_retriever(
            similarity_top_k=top_k * 3,  # Get more to filter
            filters=metadata_filters
        )

        # Retrieve nodes - already filtered by metadata
        # Run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        figure_nodes = await loop.run_in_executor(None, retriever.retrieve, query)

        # Format results
        results = []
        for node in figure_nodes[:top_k]:
            results.append({
                "paper_id": node.metadata.get('paper_id'),
                "paper_title": node.metadata.get('title'),
                "figure_index": node.metadata.get('figure_index'),
                "figure_type": node.metadata.get('figure_type'),
                "page": node.metadata.get('page'),
                "description": node.text,
                "score": node.score
            })

        return results

    async def delete_paper(self, paper_id: str) -> bool:
        """
        Remove a paper and all its associated documents (text, figures) from the vector index.

        Args:
            paper_id: Paper ID to remove

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Create metadata filter to match all documents with this paper_id
            filters = MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="paper_id",
                        value=str(paper_id),
                        operator=FilterOperator.EQ
                    )
                ]
            )

            # Delete all nodes matching this paper_id using the vector store's delete_nodes method
            # This deletes both the main paper document and any figure documents
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.index._vector_store.delete_nodes(filters=filters)
            )

            return True
        except Exception as e:
            print(f"Error deleting paper {paper_id} from vector index: {str(e)}")
            return False

    async def delete_papers(self, paper_ids: List[str]) -> Dict[str, bool]:
        """
        Remove multiple papers from the vector index.

        Args:
            paper_ids: List of paper IDs to remove

        Returns:
            Dictionary mapping paper_id to deletion success status
        """
        results = {}
        for paper_id in paper_ids:
            results[paper_id] = await self.delete_paper(paper_id)
        return results