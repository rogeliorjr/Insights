"""
Standalone cross-paper insight analysis service using Llama Index.

Analyzes relationships between papers to find:
- Methodological trends and patterns
- Mathematical consistency and contradictions
- Research gaps and missed insights
- Common themes and divergences

This version works without SQLAlchemy/database session - uses direct paper count parameter.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
from llama_index_service import LlamaIndexService

logger = logging.getLogger(__name__)


@dataclass
class InsightCategory:
    """Categories of insights we can extract"""
    METHODOLOGICAL_TRENDS = "methodological_trends"
    MATHEMATICAL_CONSISTENCY = "mathematical_consistency"
    RESEARCH_GAPS = "research_gaps"
    COMMON_THEMES = "common_themes"
    CONTRADICTIONS = "contradictions"
    NOVEL_CONNECTIONS = "novel_connections"


class PaperInsightsService:
    """
    Service for analyzing relationships and extracting insights across papers.

    Uses Llama Index RAG to:
    1. Find methodological trends
    2. Check mathematical consistency
    3. Identify research gaps
    4. Detect contradictions
    5. Discover novel connections
    """

    def __init__(self, paper_count: int = 50):
        """Initialize the insight analysis service

        Args:
            paper_count: Estimated number of papers for the user (used for top_k calculations)
        """
        self.paper_count = paper_count
        self.llama_service = LlamaIndexService()

    def _get_top_k(self, paper_ids: Optional[List[str]] = None) -> int:
        """Calculate appropriate top_k based on paper count"""
        if paper_ids:
            return max(len(paper_ids), 50)
        return max(self.paper_count, 50)

    async def analyze_methodological_trends(
        self,
        user_id: str,
        paper_ids: Optional[List[str]] = None,
        min_papers: int = 2
    ) -> Dict[str, Any]:
        """
        Analyze methodological trends across papers.

        Identifies:
        - Common approaches and techniques
        - Evolution of methods over time
        - Popular vs. emerging methods
        - Method effectiveness comparisons

        Args:
            user_id: User ID to filter papers
            paper_ids: Optional specific papers to analyze (None = all papers)
            min_papers: Minimum number of papers needed for trend analysis

        Returns:
            Dictionary with trend analysis results
        """
        logger.info(f"Analyzing methodological trends for user {user_id}")

        try:
            top_k = self._get_top_k(paper_ids)

            # Query for methodological information
            query = """Analyze the methodological approaches across these research papers.

Focus on:
1. What are the most common methods and techniques used?
2. Are there any emerging or novel methodologies?
3. How do the approaches differ between papers?
4. Which methods are most effective based on reported results?
5. Are there any methodological gaps or unexplored approaches?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers for each major section
- Use **bold** for method names and key findings
- Use bullet points for lists of methods/techniques
- Use > blockquotes when citing specific papers
- Include a brief "Summary" section at the end"""

            # Use RAG to gather and analyze methodology across papers
            response = await self.llama_service.ask_question(
                question=query,
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            # Also get semantic search for methods
            method_search = await self.llama_service.semantic_search(
                query="methodologies techniques approaches algorithms implementations",
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            return {
                "category": "methodological_trends",
                "analysis": response["answer"],
                "sources": response["sources"],
                "num_papers_analyzed": len(set(s["paper_id"] for s in response["sources"])),
                "relevant_papers": method_search,
                "metadata": {
                    "min_papers_required": min_papers,
                    "query_used": query
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing methodological trends: {str(e)}")
            raise

    async def check_mathematical_consistency(
        self,
        user_id: str,
        focus_area: Optional[str] = None,
        paper_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check mathematical consistency across papers.

        Identifies:
        - Common equations and formulations
        - Mathematical contradictions or discrepancies
        - Different notations for same concepts
        - Theoretical consistency

        Args:
            user_id: User ID to filter papers
            focus_area: Optional focus area (e.g., "loss functions", "optimization")
            paper_ids: Optional specific papers to analyze

        Returns:
            Dictionary with consistency analysis
        """
        logger.info(f"Checking mathematical consistency for user {user_id}")

        try:
            # Build query based on focus area
            if focus_area:
                query = f"""Analyze the mathematical formulations related to {focus_area} across these papers.

Focus on:
1. What are the key equations and mathematical formulations?
2. Are the mathematical approaches consistent across papers?
3. Are there any contradictions or discrepancies in the math?
4. Do different papers use different notations for the same concepts?
5. Are there any mathematical errors or inconsistencies you notice?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers for each analysis area
- Use **bold** for equation names and key concepts
- Use `code blocks` for mathematical notation when possible
- Use > blockquotes when citing specific papers
- Create a "Consistency Summary" table if comparing multiple formulations"""
            else:
                query = """Analyze the mathematical consistency across these research papers.

Focus on:
1. What are the common mathematical formulations and equations?
2. Are the theoretical frameworks consistent?
3. Are there any mathematical contradictions between papers?
4. Do papers build on each other's mathematical foundations?
5. Are there gaps in the mathematical coverage?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers for each analysis area
- Use **bold** for equation names and key concepts
- Use > blockquotes when citing specific papers
- Include a brief summary of consistency findings at the end"""

            top_k = self._get_top_k(paper_ids)

            response = await self.llama_service.ask_question(
                question=query,
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            # Search for equations specifically
            equation_search = await self.llama_service.semantic_search(
                query="equations mathematical formulations theorems proofs",
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            return {
                "category": "mathematical_consistency",
                "focus_area": focus_area or "general",
                "analysis": response["answer"],
                "sources": response["sources"],
                "num_papers_analyzed": len(set(s["paper_id"] for s in response["sources"])),
                "papers_with_equations": equation_search,
                "metadata": {
                    "query_used": query
                }
            }

        except Exception as e:
            logger.error(f"Error checking mathematical consistency: {str(e)}")
            raise

    async def identify_research_gaps(
        self,
        user_id: str,
        research_area: Optional[str] = None,
        paper_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Identify research gaps and missed opportunities across papers.

        Identifies:
        - Unexplored areas mentioned but not investigated
        - Questions raised but not answered
        - Limitations that could be addressed
        - Promising directions not pursued

        Args:
            user_id: User ID to filter papers
            research_area: Optional specific research area to focus on

        Returns:
            Dictionary with identified gaps and opportunities
        """
        logger.info(f"Identifying research gaps for user {user_id}")

        try:
            if research_area:
                query = f"""Analyze research gaps and missed opportunities related to {research_area} across these papers.

Focus on:
1. What questions do the papers raise but not fully answer?
2. What limitations do the authors acknowledge?
3. What future work directions are mentioned across papers?
4. Are there connections between papers that the authors might have missed?
5. What topics are mentioned but not deeply explored?
6. What contradictions or unresolved issues exist between papers?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers: "Unanswered Questions", "Acknowledged Limitations", "Future Directions", "Missed Connections"
- Use **bold** for gap/opportunity names
- Use numbered lists for prioritized opportunities
- Use > blockquotes when citing specific papers
- End with "Top 3 Opportunities" summary"""
            else:
                query = """Identify research gaps, missed insights, and opportunities across these papers.

Focus on:
1. What important questions remain unanswered?
2. What limitations are commonly acknowledged but not addressed?
3. What future work is suggested across papers?
4. Are there insights from one paper that could solve problems in another?
5. What areas are underexplored despite being mentioned?
6. What novel combinations or connections could be made?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers to organize by gap type
- Use **bold** for key gaps and opportunities
- Use numbered lists for actionable items
- Use > blockquotes when citing papers
- End with "Priority Research Gaps" summary"""

            top_k = self._get_top_k(paper_ids)

            response = await self.llama_service.ask_question(
                question=query,
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            # Search for limitations and future work sections
            limitations_search = await self.llama_service.semantic_search(
                query="limitations future work challenges open questions gaps",
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            return {
                "category": "research_gaps",
                "research_area": research_area or "general",
                "analysis": response["answer"],
                "sources": response["sources"],
                "num_papers_analyzed": len(set(s["paper_id"] for s in response["sources"])),
                "papers_with_limitations": limitations_search,
                "metadata": {
                    "query_used": query
                }
            }

        except Exception as e:
            logger.error(f"Error identifying research gaps: {str(e)}")
            raise

    async def find_contradictions(
        self,
        user_id: str,
        topic: Optional[str] = None,
        paper_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Find contradictions and disagreements across papers.

        Identifies:
        - Conflicting results or conclusions
        - Different interpretations of same phenomena
        - Methodological disagreements
        - Theoretical conflicts

        Args:
            user_id: User ID to filter papers
            topic: Optional specific topic to check for contradictions

        Returns:
            Dictionary with identified contradictions
        """
        logger.info(f"Finding contradictions for user {user_id}")

        try:
            if topic:
                query = f"""Find contradictions and disagreements about {topic} across these papers.

Focus on:
1. Do papers report conflicting results or findings?
2. Are there different interpretations or conclusions?
3. Do papers disagree on methodology or approach?
4. Are there theoretical conflicts or incompatibilities?
5. What might explain these contradictions?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers for each type of contradiction
- Use a comparison format: **Paper A** claims X, while **Paper B** claims Y
- Use > blockquotes for direct claims from papers
- Use bullet points for possible explanations
- End with "Most Significant Contradictions" summary"""
            else:
                query = """Identify contradictions, conflicts, and disagreements across these research papers.

Focus on:
1. What conflicting results or findings exist?
2. Where do papers draw different conclusions from similar data?
3. What methodological disagreements are present?
4. Are there theoretical incompatibilities?
5. Which contradictions are most significant?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers for each category of contradiction
- Format as: **Paper A** vs **Paper B** for clear comparisons
- Use > blockquotes for specific claims
- Rate contradiction severity (Minor/Moderate/Major)
- End with prioritized list of contradictions to investigate"""

            top_k = self._get_top_k(paper_ids)

            response = await self.llama_service.ask_question(
                question=query,
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            # Search for results and conclusions sections
            results_search = await self.llama_service.semantic_search(
                query="results findings conclusions claims evidence",
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            return {
                "category": "contradictions",
                "topic": topic or "general",
                "analysis": response["answer"],
                "sources": response["sources"],
                "num_papers_analyzed": len(set(s["paper_id"] for s in response["sources"])),
                "papers_with_results": results_search,
                "metadata": {
                    "query_used": query
                }
            }

        except Exception as e:
            logger.error(f"Error finding contradictions: {str(e)}")
            raise

    async def discover_novel_connections(
        self,
        user_id: str,
        theme: Optional[str] = None,
        paper_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Discover novel connections and synthesis opportunities across papers.

        Identifies:
        - Complementary approaches that could be combined
        - Insights from one paper applicable to another
        - Common underlying principles
        - Potential collaborations or integrations

        Args:
            user_id: User ID to filter papers
            theme: Optional theme to explore connections around

        Returns:
            Dictionary with discovered connections
        """
        logger.info(f"Discovering novel connections for user {user_id}")

        try:
            if theme:
                query = f"""Discover novel connections and synthesis opportunities related to {theme} across these papers.

Focus on:
1. What complementary approaches could be combined?
2. How could insights from one paper enhance another?
3. What common underlying principles exist?
4. What novel combinations or integrations are possible?
5. Are there unexpected relationships between papers?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers: "Complementary Approaches", "Cross-Paper Insights", "Underlying Principles", "Novel Combinations"
- Use **bold** for paper titles and key concepts
- Format connections as: **Paper A** + **Paper B** → Potential outcome
- Use bullet points for specific synthesis ideas
- End with "Top 3 Novel Research Directions" """
            else:
                query = """Discover novel connections, relationships, and synthesis opportunities across these papers.

Focus on:
1. What papers have complementary strengths that could be combined?
2. How do insights from different papers relate to each other?
3. What underlying patterns or principles connect the papers?
4. What innovative combinations or integrations are possible?
5. Are there unexpected connections worth exploring?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers to organize connection types
- Use **bold** for paper titles and key concepts
- Format as: **Paper A** ↔ **Paper B**: connection description
- Use numbered lists for actionable synthesis opportunities
- End with "Most Promising Combinations" summary"""

            top_k = self._get_top_k(paper_ids)

            response = await self.llama_service.ask_question(
                question=query,
                user_id=user_id,
                paper_ids=paper_ids,
                top_k=top_k
            )

            return {
                "category": "novel_connections",
                "theme": theme or "general",
                "analysis": response["answer"],
                "sources": response["sources"],
                "num_papers_analyzed": len(set(s["paper_id"] for s in response["sources"])),
                "metadata": {
                    "query_used": query
                }
            }

        except Exception as e:
            logger.error(f"Error discovering novel connections: {str(e)}")
            raise

    async def generate_comprehensive_insights(
        self,
        user_id: str,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive insights across all categories.

        Args:
            user_id: User ID to filter papers
            focus_areas: Optional list of specific areas to focus on

        Returns:
            Dictionary with comprehensive multi-category analysis
        """
        logger.info(f"Generating comprehensive insights for user {user_id}")

        try:
            # Run all analyses in parallel would be ideal, but let's do sequentially for now
            results = {
                "user_id": user_id,
                "focus_areas": focus_areas or ["general"],
                "insights": {}
            }

            # Methodological trends
            results["insights"]["methodological_trends"] = await self.analyze_methodological_trends(
                user_id=user_id
            )

            # Mathematical consistency
            results["insights"]["mathematical_consistency"] = await self.check_mathematical_consistency(
                user_id=user_id,
                focus_area=focus_areas[0] if focus_areas else None
            )

            # Research gaps
            results["insights"]["research_gaps"] = await self.identify_research_gaps(
                user_id=user_id,
                research_area=focus_areas[0] if focus_areas else None
            )

            # Contradictions
            results["insights"]["contradictions"] = await self.find_contradictions(
                user_id=user_id,
                topic=focus_areas[0] if focus_areas else None
            )

            # Novel connections
            results["insights"]["novel_connections"] = await self.discover_novel_connections(
                user_id=user_id,
                theme=focus_areas[0] if focus_areas else None
            )

            # Generate executive summary
            summary_query = """Based on your analysis of all my papers, provide an executive summary:

1. What are the 3-5 most important insights from analyzing my papers together?
2. What are the biggest opportunities I should pursue?
3. What critical gaps or contradictions need attention?
4. What novel research directions emerge from combining these papers?

FORMAT YOUR RESPONSE using markdown:
- Use ### headers for each section
- Use **bold** for key insights and findings
- Use numbered lists for prioritized items
- Keep each point concise (1-2 sentences)
- End with a "Bottom Line" one-paragraph summary"""

            summary_response = await self.llama_service.ask_question(
                question=summary_query,
                user_id=user_id,
                top_k=10
            )

            results["executive_summary"] = {
                "summary": summary_response["answer"],
                "num_papers_analyzed": len(set(
                    s["paper_id"]
                    for insight in results["insights"].values()
                    for s in insight.get("sources", [])
                ))
            }

            return results

        except Exception as e:
            logger.error(f"Error generating comprehensive insights: {str(e)}")
            raise

    async def compare_specific_papers(
        self,
        paper_ids: List[str],
        user_id: str,
        comparison_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Deep comparison of specific papers.

        Args:
            paper_ids: List of paper IDs to compare (2-5 papers recommended)
            user_id: User ID for verification
            comparison_aspects: Optional aspects to focus on

        Returns:
            Dictionary with detailed comparison
        """
        logger.info(f"Comparing {len(paper_ids)} specific papers")

        if len(paper_ids) < 2:
            raise ValueError("Need at least 2 papers to compare")

        try:
            aspects = comparison_aspects or [
                "methodology",
                "results",
                "mathematical_formulations",
                "limitations",
                "contributions"
            ]

            comparisons = {}

            for aspect in aspects:
                comparison = await self.llama_service.compare_papers(
                    paper_ids=paper_ids,
                    aspect=aspect,
                    user_id=user_id
                )
                comparisons[aspect] = comparison

            # Generate synthesis
            synthesis_query = f"""After comparing these {len(paper_ids)} papers across multiple aspects, provide:

1. How do these papers relate to and build on each other?
2. What are the key differences in their approaches?
3. What insights from one paper could benefit the others?
4. What research opportunities emerge from combining these papers?
5. Are there any contradictions or inconsistencies to address?"""

            synthesis = await self.llama_service.ask_question(
                question=synthesis_query,
                user_id=user_id,
                top_k=len(paper_ids) * 2
            )

            return {
                "paper_ids": paper_ids,
                "num_papers": len(paper_ids),
                "comparisons": comparisons,
                "synthesis": synthesis["answer"],
                "metadata": {
                    "aspects_compared": aspects
                }
            }

        except Exception as e:
            logger.error(f"Error comparing specific papers: {str(e)}")
            raise
