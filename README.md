# Standalone Insights

A standalone module for cross-paper research analysis using LlamaIndex RAG. Analyzes relationships between research papers to extract actionable insights.

## Analysis Types

| Category | Description |
|----------|-------------|
| **Methodological Trends** | Find common methods and approaches across papers |
| **Mathematical Consistency** | Check if equations and formulations are consistent |
| **Research Gaps** | Identify missed opportunities and unexplored areas |
| **Contradictions** | Find conflicting results or claims |
| **Novel Connections** | Discover synthesis opportunities |

## Requirements

- Python 3.9+
- PostgreSQL with pgvector extension
- OpenAI API key

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export DATABASE_URL="postgresql://user:password@localhost:5432/your_database"

# Optional
export LLAMAINDEX_API_KEY="..."  # For LlamaCloud features
export VECTOR_INSERT_BATCH_SIZE="50"  # Batch size for vector insertions
```

## Usage

### GUI (Recommended for beginners)

```bash
# Launch the graphical interface
python -m Insights.gui

# Or from Python
from Insights import run_gui
run_gui()
```

The GUI provides:
- Dropdown for analysis type selection
- Input fields for User ID, focus area, and paper IDs
- One-click analysis execution
- Dark-themed results display

### CLI

```bash
# Single category analysis
python -m Insights.cli \
  --user-id "user-uuid" \
  --category research_gaps \
  --focus "machine learning"

# Comprehensive analysis (all categories)
python -m Insights.cli \
  --user-id "user-uuid" \
  --category comprehensive

# Analyze specific papers
python -m Insights.cli \
  --user-id "user-uuid" \
  --category contradictions \
  --paper-ids "paper-1" "paper-2" "paper-3"

# Output as JSON
python -m Insights.cli \
  --user-id "user-uuid" \
  --category methodological_trends \
  --output json \
  --output-file results.json
```

### CLI Options

| Option | Required | Description |
|--------|----------|-------------|
| `--user-id` | Yes | User ID to filter papers |
| `--category` | Yes | Analysis type (see above) |
| `--focus` | No | Focus area or topic |
| `--paper-ids` | No | Specific paper IDs (space-separated) |
| `--paper-count` | No | Estimated paper count (default: 50) |
| `--output` | No | Format: `text` or `json` (default: text) |
| `--output-file` | No | Write output to file |

### Python API

```python
import asyncio
from standalone_insights import PaperInsightsService, InsightCategory

async def analyze():
    # Initialize service
    service = PaperInsightsService(paper_count=50)

    # Run single analysis
    result = await service.identify_research_gaps(
        user_id="user-uuid",
        research_area="neural networks"
    )
    print(result["analysis"])

    # Run comprehensive analysis
    comprehensive = await service.generate_comprehensive_insights(
        user_id="user-uuid",
        focus_areas=["deep learning"]
    )
    print(comprehensive["executive_summary"]["summary"])

asyncio.run(analyze())
```

### Available Methods

```python
# Methodological Trends
await service.analyze_methodological_trends(user_id, paper_ids=None, min_papers=2)

# Mathematical Consistency
await service.check_mathematical_consistency(user_id, focus_area=None, paper_ids=None)

# Research Gaps
await service.identify_research_gaps(user_id, research_area=None, paper_ids=None)

# Contradictions
await service.find_contradictions(user_id, topic=None, paper_ids=None)

# Novel Connections
await service.discover_novel_connections(user_id, theme=None, paper_ids=None)

# Comprehensive (all categories)
await service.generate_comprehensive_insights(user_id, focus_areas=None)

# Compare Specific Papers
await service.compare_specific_papers(paper_ids, user_id, comparison_aspects=None)
```

## Response Format

All analysis methods return a dictionary with:

```python
{
    "category": "research_gaps",
    "analysis": "Markdown formatted analysis text...",
    "sources": [
        {
            "paper_id": "uuid",
            "title": "Paper Title",
            "excerpt": "Relevant excerpt...",
            "score": 0.85
        }
    ],
    "num_papers_analyzed": 5,
    "metadata": {
        "query_used": "..."
    }
}
```

## Database Schema

This module expects papers to be indexed in a PostgreSQL database with pgvector. The LlamaIndex service uses a table named `data_llamaindex` with the following metadata fields:

- `paper_id`: Unique paper identifier
- `user_id`: Owner user ID
- `title`: Paper title
- `abstract`: Paper abstract
- `doc_type`: "paper_text" or "figure"

## Architecture

```
standalone_insights/
├── __init__.py              # Package exports
├── config.py                # Environment configuration
├── schemas.py               # Data classes for requests/responses
├── llama_index_service.py   # RAG engine (vector search, Q&A)
├── paper_insights.py        # Core analysis logic
├── cli.py                   # Command-line interface
├── gui.py                   # Tkinter GUI (no extra dependencies)
└── requirements.txt         # Dependencies
```

## Dependencies

- `llama-index-core` - RAG framework
- `llama-index-embeddings-openai` - OpenAI embeddings
- `llama-index-llms-openai` - OpenAI LLM integration
- `llama-index-vector-stores-postgres` - PostgreSQL vector store
- `sqlalchemy` - Database toolkit
- `psycopg2-binary` - PostgreSQL adapter
- `openai` - OpenAI API client