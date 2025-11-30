"""
Standalone Insights Analysis Module

This module provides isolated insight analysis capabilities without frontend dependencies.
It can analyze research papers for:
- Methodological Trends
- Mathematical Consistency
- Research Gaps
- Contradictions
- Novel Connections
"""

from .paper_insights import PaperInsightsService, InsightCategory
from .llama_index_service import LlamaIndexService
from .config import settings

__all__ = [
    "PaperInsightsService",
    "InsightCategory",
    "LlamaIndexService",
    "settings",
    "run_gui"
]


def run_gui():
    """Launch the GUI application"""
    from .gui import main
    main()