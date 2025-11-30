"""Standalone schemas for insight analysis - no frontend dependencies"""

from typing import Optional, List, Any, Dict
from dataclasses import dataclass, field


@dataclass
class InsightSource:
    """Source citation for an insight"""
    paper_id: str
    title: str
    excerpt: str
    score: float


@dataclass
class InsightAnalysisRequest:
    """Request for cross-paper insights analysis"""
    category: str  # methodological_trends, mathematical_consistency, research_gaps, contradictions, novel_connections
    focus_area: Optional[str] = None
    paper_ids: Optional[List[str]] = None


@dataclass
class InsightAnalysisResponse:
    """Response from insight analysis"""
    category: str
    analysis: str
    sources: List[InsightSource]
    num_papers_analyzed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComprehensiveInsightsResponse:
    """Response from comprehensive insights analysis"""
    user_id: str
    focus_areas: List[str]
    insights: Dict[str, Any]
    executive_summary: Dict[str, Any]


@dataclass
class PaperComparisonRequest:
    """Request for comparing specific papers"""
    paper_ids: List[str]
    comparison_aspects: Optional[List[str]] = None


@dataclass
class PaperComparisonResponse:
    """Response from paper comparison"""
    paper_ids: List[str]
    num_papers: int
    comparisons: Dict[str, Any]
    synthesis: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticSearchResult:
    """Result from semantic search"""
    paper_id: str
    title: str
    abstract: str
    score: float
    excerpt: str
    github_url: Optional[str] = None


@dataclass
class RAGQueryResponse:
    """Response from RAG query"""
    question: str
    answer: str
    sources: List[InsightSource]
    metadata: Dict[str, Any] = field(default_factory=dict)
