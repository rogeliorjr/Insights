#!/usr/bin/env python3
"""
Standalone CLI for paper insight analysis.

Usage:
    python -m Insights.cli --user-id <USER_ID> --category <CATEGORY> [--focus <FOCUS_AREA>]

Categories:
    - methodological_trends: Find common methods and approaches across papers
    - mathematical_consistency: Check if equations and formulations are consistent
    - research_gaps: Identify missed opportunities and unexplored areas
    - contradictions: Find conflicting results or claims
    - novel_connections: Discover synthesis opportunities
    - comprehensive: Run all analyses

Environment Variables Required:
    - OPENAI_API_KEY: Your OpenAI API key
    - DATABASE_URL: PostgreSQL connection string with pgvector

Example:
    export OPENAI_API_KEY="sk-..."
    export DATABASE_URL="postgresql://user:pass@localhost/insightsync_db"
    python -m Insights.cli --user-id "abc-123" --category research_gaps --focus "machine learning"
"""

import argparse
import asyncio
import json
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze research papers for insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--user-id",
        required=True,
        help="User ID to filter papers"
    )

    parser.add_argument(
        "--category",
        required=True,
        choices=[
            "methodological_trends",
            "mathematical_consistency",
            "research_gaps",
            "contradictions",
            "novel_connections",
            "comprehensive"
        ],
        help="Type of analysis to perform"
    )

    parser.add_argument(
        "--focus",
        default=None,
        help="Optional focus area or topic for the analysis"
    )

    parser.add_argument(
        "--paper-ids",
        nargs="+",
        default=None,
        help="Optional specific paper IDs to analyze (space-separated)"
    )

    parser.add_argument(
        "--paper-count",
        type=int,
        default=50,
        help="Estimated number of papers (used for query optimization)"
    )

    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional file to write output to"
    )

    return parser.parse_args()


async def run_analysis(
    user_id: str,
    category: str,
    focus: Optional[str] = None,
    paper_ids: Optional[list] = None,
    paper_count: int = 50
) -> dict:
    """Run the specified analysis category"""

    from paper_insights import PaperInsightsService

    service = PaperInsightsService(paper_count=paper_count)

    if category == "methodological_trends":
        return await service.analyze_methodological_trends(
            user_id=user_id,
            paper_ids=paper_ids
        )

    elif category == "mathematical_consistency":
        return await service.check_mathematical_consistency(
            user_id=user_id,
            focus_area=focus,
            paper_ids=paper_ids
        )

    elif category == "research_gaps":
        return await service.identify_research_gaps(
            user_id=user_id,
            research_area=focus,
            paper_ids=paper_ids
        )

    elif category == "contradictions":
        return await service.find_contradictions(
            user_id=user_id,
            topic=focus,
            paper_ids=paper_ids
        )

    elif category == "novel_connections":
        return await service.discover_novel_connections(
            user_id=user_id,
            theme=focus,
            paper_ids=paper_ids
        )

    elif category == "comprehensive":
        focus_areas = [focus] if focus else None
        return await service.generate_comprehensive_insights(
            user_id=user_id,
            focus_areas=focus_areas
        )

    else:
        raise ValueError(f"Unknown category: {category}")


def format_output(result: dict, output_format: str) -> str:
    """Format the analysis result for output"""

    if output_format == "json":
        return json.dumps(result, indent=2, default=str)

    # Text format
    lines = []
    lines.append("=" * 60)
    lines.append(f"ANALYSIS: {result.get('category', 'Unknown').upper()}")
    lines.append("=" * 60)

    if "focus_area" in result:
        lines.append(f"Focus Area: {result['focus_area']}")
    if "research_area" in result:
        lines.append(f"Research Area: {result['research_area']}")
    if "topic" in result:
        lines.append(f"Topic: {result['topic']}")
    if "theme" in result:
        lines.append(f"Theme: {result['theme']}")

    lines.append(f"Papers Analyzed: {result.get('num_papers_analyzed', 'N/A')}")
    lines.append("-" * 60)
    lines.append("")

    # Main analysis
    if "analysis" in result:
        lines.append("ANALYSIS:")
        lines.append(result["analysis"])
        lines.append("")

    # Executive summary (for comprehensive)
    if "executive_summary" in result:
        lines.append("-" * 60)
        lines.append("EXECUTIVE SUMMARY:")
        lines.append(result["executive_summary"].get("summary", ""))
        lines.append("")

    # Sources
    if "sources" in result and result["sources"]:
        lines.append("-" * 60)
        lines.append("SOURCES:")
        for i, source in enumerate(result["sources"], 1):
            lines.append(f"  {i}. {source.get('title', 'Unknown')} (score: {source.get('score', 0):.3f})")
            if source.get('excerpt'):
                lines.append(f"     Excerpt: {source['excerpt'][:100]}...")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    args = parse_args()

    logger.info(f"Starting analysis: {args.category} for user {args.user_id}")

    try:
        # Run the analysis
        result = asyncio.run(run_analysis(
            user_id=args.user_id,
            category=args.category,
            focus=args.focus,
            paper_ids=args.paper_ids,
            paper_count=args.paper_count
        ))

        # Format output
        output = format_output(result, args.output)

        # Write to file or stdout
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(output)
            logger.info(f"Output written to {args.output_file}")
        else:
            print(output)

        logger.info("Analysis complete")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
