#!/usr/bin/env python3
"""
Simple GUI for paper insight analysis using Tkinter (no extra dependencies).

Usage:
    python -m Insights.gui
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import asyncio
import threading
from typing import Optional, List, Tuple

# Default user
DEFAULT_USER_EMAIL = "rogeliorjr1@gmail.com"


class InsightsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Research Paper Insights Analyzer")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)

        # Store papers data
        self.papers: List[Tuple[str, str]] = []  # List of (paper_id, title)
        self.user_id: Optional[str] = None

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)  # Results frame row

        self._create_widgets()

        # Auto-load user and papers on startup
        self.root.after(100, self._load_user_and_papers)

    def _create_widgets(self):
        # === Input Frame ===
        input_frame = ttk.LabelFrame(self.root, text="Analysis Configuration", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        input_frame.columnconfigure(1, weight=1)

        # User ID
        ttk.Label(input_frame, text="User ID:").grid(row=0, column=0, sticky="w", pady=5)
        self.user_id_var = tk.StringVar()
        self.user_id_entry = ttk.Entry(input_frame, textvariable=self.user_id_var, width=50)
        self.user_id_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=(10, 0))

        # Category Selection
        ttk.Label(input_frame, text="Analysis Type:").grid(row=1, column=0, sticky="w", pady=5)
        self.category_var = tk.StringVar(value="research_gaps")
        self.categories = {
            "Methodological Trends": "methodological_trends",
            "Mathematical Consistency": "mathematical_consistency",
            "Research Gaps": "research_gaps",
            "Contradictions": "contradictions",
            "Novel Connections": "novel_connections",
            "Comprehensive (All)": "comprehensive"
        }
        self.category_combo = ttk.Combobox(
            input_frame,
            textvariable=self.category_var,
            values=list(self.categories.keys()),
            state="readonly",
            width=47
        )
        self.category_combo.set("Research Gaps")
        self.category_combo.grid(row=1, column=1, sticky="ew", pady=5, padx=(10, 0))

        # Focus Area (optional)
        ttk.Label(input_frame, text="Focus Area:").grid(row=2, column=0, sticky="w", pady=5)
        self.focus_var = tk.StringVar()
        self.focus_entry = ttk.Entry(input_frame, textvariable=self.focus_var, width=50)
        self.focus_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=(10, 0))
        ttk.Label(input_frame, text="(optional)", foreground="gray").grid(row=2, column=2, sticky="w", padx=5)

        # === Papers Frame ===
        papers_frame = ttk.LabelFrame(self.root, text="Available Papers (select to filter, or leave empty for all)", padding=10)
        papers_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        papers_frame.columnconfigure(0, weight=1)

        # Papers listbox with scrollbar
        papers_list_frame = ttk.Frame(papers_frame)
        papers_list_frame.grid(row=0, column=0, sticky="ew")
        papers_list_frame.columnconfigure(0, weight=1)

        self.papers_listbox = tk.Listbox(
            papers_list_frame,
            selectmode=tk.MULTIPLE,
            height=5,
            exportselection=False
        )
        self.papers_listbox.grid(row=0, column=0, sticky="ew")

        papers_scrollbar = ttk.Scrollbar(papers_list_frame, orient="vertical", command=self.papers_listbox.yview)
        papers_scrollbar.grid(row=0, column=1, sticky="ns")
        self.papers_listbox.config(yscrollcommand=papers_scrollbar.set)

        # Paper selection buttons
        papers_btn_frame = ttk.Frame(papers_frame)
        papers_btn_frame.grid(row=1, column=0, pady=(5, 0))

        ttk.Button(papers_btn_frame, text="Select All", command=self._select_all_papers, width=12).pack(side="left", padx=2)
        ttk.Button(papers_btn_frame, text="Clear Selection", command=self._clear_paper_selection, width=12).pack(side="left", padx=2)
        ttk.Button(papers_btn_frame, text="Refresh Papers", command=self._refresh_papers, width=12).pack(side="left", padx=2)

        self.papers_count_var = tk.StringVar(value="Loading papers...")
        ttk.Label(papers_frame, textvariable=self.papers_count_var, foreground="gray").grid(row=2, column=0, sticky="w", pady=(5, 0))

        # === Button Frame ===
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=2, column=0, pady=10)

        self.analyze_btn = ttk.Button(
            button_frame,
            text="Run Analysis",
            command=self._run_analysis,
            width=20
        )
        self.analyze_btn.pack(side="left", padx=5)

        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear Results",
            command=self._clear_results,
            width=15
        )
        self.clear_btn.pack(side="left", padx=5)

        # === Progress ===
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_label = ttk.Label(self.root, textvariable=self.progress_var, foreground="blue")
        self.progress_label.grid(row=2, column=0, sticky="e", padx=20)

        # === Results Frame ===
        results_frame = ttk.LabelFrame(self.root, text="Analysis Results", padding=10)
        results_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=("Courier", 11),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white"
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")

        # === Status Bar ===
        self.status_var = tk.StringVar(value="Loading user and papers...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.grid(row=4, column=0, sticky="ew", padx=5, pady=5)

    def _load_user_and_papers(self):
        """Load user ID and papers from database on startup"""
        self.progress_var.set("Loading...")
        thread = threading.Thread(target=self._load_user_and_papers_thread, daemon=True)
        thread.start()

    def _load_user_and_papers_thread(self):
        """Background thread for loading user and papers"""
        try:
            user_id, papers = asyncio.run(self._fetch_user_and_papers())
            self.root.after(0, lambda uid=user_id, p=papers: self._populate_user_and_papers(uid, p))
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self._show_load_error(msg))

    async def _fetch_user_and_papers(self) -> Tuple[Optional[str], List[Tuple[str, str]]]:
        """Fetch user ID and papers from database"""
        import os
        from pathlib import Path
        from sqlalchemy import create_engine, text

        # Try to load from .env if DATABASE_URL not in environment
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            # Check Insights/.env first, then backend/.env
            env_files = [
                Path(__file__).parent / ".env",  # Insights/.env
                Path(__file__).parent.parent / "backend" / ".env",  # backend/.env
            ]
            for env_file in env_files:
                if env_file.exists():
                    with open(env_file) as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("DATABASE_URL="):
                                database_url = line.split("=", 1)[1].strip()
                                break
                    if database_url:
                        break

        if not database_url:
            database_url = "postgresql://user:password@localhost/insightsync_db"

        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        engine = create_engine(database_url)

        with engine.connect() as conn:
            # Get user ID by email (case-insensitive)
            result = conn.execute(
                text("SELECT id FROM users WHERE LOWER(email) = LOWER(:email)"),
                {"email": DEFAULT_USER_EMAIL}
            )
            row = result.fetchone()
            if not row:
                raise ValueError(f"User not found: {DEFAULT_USER_EMAIL}")

            user_id = str(row[0])

            # Get papers for this user
            result = conn.execute(
                text("SELECT id, title FROM papers WHERE user_id = :user_id ORDER BY created_at DESC"),
                {"user_id": user_id}
            )
            papers = [(str(row[0]), row[1]) for row in result.fetchall()]

        return user_id, papers

    def _populate_user_and_papers(self, user_id: str, papers: List[Tuple[str, str]]):
        """Populate UI with loaded user and papers"""
        self.user_id = user_id
        self.papers = papers

        # Set user ID
        self.user_id_var.set(user_id)
        self.user_id_entry.config(state="readonly")

        # Populate papers listbox
        self.papers_listbox.delete(0, tk.END)
        for paper_id, title in papers:
            # Truncate long titles
            display_title = title[:70] + "..." if len(title) > 70 else title
            self.papers_listbox.insert(tk.END, display_title)

        self.papers_count_var.set(f"{len(papers)} papers available")
        self.progress_var.set("Ready")
        self.status_var.set(f"Loaded {len(papers)} papers for {DEFAULT_USER_EMAIL}")

    def _show_load_error(self, error_msg: str):
        """Show error when loading fails"""
        self.papers_count_var.set("Failed to load papers")
        self.progress_var.set("Error")
        self.status_var.set(f"Load error: {error_msg}")

    def _select_all_papers(self):
        """Select all papers in the listbox"""
        self.papers_listbox.select_set(0, tk.END)

    def _clear_paper_selection(self):
        """Clear paper selection"""
        self.papers_listbox.selection_clear(0, tk.END)

    def _refresh_papers(self):
        """Refresh papers from database"""
        self._load_user_and_papers()

    def _get_selected_paper_ids(self) -> Optional[List[str]]:
        """Get selected paper IDs from listbox"""
        selected_indices = self.papers_listbox.curselection()
        if not selected_indices:
            return None  # No selection = analyze all
        return [self.papers[i][0] for i in selected_indices]

    def _run_analysis(self):
        """Run the analysis in a background thread"""
        user_id = self.user_id_var.get().strip()
        if not user_id:
            messagebox.showerror("Error", "Please enter a User ID")
            return

        # Get category value
        category_display = self.category_combo.get()
        category = self.categories.get(category_display, "research_gaps")

        focus = self.focus_var.get().strip() or None

        # Get paper IDs from listbox selection
        paper_ids = self._get_selected_paper_ids()

        # Disable button and show progress
        self.analyze_btn.config(state="disabled")
        self.progress_var.set("Analyzing...")

        selected_count = len(paper_ids) if paper_ids else len(self.papers)
        self.status_var.set(f"Running {category} analysis on {selected_count} papers...")

        # Run in background thread
        thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(user_id, category, focus, paper_ids),
            daemon=True
        )
        thread.start()

    def _run_analysis_thread(self, user_id: str, category: str, focus: Optional[str], paper_ids: Optional[list]):
        """Background thread for running analysis"""
        try:
            result = asyncio.run(self._do_analysis(user_id, category, focus, paper_ids))
            self.root.after(0, lambda r=result: self._display_results(r))
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self._show_error(msg))

    async def _do_analysis(self, user_id: str, category: str, focus: Optional[str], paper_ids: Optional[list]) -> dict:
        """Perform the actual analysis"""
        from .config import settings
        settings.validate_for_analysis()  # Ensure OPENAI_API_KEY is set

        from .paper_insights import PaperInsightsService

        service = PaperInsightsService(paper_count=50)

        if category == "methodological_trends":
            return await service.analyze_methodological_trends(user_id=user_id, paper_ids=paper_ids)
        elif category == "mathematical_consistency":
            return await service.check_mathematical_consistency(user_id=user_id, focus_area=focus, paper_ids=paper_ids)
        elif category == "research_gaps":
            return await service.identify_research_gaps(user_id=user_id, research_area=focus, paper_ids=paper_ids)
        elif category == "contradictions":
            return await service.find_contradictions(user_id=user_id, topic=focus, paper_ids=paper_ids)
        elif category == "novel_connections":
            return await service.discover_novel_connections(user_id=user_id, theme=focus, paper_ids=paper_ids)
        elif category == "comprehensive":
            focus_areas = [focus] if focus else None
            return await service.generate_comprehensive_insights(user_id=user_id, focus_areas=focus_areas)
        else:
            raise ValueError(f"Unknown category: {category}")

    def _display_results(self, result: dict):
        """Display analysis results in the text widget"""
        self.results_text.delete(1.0, tk.END)

        # Format output
        lines = []
        lines.append("=" * 60)
        lines.append(f"  ANALYSIS: {result.get('category', 'Unknown').upper().replace('_', ' ')}")
        lines.append("=" * 60)
        lines.append("")

        # Focus/topic info
        for key in ["focus_area", "research_area", "topic", "theme"]:
            if key in result and result[key] != "general":
                lines.append(f"Focus: {result[key]}")

        lines.append(f"Papers Analyzed: {result.get('num_papers_analyzed', 'N/A')}")
        lines.append("-" * 60)
        lines.append("")

        # Main analysis
        if "analysis" in result:
            lines.append(result["analysis"])
            lines.append("")

        # Executive summary (for comprehensive)
        if "executive_summary" in result:
            lines.append("-" * 60)
            lines.append("EXECUTIVE SUMMARY")
            lines.append("-" * 60)
            lines.append(result["executive_summary"].get("summary", ""))
            lines.append("")

        # Sources
        if "sources" in result and result["sources"]:
            lines.append("-" * 60)
            lines.append("SOURCES")
            lines.append("-" * 60)
            for i, source in enumerate(result["sources"], 1):
                lines.append(f"{i}. {source.get('title', 'Unknown')}")
                lines.append(f"   Score: {source.get('score', 0):.3f}")
                if source.get('excerpt'):
                    excerpt = source['excerpt'][:150] + "..." if len(source.get('excerpt', '')) > 150 else source.get('excerpt', '')
                    lines.append(f"   Excerpt: {excerpt}")
                lines.append("")

        lines.append("=" * 60)

        self.results_text.insert(tk.END, "\n".join(lines))
        self.results_text.see(1.0)

        # Reset UI
        self.analyze_btn.config(state="normal")
        self.progress_var.set("Complete")
        self.status_var.set(f"Analysis complete - {result.get('num_papers_analyzed', 0)} papers analyzed")

    def _show_error(self, error_msg: str):
        """Display error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"ERROR:\n\n{error_msg}")

        self.analyze_btn.config(state="normal")
        self.progress_var.set("Error")
        self.status_var.set("Analysis failed - check configuration")

        messagebox.showerror("Analysis Error", error_msg)

    def _clear_results(self):
        """Clear the results text"""
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set("Ready")
        self.status_var.set("Results cleared")


def main():
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    app = InsightsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
