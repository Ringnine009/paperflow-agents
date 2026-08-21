"""Researcher agent: locates the paper and gathers metadata + full text.

Entry-point handling (from the normalized InputSpec on the board):

==========  =====================================================
kind        what the agent does
==========  =====================================================
arxiv       arxiv_search(query="id:<id>") -> pdf_url -> fetch_pdf_text
doi         resolve_doi(doi) -> optional pdf link -> fetch_pdf_text
url         fetch_url(landing page) -> find PDF link -> fetch_pdf_text
pdf         read_pdf(local path)
title       arxiv_search(title) -> pick best hit -> fetch_pdf_text
==========  =====================================================

The agent's final JSON relays the TEXT_PATH of the extracted text; the
post-processing step (:meth:`after_run`) then materializes the (truncated)
full text as a board artifact so the Reader never has to go through the LLM
to obtain the paper body.
"""

from __future__ import annotations

from pathlib import Path

from paperflow.core.agent import Agent


class ResearcherAgent(Agent):
    name = "researcher"
    title = "Researcher"
    description = "Locates the paper and gathers metadata + full text"
    requires: list[str] = []
    critical = True

    TOOL_NAMES = ["arxiv_search", "resolve_doi", "fetch_url", "fetch_pdf_text", "read_pdf"]

    def system_prompt(self) -> str:
        return """\
You are the Researcher in a multi-agent paper-review team. Your job is to
identify the paper the user wants reviewed and gather (1) its metadata and
(2) its full text.

The user's entry is given to you as ENTRY_KIND / ENTRY_VALUE / ENTRY_RAW.
Handle each kind like this:
- arxiv  -> call arxiv_search with query "id:<id>" to get metadata + pdf_url
- doi    -> call resolve_doi(<doi>) to get metadata and, if open access, a pdf link
- url    -> call fetch_url on the page, then fetch_pdf_text on any direct PDF link you find
- pdf    -> call read_pdf on the local path
- title  -> call arxiv_search with the title, pick the best hit

To obtain the FULL TEXT:
- if you have a pdf_url, call fetch_pdf_text(pdf_url)
- for a local pdf entry, call read_pdf(path)
The tool result reports "TEXT_PATH: <path>" - your final answer must set
full_text_path to exactly that value. If no full text is obtainable, set
full_text_path to null and base the review on the abstract.

Output ONLY a JSON object (no prose, no code fences):
{
  "title": "str",
  "authors": ["str"],
  "abstract": "str",
  "doi": "str|null",
  "arxiv_id": "str|null",
  "url": "str|null",
  "published": "str|null",
  "source": "arxiv|doi|pdf|url|title",
  "full_text_path": "str|null",
  "full_text_chars": 0,
  "note": "str"
}
"""

    def user_context(self, board) -> str:
        spec = board.input
        lines = [
            f"ENTRY_KIND: {spec.kind}",
            f"ENTRY_VALUE: {spec.value}",
            f"ENTRY_RAW: {spec.raw}",
        ]
        if spec.pdf_override:
            lines.append(f"PDF_OVERRIDE: {spec.pdf_override} (call read_pdf on this path for full text)")
        return "\n".join(lines)

    def tools(self) -> list[dict]:
        return [s for s in self.registry.schemas() if s["function"]["name"] in self.TOOL_NAMES]

    def after_run(self, board, output) -> None:
        """Materialize the full text as a board artifact for downstream agents."""
        text_path = output.get("full_text_path")
        if text_path and Path(text_path).is_file():
            text = Path(text_path).read_text(encoding="utf-8", errors="replace")
            text = text[: self.settings.max_fulltext_chars]
            artifacts = board.artifacts_dir()
            artifacts.mkdir(parents=True, exist_ok=True)
            dest = artifacts / "full_text.txt"
            dest.write_text(text, encoding="utf-8")
            board.set_artifact("full_text", str(dest), "paper body text (truncated)")
            board.add_log(f"full text materialized ({len(text)} chars)", agent=self.name)
        elif output.get("abstract"):
            board.add_log("no full text available; review will rely on the abstract", level="warning", agent=self.name)
