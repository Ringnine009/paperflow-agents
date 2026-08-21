"""Synthesizer agent: integrates everything into the final review report.

Takes the Researcher's metadata, the Reader's analysis and (when available)
the Critic's verdicts, searches for related work, and writes the final
structured Markdown review. The review is deliberately scaffolded with fixed
headings so reports are consistent across runs.
"""

from __future__ import annotations

from pathlib import Path

from paperflow.agents.common import load_artifact_json
from paperflow.config import RESEARCH_FOCUS
from paperflow.core.agent import Agent
from paperflow.core.jsonutil import json_dumps

REPORT_HEADINGS = (
    "Overview",
    "Method",
    "Key Claims & Evidence",
    "Strengths",
    "Limitations",
    "Related Work",
    "Relevance to My Research Direction",
    "Suggested Next Steps",
)


class SynthesizerAgent(Agent):
    name = "synthesizer"
    title = "Synthesizer"
    description = "Writes the final structured review report"
    requires: list[str] = ["researcher", "reader"]  # critic is optional
    critical = True

    def system_prompt(self) -> str:
        headings = "\n".join(f"## {h}" for h in REPORT_HEADINGS)
        return f"""\
You are the Synthesizer in a multi-agent paper-review team. You integrate the
Researcher's metadata, the Reader's analysis and the Critic's verdicts (when
present) into ONE final review written in Markdown.

Structure - include EVERY heading, in order:
{headings}

Requirements:
- "Key Claims & Evidence": one bullet per claim, with the verdict in bold
  (e.g. **[supported]**) and a short evidence note.
- "Related Work": call arxiv_search (1-3 queries is fine) to find 2-4
  genuinely related papers and link each with its arXiv id/title.
- "Relevance to My Research Direction": a **score /10** plus 2-4 sentences
  justifying it against this focus: {RESEARCH_FOCUS}
- If the Critic output is unavailable, say so honestly in Limitations.
- Be precise and critical; never invent numbers that were not provided.

Output ONLY the Markdown report (no extra commentary before or after).
"""

    def user_context(self, board) -> str:
        info = load_artifact_json(board, "researcher_output") or {}
        reader_output = load_artifact_json(board, "reader_output") or {}
        critic_output = load_artifact_json(board, "critic_output")
        lines = [
            "PAPER METADATA:",
            json_dumps({k: info.get(k) for k in ("title", "authors", "arxiv_id", "doi", "url", "published", "abstract")}),
            "",
            "READER ANALYSIS:",
            json_dumps({"summary": reader_output.get("summary"), "claims": reader_output.get("claims"),
                        "data_points": reader_output.get("data_points"),
                        "method_summary": reader_output.get("method_summary")}),
        ]
        if critic_output:
            lines += ["", "CRITIC VERDICTS:", json_dumps(critic_output)]
        else:
            lines += ["", "CRITIC OUTPUT: unavailable (fact-checking was skipped)"]
        lines.append("\nUse arxiv_search to gather related work links before writing.")
        return "\n".join(lines)

    def tools(self) -> list[dict]:
        return [s for s in self.registry.schemas() if s["function"]["name"] == "arxiv_search"]

    def parse_output(self, content: str) -> str:
        # the Synthesizer returns Markdown, not JSON
        text = content.strip()
        # drop LLM preambles ("Let me compose the final review...") - the
        # report must start at the first H1 heading
        heading = text.find("# ")
        if heading > 0:
            text = text[heading:].strip()
        return text

    def summarize(self, output: str) -> str:
        return output[:120].replace("\n", " ") + ("..." if len(output) > 120 else "")

    def save_output(self, board, output: str) -> Path:
        report_dir = board.path.parent
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "report.md"
        path.write_text(output, encoding="utf-8")
        board.set_report(str(path), output)
        return path
