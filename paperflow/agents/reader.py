"""Reader agent: parses the paper body into a structured analysis.

Turns raw full text into sections, key claims (each with a verbatim quote),
extracted data points, a method summary and open questions. It works from
the artifacts the Researcher placed on the board - no tools of its own.
"""

from __future__ import annotations

from paperflow.agents.common import load_artifact_json, load_artifact_text
from paperflow.core.agent import Agent
from paperflow.core.jsonutil import json_dumps
from paperflow.tools.texttools import verify_claims


class ReaderAgent(Agent):
    name = "reader"
    title = "Reader"
    description = "Parses the paper structure and extracts claims & data"
    requires: list[str] = ["researcher"]
    critical = True

    def system_prompt(self) -> str:
        return """\
You are the Reader in a multi-agent paper-review team. You read the paper
body provided to you and produce a structured analysis.

Rules:
- Every claim MUST include a short verbatim quote taken from the provided
  text (a distinctive phrase, 3-15 words). Never paraphrase inside quotes.
- data_points capture concrete numbers (metrics, scores, sizes) with context.
- If the full text is unavailable, base the analysis on the abstract and say
  so in `summary`.

Output ONLY a JSON object (no prose, no code fences):
{
  "title": "str",
  "summary": "str",
  "sections": [{"heading": "str", "content": "str"}],
  "claims": [{"claim": "str", "section": "str", "quote": "str", "evidence": "str"}],
  "data_points": [{"metric": "str", "value": "str", "context": "str"}],
  "method_summary": "str",
  "open_questions": ["str"]
}
"""

    def user_context(self, board) -> str:
        info = load_artifact_json(board, "researcher_output") or {}
        full_text = load_artifact_text(board, "full_text") or ""
        lines = [
            "PAPER TITLE: " + str(info.get("title", "?")),
            "AUTHORS: " + ", ".join(info.get("authors", []) or []),
            "ABSTRACT: " + str(info.get("abstract", "")),
            "",
            "FULL_TEXT_PATH: " + str(board.artifacts.get("full_text", {}).get("path", "")),
            "FULL TEXT (provided below; quotes must come from it):",
            full_text if full_text.strip() else "(full text unavailable - base analysis on the abstract)",
        ]
        return "\n".join(lines)

    def tools(self) -> list[dict]:
        return []

    def parse_output(self, content: str) -> dict:
        import json as _json

        from paperflow.core.jsonutil import extract_json

        data = extract_json(content)
        # normalize: guarantee the contract keys exist (lists default to [])
        return {
            "title": data.get("title", ""),
            "summary": data.get("summary", ""),
            "sections": data.get("sections", []),
            "claims": data.get("claims", []),
            "data_points": data.get("data_points", []),
            "method_summary": data.get("method_summary", ""),
            "open_questions": data.get("open_questions", []),
        }

    def after_run(self, board, output: dict) -> None:
        """Deterministically verify every claim's quote against the full text.

        Quote presence becomes a *code* guarantee: each claim is annotated
        with ``quote_verified`` (found / not found) and the annotated claims
        are persisted back onto the artifact the Critic and Synthesizer
        consume. A quote the code cannot find is later downgraded to
        "unverified" by the Critic regardless of what any LLM says.
        """
        full_text = load_artifact_text(board, "full_text")
        claims = output.get("claims") or []
        if not (full_text and claims):
            return
        verify_claims(full_text, claims)
        artifacts = board.artifacts_dir()
        artifacts.mkdir(parents=True, exist_ok=True)
        path = artifacts / f"{self.name}_output.json"
        path.write_text(json_dumps(output), encoding="utf-8")
        board.set_artifact(f"{self.name}_output", str(path), self.description)
        found = sum(1 for c in claims if c.get("quote_verified"))
        board.add_log(
            f"quote verification: {found}/{len(claims)} quotes found verbatim in full text",
            agent=self.name,
        )
