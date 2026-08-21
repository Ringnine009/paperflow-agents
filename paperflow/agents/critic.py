"""Critic agent: fact-checks the Reader's claims against the paper body.

For every claim with a quote, the Critic calls ``search_text`` on the full
text artifact (a cheap, local tool) to confirm the quote actually appears
and to fetch surrounding context before issuing a verdict:
supported / partially supported / unsupported / unverifiable.

The Critic is an *optional* agent: if it fails, the pipeline continues and
the Synthesizer notes the absence of fact-checking.
"""

from __future__ import annotations

from paperflow.agents.common import load_artifact_json
from paperflow.core.agent import Agent
from paperflow.core.jsonutil import json_dumps


class CriticAgent(Agent):
    name = "critic"
    title = "Critic"
    description = "Fact-checks claims against the paper text"
    requires: list[str] = ["reader"]
    critical = False  # graceful degradation: review proceeds without fact-checking

    def system_prompt(self) -> str:
        return """\
You are the Critic in a multi-agent paper-review team. Your job is to verify
the Reader's claims against the actual paper text.

For each claim:
1. If it has a quote, call search_text with `path` = FULL_TEXT_PATH and a
   short distinctive `query` taken from the quote, to confirm the quote
   appears in the paper and to read the surrounding context.
2. Verdicts: "supported" (quote found + context agrees), "partially
   supported" (quote found but overstated), "unsupported" (quote not found
   or context contradicts), "unverifiable" (cannot check).
3. Also list method limitations and suspicious points (small sample sizes,
   missing baselines, potential confounds, overclaimed numbers).

Output ONLY a JSON object (no prose, no code fences):
{
  "verdicts": [
    {"claim": "str", "verdict": "supported|partially supported|unsupported|unverifiable",
     "evidence_quote": "str", "note": "str"}
  ],
  "limitations": [{"issue": "str", "severity": "high|medium|low", "why": "str"}],
  "overall_assessment": "str"
}
"""

    def user_context(self, board) -> str:
        reader_output = load_artifact_json(board, "reader_output") or {}
        full_text_path = board.artifacts.get("full_text", {}).get("path", "")
        claims = reader_output.get("claims", [])
        return (
            "FULL_TEXT_PATH: " + str(full_text_path) + "\n"
            "CLAIMS TO VERIFY:\n" + json_dumps(claims)
        )

    def tools(self) -> list[dict]:
        return [s for s in self.registry.schemas() if s["function"]["name"] == "search_text"]

    def parse_output(self, content: str) -> dict:
        from paperflow.core.jsonutil import extract_json

        data = extract_json(content)
        return {
            "verdicts": data.get("verdicts", []),
            "limitations": data.get("limitations", []),
            "overall_assessment": data.get("overall_assessment", ""),
        }
