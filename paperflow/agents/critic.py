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

#: the deterministic statuses a claim can carry (see paperflow.tools.texttools)
_UNVERIFIED = "unverified"
_UNVERIFIABLE = "unverifiable"

_NO_ANNOTATION_REASON = (
    "no deterministic quote check was recorded for this claim (deterministic check)"
)
_NOT_FOUND_REASON = "quote not found in paper text (deterministic check)"


def _machine_verdict(claim: dict) -> tuple[str | None, str]:
    """Resolve a claim's machine status to (verdict, reason).

    Returns ``(None, "")`` when the claim *is* deterministically verified -
    the LLM's own verdict then stands. A claim with no annotation at all is
    ``unverifiable``, never silently accepted.
    """
    status = claim.get("quote_status")
    if status == "verified":
        return None, ""
    if status == _UNVERIFIABLE:
        reason = claim.get("quote_verification") or "the quote could not be checked"
        return _UNVERIFIABLE, f"{reason} (deterministic check)"
    if status == _UNVERIFIED:
        return _UNVERIFIED, _NOT_FOUND_REASON
    if claim.get("quote_verified") is True:  # legacy artifact: boolean only
        return None, ""
    if claim.get("quote_verified") is False:
        return _UNVERIFIED, _NOT_FOUND_REASON
    return _UNVERIFIABLE, _NO_ANNOTATION_REASON


class CriticAgent(Agent):
    name = "critic"
    title = "Critic"
    description = "Fact-checks claims against the paper text"
    requires: list[str] = ["reader"]
    critical = False  # graceful degradation: review proceeds without fact-checking
    # verifying one claim = one search_text round; give the loop headroom
    tool_rounds_override = 14

    def system_prompt(self) -> str:
        return """\
You are the Critic in a multi-agent paper-review team. Your job is to verify
the Reader's claims against the actual paper text.

Quote presence is ALREADY verified deterministically by code: every claim
carries `quote_verified` (true/false) and `quote_verification` (reason) from
a whitespace-normalized string search against the paper text. A claim whose
quote was not found will be downgraded to "unverified" by the system - you
do NOT need to (and should not) re-check quote existence with search_text.

Your job is the *semantic* part:
1. Verdicts (per claim, EVERY claim gets one):
   - "supported" - the claim follows from the quoted evidence
   - "partially supported" - the quote supports the claim but it is overstated
   - "unsupported" - the surrounding context contradicts the claim
   - "unverifiable" - there is no way to check (e.g. no full text available)
2. You may call search_text (path = FULL_TEXT_PATH) to read the CONTEXT
   around a claim you want to scrutinize for contradiction or overstatement -
   only when you actually need it; do not call it per claim by default.
3. Also list method limitations and suspicious points (small sample sizes,
   missing baselines, potential confounds, overclaimed numbers).

Output ONLY a JSON object (no prose, no code fences):
{
  "verdicts": [
    {"claim_index": 0, "claim": "str",
     "verdict": "supported|partially supported|unsupported|unverifiable",
     "evidence_quote": "str", "note": "str"}
  ],
  "limitations": [{"issue": "str", "severity": "high|medium|low", "why": "str"}],
  "overall_assessment": "str"
}

claim_index is MANDATORY for every verdict: the 0-based index of the claim
in the CLAIMS TO VERIFY list above. It lets the system align your verdict
with the right claim even when you paraphrase it.
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

    def after_run(self, board, output: dict) -> None:
        """Deterministic verdict resolution: downgrade quotes the code rejected.

        Quote presence is not an LLM opinion, so the machine status decides:

        * quote not found in the paper text       -> verdict ``unverified``
        * the check could not run (no full text)  -> verdict ``unverifiable``
        * no deterministic annotation at all      -> verdict ``unverifiable``

        The last case is the important one: a *missing* annotation used to be
        ignored entirely (``claim.get("quote_verified") is False`` never
        fired), so an unchecked claim kept whatever the LLM said. "We do not
        know" is now reported as "we do not know". The LLM's note is
        preserved and prefixed with the deterministic reason.

        Alignment is by ``claim_index`` first (the Critic is instructed to
        echo it) and falls back to exact claim wording - two independent LLM
        calls phrase the same claim differently, so wording alone is not a
        reliable key.
        """
        reader_output = load_artifact_json(board, "reader_output") or {}
        claims = reader_output.get("claims", [])
        by_text = {
            str(c.get("claim", "")).strip().lower(): c for c in claims
        }
        downgraded = 0
        uncheckable = 0
        for verdict in output.get("verdicts", []):
            claim = self._match_claim(verdict, claims, by_text)
            if claim is None:
                continue
            machine_verdict, reason = _machine_verdict(claim)
            if machine_verdict is None:
                continue
            note = f"{reason}; {verdict.get('note', '')}".strip(" ;")
            verdict["verdict"] = machine_verdict
            verdict["note"] = note
            verdict["deterministic"] = True
            downgraded += 1
            uncheckable += int(machine_verdict == "unverifiable")
        if downgraded:
            self.save_output(board, output)  # persist the corrected verdicts
            message = f"deterministic downgrade: {downgraded} claim(s) not verifiable by quote check"
            if uncheckable:
                message += f" ({uncheckable} unverifiable - no full text / no check was recorded)"
            board.add_log(message, agent=self.name)

    @staticmethod
    def _match_claim(verdict: dict, claims: list[dict], by_text: dict) -> dict | None:
        """Resolve a verdict to its reader claim: by claim_index, then wording."""
        index = verdict.get("claim_index")
        if isinstance(index, str) and index.strip().isdigit():
            index = int(index)
        if isinstance(index, int) and 0 <= index < len(claims):
            return claims[index]
        return by_text.get(str(verdict.get("claim", "")).strip().lower())
