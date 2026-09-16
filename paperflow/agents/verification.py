"""Machine-owned verification ledger for the final report.

Problem this solves: quote presence is decided deterministically, but the
report the human reads is written by an LLM. In the archived runs the
Synthesizer happily labelled claims **[supported]** that the code had just
downgraded - ``unverified`` appeared zero times across six such reports. A
deterministic decision that never reaches the deliverable is not a guarantee.

Two mechanisms, both owned by code:

1. :func:`inject_machine_verdicts` rewrites the report *after* the LLM wrote
   it: claim bullets that can be aligned to a machine-downgraded claim get
   their verdict marker replaced, and a ``## Verification Ledger`` section -
   generated entirely from the artifacts, never by the model - is appended.
2. :func:`check_report_consistency` re-reads the finished report and reports
   every place where its text disagrees with the machine decision. The
   pipeline logs the violations, so a disagreement is visible instead of
   silent.

Alignment between an LLM bullet and a reader claim is deliberately multi-key
(exact wording, quote fragment, token containment) because the Synthesizer
paraphrases: matching on exact wording alone linked 0 of 7 public examples.
"""

from __future__ import annotations

import re

from paperflow.agents.common import load_artifact_json
from paperflow.tools.texttools import canonical_tokens, normalize_ws

#: machine status vocabulary
VERIFIED = "verified"
UNVERIFIED = "unverified"  # the deterministic check ran and did not find the quote
UNVERIFIABLE = "unverifiable"  # the check could not run (no full text)
UNKNOWN = "unknown"  # no deterministic annotation on the claim at all

STATUS_MARKER = {
    VERIFIED: "[verified]",
    UNVERIFIED: "[unverified]",
    UNVERIFIABLE: "[unverifiable]",
    # unannotated legacy artifacts: nothing was ever checked, which is not the
    # same as "checked and failed" - and must certainly not read as verified
    UNKNOWN: "[unverifiable]",
}
#: markers that do not assert support
NON_SUPPORT_MARKERS = {"[unverified]", "[unverifiable]", "[unsupported]", "[unknown]"}

LEDGER_HEADING = "## Verification Ledger"
CLAIM_SECTION = "## Key Claims & Evidence"

#: a bold verdict marker ("**[supported]**") or a bare one ("[supported]")
_MARKER = re.compile(r"\*\*\[\s*(?P<bold>[A-Za-z][A-Za-z /_-]{1,24})\s*\]\*\*|\[\s*(?P<plain>[A-Za-z][A-Za-z /_-]{1,24})\s*\]")
_SECTION = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,3}[-*]\s+\S")


def _reader_claims(board) -> list[dict]:
    return (load_artifact_json(board, "reader_output") or {}).get("claims", []) or []


def _critic_verdicts(board) -> dict[int, dict]:
    verdicts = (load_artifact_json(board, "critic_output") or {}).get("verdicts", []) or []
    by_index: dict[int, dict] = {}
    by_text: dict[str, dict] = {}
    for verdict in verdicts:
        index = verdict.get("claim_index")
        if isinstance(index, str) and index.strip().isdigit():
            index = int(index)
        if isinstance(index, int):
            by_index[index] = verdict
        else:
            by_text[normalize_ws(str(verdict.get("claim", "")))] = verdict
    for index, claim in enumerate(_reader_claims(board)):
        if index not in by_index:
            match = by_text.get(normalize_ws(str(claim.get("claim", ""))))
            if match:
                by_index[index] = match
    return by_index


def claim_statuses(board) -> list[dict]:
    """Machine-owned status of every reader claim, joined with the verdict.

    Each row: ``claim_index``, ``claim``, ``quote``, ``status`` (one of
    ``verified`` / ``unverified`` / ``unverifiable`` / ``unknown``),
    ``reason``, ``marker``, ``verdict`` (the critic's verdict, which the
    downgrade may have overridden) and ``verdict_note``.
    """
    verdicts = _critic_verdicts(board)
    rows: list[dict] = []
    for index, claim in enumerate(_reader_claims(board)):
        status = _status_of(claim)
        verdict = verdicts.get(index) or {}
        rows.append(
            {
                "claim_index": index,
                "claim": str(claim.get("claim", "")),
                "quote": str(claim.get("quote", "")),
                "status": status,
                "reason": str(claim.get("quote_verification", "") or ""),
                "match_mode": claim.get("quote_match_mode"),
                "marker": STATUS_MARKER[status],
                "verdict": verdict.get("verdict"),
                "verdict_note": verdict.get("note"),
            }
        )
    return rows


def _status_of(claim: dict) -> str:
    """Map a claim's annotations to the machine status vocabulary.

    ``quote_status`` (explicit, written by the Reader) wins when present;
    otherwise the boolean ``quote_verified`` is used. A missing annotation is
    ``unknown`` - never silently treated as verified.
    """
    explicit = claim.get("quote_status")
    if explicit in (VERIFIED, UNVERIFIED, UNVERIFIABLE):
        return explicit
    verified = claim.get("quote_verified")
    if verified is True:
        return VERIFIED
    if verified is False:
        return UNVERIFIED
    return UNKNOWN


def counts(rows: list[dict]) -> dict:
    summary = {"total": len(rows), VERIFIED: 0, UNVERIFIED: 0, UNVERIFIABLE: 0, UNKNOWN: 0}
    for row in rows:
        summary[row["status"]] += 1
    summary["problems"] = summary["total"] - summary[VERIFIED]
    return summary


# ---------------------------------------------------------------------------
# report rewriting
# ---------------------------------------------------------------------------

def inject_machine_verdicts(board, markdown: str) -> tuple[str, dict]:
    """Rewrite `markdown` so the machine verdicts are visible, and report what changed.

    Returns ``(markdown, info)`` where ``info`` carries the counts, the number
    of repaired bullets and whether the ledger was added. Idempotent.
    """
    rows = claim_statuses(board)
    if not rows:
        return markdown, {"counts": counts(rows), "repaired": 0, "ledger_added": False}
    summary = counts(rows)

    body = _strip_ledger(markdown)
    body, repaired = _repair_bullets(body, rows)
    body = body.rstrip() + "\n\n" + _render_ledger(rows, summary)
    return body, {"counts": summary, "repaired": repaired, "ledger_added": True}


def _strip_ledger(markdown: str) -> str:
    """Drop a previously injected ledger so injection is idempotent."""
    lines = markdown.splitlines()
    kept: list[str] = []
    skipping = False
    for line in lines:
        if line.strip() == LEDGER_HEADING:
            skipping = True
            continue
        if skipping and _SECTION.match(line):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def _repair_bullets(markdown: str, rows: list[dict]) -> tuple[str, int]:
    """Replace the verdict marker of bullets that belong to a problem claim."""
    problems = {row["claim_index"]: row for row in rows if row["status"] != VERIFIED}
    if not problems:
        return markdown, 0

    lines = markdown.splitlines()
    start, end = _claim_section_bounds(lines)
    if start is None:
        return markdown, 0

    bullets = [i for i in range(start, end) if _BULLET.match(lines[i])]
    assigned = _align_bullets([lines[i] for i in bullets], rows)
    repaired = 0
    for position, index in enumerate(bullets):
        row = problems.get(assigned.get(position, -1))
        if row is None:
            continue
        lines[index], changed = _rewrite_marker(lines[index], row)
        repaired += int(changed)
    return "\n".join(lines), repaired


def _claim_section_bounds(lines: list[str]) -> tuple[int | None, int]:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == CLAIM_SECTION:
            start = i + 1
            continue
        if start is not None and _SECTION.match(line):
            return start, i
    return (start, len(lines)) if start is not None else (None, 0)


def _rewrite_marker(line: str, row: dict) -> tuple[str, bool]:
    marker = row["marker"]
    if f"**{marker}**" in line:
        return line, False  # already says what the machine says
    match = _MARKER.search(line)
    if match:
        return line[: match.start()] + f"**{marker}**" + line[match.end() :], True
    return line.rstrip() + f" **{marker}**", True


def _align_bullets(bullet_texts: list[str], rows: list[dict], threshold: float = 0.5) -> dict[int, int]:
    """Map bullet position -> claim index using wording/quote/token overlap."""
    pairs: list[tuple[float, int, int]] = []
    for position, text in enumerate(bullet_texts):
        for row in rows:
            score = _alignment_score(text, row)
            if score >= threshold:
                pairs.append((score, position, row["claim_index"]))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    taken_bullets: set[int] = set()
    taken_claims: set[int] = set()
    assigned: dict[int, int] = {}
    for _score, position, claim_index in pairs:
        if position in taken_bullets or claim_index in taken_claims:
            continue
        assigned[position] = claim_index
        taken_bullets.add(position)
        taken_claims.add(claim_index)
    return assigned


def _alignment_score(bullet: str, row: dict) -> float:
    text = normalize_ws(bullet)
    claim = normalize_ws(row["claim"])
    if claim and claim in text:
        return 1.0
    quote = normalize_ws(row["quote"])
    if len(quote) >= 20 and quote in text:
        return 0.9
    claim_tokens = set(canonical_tokens(row["claim"]))
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & set(canonical_tokens(bullet))) / len(claim_tokens)


def _render_ledger(rows: list[dict], summary: dict) -> str:
    lines = [
        LEDGER_HEADING,
        "",
        "Generated by `paperflow.agents.verification` from the run artifacts - this "
        "section is written by code, not by the model. Quote presence is decided by a "
        "deterministic string match of each claim's quote against the extracted paper text.",
        "",
        f"- Deterministic quote check: **{summary[VERIFIED]}/{summary['total']}** claims carry a "
        "quote that was located in the paper text.",
    ]
    if summary["problems"]:
        lines += [
            f"- **{summary['problems']} claim(s) could not be verified** and must not be read "
            "as supported evidence:",
            "",
        ]
        for row in rows:
            if row["status"] == VERIFIED:
                continue
            detail = row["reason"] or "no deterministic check was recorded for this claim"
            entry = (
                f"  - **{row['marker']}** claim #{row['claim_index']}: {row['claim']} "
                f"\n    - quote: \"{row['quote']}\""
                f"\n    - deterministic check: {detail}"
            )
            if row["verdict"]:
                entry += f"\n    - critic verdict: {row['verdict']}"
            if row["match_mode"]:
                entry += f" (match mode: {row['match_mode']})"
            lines.append(entry)
    else:
        lines.append("- No unverifiable claims: every claim's quote was located in the paper text.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# consistency check
# ---------------------------------------------------------------------------

def check_report_consistency(board, markdown: str) -> list[str]:
    """Report every disagreement between the report text and the machine decision.

    Returns a list of human-readable violations (empty = consistent). Called
    after the report is written; the pipeline logs whatever comes back.
    """
    rows = claim_statuses(board)
    if not rows:
        return []
    problems = [row for row in rows if row["status"] != VERIFIED]
    if not problems:
        return []

    violations: list[str] = []
    if LEDGER_HEADING not in markdown:
        violations.append(
            f"report.md has no '{LEDGER_HEADING}' section, so its {len(problems)} "
            "non-verified claim(s) are invisible to the reader"
        )
        ledger = ""
    else:
        ledger = markdown.split(LEDGER_HEADING, 1)[1]

    for row in problems:
        if normalize_ws(row["quote"]) and normalize_ws(row["quote"]) not in normalize_ws(ledger):
            violations.append(
                f"claim #{row['claim_index']} ({row['status']}) is missing from the verification ledger"
            )
        elif f"**{row['marker']}**" not in ledger:
            violations.append(
                f"claim #{row['claim_index']} is listed in the ledger without its "
                f"{row['marker']} marker"
            )

    lines = markdown.splitlines()
    start, end = _claim_section_bounds(lines)
    if start is not None:
        bullets = [i for i in range(start, end) if _BULLET.match(lines[i])]
        assigned = _align_bullets([lines[i] for i in bullets], rows)
        for position, index in enumerate(bullets):
            row = next((r for r in problems if r["claim_index"] == assigned.get(position, -1)), None)
            if row is None:
                continue
            match = _MARKER.search(lines[index])
            label = (match.group("bold") or match.group("plain")) if match else None
            marker = f"[{label}]" if label else None
            if marker is None:
                violations.append(
                    f"bullet for claim #{row['claim_index']} carries no verdict marker although "
                    f"the machine check says {row['status']}"
                )
            elif marker not in NON_SUPPORT_MARKERS:
                violations.append(
                    f"bullet for claim #{row['claim_index']} still claims {marker} although the "
                    f"machine check says {row['status']}"
                )
    return violations
