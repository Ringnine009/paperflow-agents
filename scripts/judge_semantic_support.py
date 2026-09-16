"""The (secondary) LLM semantic-support judge for the three-arm experiment.

What it adds
------------
The project's deterministic verifier answers *"is this quote in the paper?"*.
It cannot answer *"does the quote support the claim?"* - that is a semantic
question. This module runs that second question under a declared protocol so
the experiment can report it honestly instead of implying the deterministic
check covered it.

Protocol (all of it is recorded in the output)
----------------------------------------------
* model: the same pinned model as the arms, ``temperature=0`` (the judge must
  not be the variance source it is measuring),
* prompt: fixed and versioned by :data:`PROMPT_VERSION`; the judge returns
  ``yes`` / ``no`` / ``unclear`` per item, with a one-line reason,
* items: the report's claim bullets. For arms B/C the bullet is also paired
  with the quote the Reader recorded for it; for arm A there is no quote
  artifact, so the judge is asked the attributable question instead ("does the
  paper support this?") and the column is labelled accordingly,
* batched (``BATCH`` items per call) to keep the cost negligible, which is
  also why the batch is a recorded parameter rather than an optimisation
  hidden in the code,
* **reliability is measured, not assumed**: every batch is judged twice with
  the two options swapped in order. Agreement between the two passes is
  reported as percent agreement and Cohen's kappa over the paired verdicts.

A judge that is inconsistent is worse than no judge, so an agreement figure
is part of the deliverable, not an optional extra.
"""

from __future__ import annotations

import json
import re

PROMPT_VERSION = "1.0"
BATCH = 10
VERDICTS = ("yes", "no", "unclear")

_SYSTEM_QUOTE = """\
You are a strict claim-verification judge for a paper-review audit.

For each item you receive:
  - CLAIM: a sentence taken from a review of a research paper,
  - QUOTE: the verbatim evidence the review attached to that claim.

Decide only this: does QUOTE, read literally, support CLAIM?

Rules:
- "yes" only when the quote on its own contains what the claim asserts.
- "no" when the quote does not contain it, or the claim states something the
  quote contradicts (a wrong number is a "no").
- "unclear" when the quote is empty, not a quote from the paper, or too
  fragmentary to decide.
- Do not reward a plausible-sounding claim, and do not use outside knowledge
  of the paper. Judge the pair you were given and nothing else.

Answer with JSON only, no prose and no code fences:
{"verdicts": [{"index": 0, "verdict": "yes|no|unclear", "reason": "<10 words max>"}]}
"""

_SYSTEM_PAPER = """\
You are a strict claim-attribution judge for a paper-review audit.

For each item you receive:
  - CLAIM: a sentence taken from a review of a research paper,
  - PAPER: the full text of the paper being reviewed.

Decide only this: does PAPER contain the specific factual content CLAIM
asserts - in particular its numbers and which configuration they belong to?

Rules:
- "yes" when the paper states what the claim states.
- "no" when the paper contradicts the claim, or the claim states a specific
  figure that appears nowhere in the paper.
- "unclear" when the claim is a judgement or opinion that the paper neither
  states nor contradicts.
- The review is not required to quote; you are checking attribution only.

Answer with JSON only, no prose and no code fences:
{"verdicts": [{"index": 0, "verdict": "yes|no|unclear", "reason": "<10 words max>"}]}
"""


def build_messages(items: list[dict], paper_text: str, *, quoted: bool) -> list[dict]:
    """The exact conversation sent to the judge for one batch."""
    system = _SYSTEM_QUOTE if quoted else _SYSTEM_PAPER
    lines = []
    for position, item in enumerate(items):
        lines.append(f"ITEM {position}")
        lines.append(f"CLAIM: {item['text']}")
        if quoted:
            lines.append(f"QUOTE: {item.get('quote') or '(none)'}")
        else:
            lines.append(f"PAPER: {paper_text}")
        lines.append("")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _read_payload(content: str) -> dict:
    """The judge's JSON object, tolerating fences and surrounding prose."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _entries(content: str) -> list[dict]:
    """The `verdicts` list, index-normalised, in whatever order it arrived."""
    data = _read_payload(content)
    entries = data.get("verdicts", []) or []
    return [entry for entry in entries if isinstance(entry, dict)]


def _normalised_index(entry: dict) -> int | None:
    index = entry.get("index")
    if isinstance(index, str) and index.strip().isdigit():
        index = int(index)
    return index if isinstance(index, int) else None


def parse_verdicts(content: str, expected: int) -> list[str]:
    """Read the judge's JSON verdicts; missing/invalid entries become "unclear".

    A judge that fails to answer must not be silently counted as agreement, so
    an unparseable batch is recorded as unclear rather than dropped.
    """
    verdicts = ["unclear"] * expected
    for entry in _entries(content):
        index = _normalised_index(entry)
        verdict = str(entry.get("verdict", "")).strip().lower()
        if index is not None and 0 <= index < expected and verdict in VERDICTS:
            verdicts[index] = verdict
    return verdicts


def parse_reasons(content: str, expected: int) -> list[str]:
    """The judge's one-line justification per item, for the audit trail.

    Kept because a verdict alone is not reviewable: the first live run stored
    no reasons, so its "no" verdicts could not be characterised afterwards.
    A missing reason is empty text - never invented, never back-filled.
    """
    reasons = [""] * expected
    for entry in _entries(content):
        index = _normalised_index(entry)
        if index is None or not (0 <= index < expected):
            continue
        reason = entry.get("reason")
        if reason is None:
            continue
        reasons[index] = str(reason).strip()
    return reasons


def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    """Cohen's kappa between two passes over the same items.

    Returns None when it is undefined (no items, or both passes constant and
    identical - the usual degeneracy, which is reported as 1.0 agreement with
    kappa undefined rather than as a fake number).
    """
    if not first or len(first) != len(second):
        return None
    labels = sorted(set(first) | set(second))
    n = len(first)
    observed = sum(1 for a, b in zip(first, second) if a == b) / n
    expected = sum((first.count(label) / n) * (second.count(label) / n) for label in labels)
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def percent_agreement(first: list[str], second: list[str]) -> float | None:
    if not first or len(first) != len(second):
        return None
    return round(sum(1 for a, b in zip(first, second) if a == b) / len(first), 4)


def summarise(items: list[dict]) -> dict:
    """Aggregate judged items into the arm-level column."""
    total = len(items)
    if not total:
        return {"n": 0, "yes": 0, "no": 0, "unclear": 0, "support_rate": None}
    counts = {verdict: sum(1 for i in items if i["verdict"] == verdict) for verdict in VERDICTS}
    decidable = counts["yes"] + counts["no"]
    return {
        "n": total,
        **counts,
        # denominator is every judged item: an "unclear" is not a pass
        "support_rate": round(counts["yes"] / total, 4),
        "support_rate_of_decidable": round(counts["yes"] / decidable, 4) if decidable else None,
    }
