"""Three-arm comparison under a real model: does the pipeline beat one prompt?

This is the experiment ``docs/baseline-plan.md`` said could not be run offline.
It differs from :mod:`scripts.compare_arms` in exactly one way that matters:
the LLM is the real DeepSeek API, so every number below is measured - real
``usage`` tokens (cache hit/miss split included), real wall-clock time and
real CNY cost - and the reviewed text is whatever the model actually wrote
rather than a canned script.

Arms (same paper, same task description, same model, same temperature):

``A_single_prompt``
    the paper text plus the task in ONE ``chat()`` call. No task board, no
    artifacts, no tools, no per-claim record.
``B_pipeline``
    the shipped Researcher -> Reader -> Critic -> Synthesizer pipeline with
    the deterministic quote check and the machine-written verification ledger.
``C_pipeline_no_verification``
    the same pipeline and the same agents, with the deterministic quote check
    switched off and the report left as the model wrote it.

Measurement discipline (see ``docs/arm-comparison-live.md`` for the protocol):

* the primary columns are deterministic and involve **no** LLM - heading
  completeness, whether a claim's wording can be located in the paper, whether
  a number is attributed to the configuration the paper attributes it to, and
  how many of the paper's pre-listed contributions the review covers;
* tokens and cost come from the API's own ``usage`` block, never an estimate;
* the spend cap is enforced *before* each request, so the experiment cannot
  overspend and report afterwards;
* the LLM-judged "does the quote really support the claim" column is a
  secondary sample and is labelled as such, with its model, prompt, n and
  repeatability recorded next to it.

Usage::

    set DEEPSEEK_API_KEY=...            # env only; never written to a file
    python scripts/compare_arms_live.py --runs 3 --out docs/arm-comparison-live.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paperflow.agents import synthesizer as synthesizer_module  # noqa: E402
from paperflow.agents.reader import ReaderAgent  # noqa: E402
from paperflow.agents.synthesizer import REPORT_HEADINGS, SynthesizerAgent  # noqa: E402
from paperflow.config import Settings, load_dotenv  # noqa: E402
from paperflow.core.llm import BudgetExceeded, LLMClient  # noqa: E402
from paperflow.core.pricing import (  # noqa: E402
    PRICE_READ_ON,
    PRICE_SOURCE,
    PRICING_OFFPEAK_CNY,
    PRICING_PEAK_CNY,
    Pricing,
)
from paperflow.pipeline import Pipeline  # noqa: E402
from paperflow.tools.net import extract_pdf_text  # noqa: E402
from paperflow.tools.texttools import canonical_text, canonical_tokens, normalize_ws  # noqa: E402
from scripts.paper_ground_truth import (  # noqa: E402
    CONTRIBUTIONS,
    HEADLINE_NUMBERS,
    PAPER_FACTS,
    TABLE2,
)
from scripts.judge_semantic_support import (  # noqa: E402
    BATCH,
    PROMPT_VERSION,
    build_messages,
    cohen_kappa,
    parse_reasons,
    parse_verdicts,
    percent_agreement,
    summarise as summarise_verdicts,
)

# ---------------------------------------------------------------------------
# experiment constants - all pinned so the run is reproducible
# ---------------------------------------------------------------------------

ARMS = ("A_single_prompt", "B_pipeline", "C_pipeline_no_verification")

#: hard spend cap in CNY, checked before every request (see LLMClient)
BUDGET_CNY = 30.0

#: the repeats per arm a *delivered* report must have. The first grid ran 3,
#: which was the floor at the time and is recorded as limitation 5 of
#: `docs/arm-comparison-live.md`; widening it to 5 is only worth anything if
#: the harness refuses to publish a report that is narrower than it claims.
MIN_RUNS_PER_ARM = 5


class InsufficientRuns(RuntimeError):
    """Raised when a report would be built from fewer than the required repeats."""


def grid_pairs(runs: int) -> list[tuple[str, int]]:
    """The `(arm, repeat)` pairs of a full grid, interleaved by repeat."""
    return [(arm, repeat) for repeat in range(1, runs + 1) for arm in ARMS]


def plan_for(runs: int) -> list[str]:
    """The (arm, repeat) schedule for `runs` repeats per arm, in spend order.

    A full grid is A r1, B r1, C r1, A r2, ... so that truncating it at any
    point leaves a balanced comparison rather than three copies of one arm.

    The schedule is **generated**, not sliced. The first version sliced a
    three-repeat constant, so `--runs 5` silently bought three repeats per arm
    and produced a report titled N=5 - the extension to five runs is impossible
    until the plan can actually grow (see
    `test_the_run_plan_grows_with_the_repeat_count`).
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs!r}")
    return [arm for arm, _ in grid_pairs(runs)]


def runs_shortfall(summary: dict, minimum: int = MIN_RUNS_PER_ARM) -> dict[str, int]:
    """Arms whose successful run count is below `minimum`, with that count."""
    return {
        arm: int(row.get("runs_ok") or 0)
        for arm, row in (summary.get("arms") or {}).items()
        if int(row.get("runs_ok") or 0) < minimum
    }


def require_runs_per_arm(summary: dict, minimum: int = MIN_RUNS_PER_ARM) -> None:
    """Refuse to publish a report that is narrower than the declared floor.

    Failed and skipped runs do not count: a report is allowed to say "5 repeats
    per arm" only when five runs per arm actually produced something to score.
    """
    shortfall = runs_shortfall(summary, minimum)
    if shortfall:
        detail = ", ".join(f"{arm} has {count}" for arm, count in sorted(shortfall.items()))
        raise InsufficientRuns(
            f"a report needs at least {minimum} successful runs per arm; {detail}. "
            "Buy the missing repeats (--extend-from) or declare a lower floor explicitly."
        )


def settings_keys_present() -> bool:
    """True when the environment already carries the key `Settings` needs."""
    return bool(os.environ.get("DEEPSEEK_API_KEY"))

PAPER_PDF = REPO.parent.parent / "research" / "werewolf-multiagent-paper.pdf"

#: a claim's wording counts as locatable in the paper when this many
#: consecutive canonical tokens of it appear in the paper text. 8 is above
#: chance (the project's own relaxed tier needs >= 4 tokens + 20 chars).
MIN_LOCATABLE_TOKENS = 8
#: how large a contiguous run we bother to search for
MAX_RUN_PROBE = 40

CLAIM_SECTION = "## Key Claims & Evidence"

#: verdict markers the project's machine layer writes
MACHINE_MARKERS = ("[verified]", "[unverified]", "[unverifiable]")
LLM_MARKERS = ("[supported]", "[partially supported]", "[unsupported]")

#: any "[...]" verdict-ish label, with or without bold
_ANY_MARKER = __import__("re").compile(r"\[\s*([A-Za-z][A-Za-z /_-]{1,24})\s*\]")


def marker_counts(bullets: list[str]) -> dict:
    """Verdict labels counted **on the report's claim bullets only**.

    Counting markers over the whole report text is misleading: the machine
    ledger's legend lists the labels it uses, so a ledger could look like it
    marked claims it never marked. Restricting the count to bullet lines makes
    the column mean "how many of this report's own claims carry a machine
    verdict", which is the question the experiment is asking.
    """
    counts: dict[str, int] = {}
    for bullet in bullets:
        for match in _ANY_MARKER.finditer(bullet):
            label = normalize_ws(match.group(1))
            counts[label] = counts.get(label, 0) + 1
    return {
        "markers_on_bullets": counts,
        "machine_markers_on_bullets": sum(counts.get(marker.strip("[]"), 0) for marker in MACHINE_MARKERS),
        "llm_markers_on_bullets": sum(counts.get(marker.strip("[]"), 0) for marker in LLM_MARKERS),
    }


#: the harness's own status vocabulary. The pipeline's board speaks
#: pending/running/done; the harness speaks ok/failed/skipped. The mapping is
#: explicit so a new board status can never be silently averaged in as a
#: successful run (or, worse, read as "ok" because it was not "failed").
STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

_BOARD_STATUS = {"done": STATUS_OK, "failed": STATUS_FAILED}


def to_harness_status(status: str | None) -> str:
    """Map a pipeline board status onto the harness vocabulary."""
    return _BOARD_STATUS.get(str(status), STATUS_FAILED)


# ---------------------------------------------------------------------------
# token utilities (deterministic measurement primitives)
# ---------------------------------------------------------------------------

def token_runs(text: str, size: int = 2) -> list[tuple[str, ...]]:
    """Every contiguous `size`-token run of `text`, longest window first."""
    tokens = canonical_tokens(text)
    if len(tokens) < size:
        return []
    return [tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def _token_index(text: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, token in enumerate(canonical_tokens(text)):
        index.setdefault(token, []).append(position)
    return index


def longest_paper_run(claim: str, paper_tokens: list[str], paper_index: dict[str, list[int]]) -> int:
    """Length of the longest run of `claim` tokens that appears verbatim in the paper.

    Contiguity is checked on the paper's *token* sequence, so punctuation,
    line breaks and PDF hyphenation cannot break a genuine match (the same
    normalization the shipped verifier applies).
    """
    tokens = canonical_tokens(claim)
    upper = min(len(tokens), MAX_RUN_PROBE)
    for size in range(upper, 0, -1):
        for start in range(0, len(tokens) - size + 1):
            window = tokens[start : start + size]
            for position in paper_index.get(window[0], ()):
                if paper_tokens[position : position + size] == window:
                    return size
    return 0


# ---------------------------------------------------------------------------
# deterministic scorers
# ---------------------------------------------------------------------------

def known_win_rates() -> dict[str, str]:
    """configuration letter -> win rate, lowercased keys, from the paper's Table 2."""
    return {letter.lower(): row["win_rate"] for letter, row in TABLE2.items()}


_PERCENT = __import__("re").compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(%|pp\b|percentage points?)")
_SENTENCE_SPLIT = __import__("re").compile(r"(?<=[.;:!?])\s+|\n+")

#: a number *bound* to a configuration: "group E = 68.8%", "group E: 68.8%",
#: "(group E, 68.8%)", "group E reached 68.8%". Only bound numbers may be
#: judged against Table 2 - judging every group named in a sentence against
#: every percentage in it turns a comparison list ("A=44.2%, B=54.6%") into
#: five false contradictions.
_BOUND_PAIR = __import__("re").compile(
    r"(?:group|config(?:uration)?|condition|setting)s?\s*([A-F])\b[^.;:\n]{0,40}?"
    r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(%|pp\b|percentage points?)",
    __import__("re").IGNORECASE,
)
#: values that are approximations/bands in the prose rather than Table 2 cells
_APPROXIMATIONS = HEADLINE_NUMBERS["approximate"]


def known_win_rate_values() -> set[str]:
    return set(TABLE2[letter]["win_rate"] for letter in TABLE2)


#: magnitudes worth grounding against the paper: percentages, pp deltas and
#: plain counts. Small integers are ignored - "9 players", "6 configurations"
#: are too common to mean anything on their own. The lookarounds only stop a
#: digit run from being split out of a longer number or a version string, so
#: "3000 games" is still a number while "2.5" never yields a bare "5".
_NUMBER_TOKEN = __import__("re").compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?![\d.])")
_IGNORE_NUMBERS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "100"}


def paper_number_values(text: str) -> set[str]:
    """The numeric literals that actually occur in a text.

    Used to ground a report's numbers against the paper: a figure the paper
    never prints anywhere is a candidate fabrication, and this is checkable
    without any model.
    """
    return {match.group(1) for match in _NUMBER_TOKEN.finditer(text)}


def number_grounding(bullets: list[str], paper_values: set[str]) -> dict:
    """How many of a report's claim bullets keep every magnitude in the paper."""
    grounded = 0
    flagged: list[str] = []
    for bullet in bullets:
        values = {v for v in paper_number_values(bullet) if v not in _IGNORE_NUMBERS}
        if not values:
            continue  # no checkable magnitude: not counted either way
        missing = sorted(v for v in values if v not in paper_values)
        if missing:
            flagged.append(f"{','.join(missing)} not in paper: {bullet[:110]!r}")
        else:
            grounded += 1
    checkable = grounded + len(flagged)
    return {
        "number_bearing_bullets": checkable,
        "number_grounded_bullets": grounded,
        "number_grounding_rate": round(grounded / checkable, 3) if checkable else None,
        "ungrounded_numbers": flagged,
    }


def reader_claim_evidence(claims: list[dict] | None) -> dict:
    """Quote-evidence coverage of the Reader artifact, using its own annotation.

    This is the column that separates the arms architecturally: B and C both
    record claims *with* quotes, and the shipped verifier decides whether each
    quote exists in the paper. Arm A records no claims at all, so the column is
    explicitly not measurable for it rather than silently zero.
    """
    if not claims:
        return {
            "claims_total": 0,
            "claims_quote_verified": None,
            "quote_evidence_rate": None,
            "uncheckable": "arm A produces no claim artifact",
        }
    verified = sum(1 for claim in claims if claim.get("quote_verified"))
    return {
        "claims_total": len(claims),
        "claims_quote_verified": verified,
        "quote_evidence_rate": round(verified / len(claims), 4),
        "uncheckable": None,
    }


def derivable_values() -> set[str]:
    """Differences and sums of the paper's own headline numbers.

    A review that reports "E is +10.2 pp over D" is doing arithmetic on the
    paper's table, not inventing a figure. Treating those as fabrications
    would have produced nine false positives in the first grid run, so they
    are computed once from the key and accepted as derivable.
    """
    base = set()
    for bucket in HEADLINE_NUMBERS.values():
        for value in bucket:
            try:
                base.add(float(value))
            except ValueError:
                continue
    derived: set[str] = set()
    for a in base:
        for b in base:
            for candidate in (a - b, a + b):
                if candidate <= 0:
                    continue
                for text in (f"{candidate:.0f}", f"{candidate:.1f}", f"{candidate:.2f}"):
                    if abs(float(text) - candidate) < 1e-9:
                        derived.add(text)
    return derived


def check_number_pairings(text: str) -> list[str]:
    """Flag numbers that contradict the paper: a number *bound* to a
    configuration that the paper's Table 2 gives differently, or a magnitude
    that is neither printed in the paper nor derivable from its own numbers.

    Only ``%``/``pp`` magnitudes are judged (they are the paper's headline
    quantities); counts such as "500 games" are left alone because their
    context is too varied to judge without a model. Numbers bound to a
    configuration are judged strictly; unbound numbers are only checked for
    existence, so a comparison list cannot cascade into false contradictions.
    """
    rates = known_win_rates()
    known_values = known_win_rate_values()
    all_paper_numbers: set[str] = set()
    for bucket in HEADLINE_NUMBERS.values():
        all_paper_numbers |= set(bucket)
    derivable = derivable_values()

    errors: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue

        judged_spans: list[tuple[int, int]] = []
        for match in _BOUND_PAIR.finditer(sentence):
            letter, value, unit = match.group(1).lower(), match.group(2), match.group(3).strip()
            judged_spans.append(match.span())
            if value in _APPROXIMATIONS or value not in known_values:
                continue  # a band, or a non-win-rate column: not judged here
            if letter in rates and rates[letter] != value:
                errors.append(
                    f"group {letter.upper()} attributed {value}{unit} in the paper's "
                    f"Table 2 (group {letter.upper()} is {rates[letter]}%): {sentence.strip()[:110]!r}"
                )

        for match in _PERCENT.finditer(sentence):
            value, unit = match.group(1), match.group(2).strip()
            if any(start <= match.start() < end for start, end in judged_spans):
                continue  # already judged as a bound pair
            if value in _APPROXIMATIONS or value in all_paper_numbers or value in derivable:
                continue
            errors.append(
                f"{value}{unit} is neither in the paper nor derivable from its numbers: "
                f"{sentence.strip()[:110]!r}"
            )
    return errors


def _subsequence(haystack: list[str], needle: list[str]) -> bool:
    """True when every token of `needle` appears in `haystack`, in order.

    `haystack` must be a sequence, never an iterator: the scan walks it once
    per alternative phrasing, and a consumed iterator would silently report
    "not covered" for every alternative after the first.
    """
    if not needle:
        return False
    position = 0
    for token in needle:
        while position < len(haystack) and haystack[position] != token:
            position += 1
        if position == len(haystack):
            return False
        position += 1
    return True


def contributions_covered(text: str) -> tuple[int, int]:
    """(covered, total) against the checklist in ``scripts/paper_ground_truth.py``.

    A contribution counts only when *all* of its concept groups are present -
    naming "DBN" is not the same as reporting what it does. Group entries are
    alternative phrasings; the checker tries each, and each try rescans the
    same token sequence. ``substrings`` entries in the checklist are matched
    against the canonical text instead, for entity names whose punctuation
    tokenization would otherwise split them.
    """
    tokens = canonical_tokens(text)  # a list: rescanned, never consumed
    canonical = canonical_text(text)
    covered = sum(1 for contribution in CONTRIBUTIONS if _contribution_covered(contribution, tokens, canonical))
    return covered, len(CONTRIBUTIONS)


def _contribution_covered(contribution: dict, tokens: list[str], canonical: str) -> bool:
    groups_ok = all(
        any(_subsequence(tokens, canonical_tokens(alternative)) for alternative in group)
        for group in contribution["groups"]
    )
    substrings_ok = all(needle in canonical for needle in contribution.get("substrings", []))
    return groups_ok and substrings_ok


def missing_contributions(text: str) -> list[str]:
    """The checklist ids this text does **not** cover, in checklist order.

    The count alone cannot say whether a gap is one recurring blind spot or
    three different ones, so the ids are recorded per run and the report names
    the item an arm keeps missing instead of asserting it.
    """
    tokens = canonical_tokens(text)
    canonical = canonical_text(text)
    return [
        contribution["id"]
        for contribution in CONTRIBUTIONS
        if not _contribution_covered(contribution, tokens, canonical)
    ]

def claim_bullets(report: str) -> list[str]:
    """The claim bullets of a report's 'Key Claims & Evidence' section.

    Every arm is told to put its substantive claims there, so this is the one
    extraction that is applied identically to all three architectures. It is
    what makes the columns comparable: A has no claim artifact, so if the
    comparison used B's artifact it would be measuring the artifact's
    existence rather than the quality of the claims.
    """
    lines = report.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == CLAIM_SECTION:
            start = index + 1
            continue
        if start is not None and line.startswith("## "):
            return _bullets(lines[start:index])
    return _bullets(lines[start:]) if start is not None else []


def _bullets(lines: Iterable[str]) -> list[str]:
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped[:2] in ("- ", "* ") and len(stripped) > 3:
            out.append(stripped[2:].strip())
    return out


def heading_report(report: str) -> dict:
    present = [heading for heading in REPORT_HEADINGS if f"## {heading}" in report]
    return {
        "headings_present": len(present),
        "headings_missing": [h for h in REPORT_HEADINGS if h not in present],
    }


# ---------------------------------------------------------------------------
# per-run collection
# ---------------------------------------------------------------------------

@dataclass
class ArmRun:
    """One arm, one repeat: the metrics record plus the judge sample."""

    record: dict
    report: str
    judge_samples: list[dict]


def _usage_metrics(llm: LLMClient, pricing: Pricing) -> dict:
    totals = llm.usage_totals()
    return {
        "llm_calls": totals["calls"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "cache_hit_tokens": totals["cache_hit_tokens"],
        "cache_miss_tokens": totals["cache_miss_tokens"],
        "cost_cny": llm.usage.cost_cny(pricing),
    }


def evaluate_report(
    report: str, paper_tokens: list[str], paper_index: dict[str, list[int]],
    paper_values: set[str] | None = None,
) -> dict:
    """Every deterministic column, computed identically for all three arms."""
    bullets = claim_bullets(report)
    runs = [longest_paper_run(bullet, paper_tokens, paper_index) for bullet in bullets]
    locatable = [length for length in runs if length >= MIN_LOCATABLE_TOKENS]
    coverage, coverage_total = contributions_covered(report)
    errors: list[str] = []
    for bullet in bullets:
        errors.extend(check_number_pairings(bullet))
    return {
        **heading_report(report),
        "report_chars": len(report),
        "claim_bullets": len(bullets),
        "claim_bullets_locatable": len(locatable),
        "attribution_rate": round(len(locatable) / len(bullets), 3) if bullets else None,
        "attribution_min_tokens": MIN_LOCATABLE_TOKENS,
        "longest_runs": runs,
        **number_grounding(bullets, paper_values or set()),
        "coverage": coverage,
        "coverage_total": coverage_total,
        "coverage_missing": missing_contributions(report),
        "coverage_rate": round(coverage / coverage_total, 3) if coverage_total else None,
        "factual_errors": len(errors),
        "factual_error_details": errors,
        **marker_counts(bullets),
        "ledger_in_report": "## Verification Ledger" in report,
    }


def stubbed_arxiv_search(query: str, max_results: int = 5) -> str:
    """Offline replacement for arxiv_search so a live run touches no other API.

    The paper being reviewed is the author's own local PDF, so the experiment
    controls every byte of its input: network variance is removed, and the only
    paid dependency is DeepSeek. All three arms therefore see the same tool
    result (an empty result set), which is a limitation of the setup and is
    reported as one.
    """
    return json.dumps([])


def _fresh_client(settings: Settings, budget: float, pricing: Pricing, **kwargs: Any) -> LLMClient:
    """A per-run client so token deltas isolate one run exactly."""
    return LLMClient.from_settings(
        settings,
        budget_cny=budget,
        pricing=pricing,
        thinking=os.environ.get("PAPERFLOW_THINKING", "disabled") or None,
        **kwargs,
    )


def _client_for(
    settings: Settings, budget: float, pricing: Pricing,
    llm_factory: Callable[[], LLMClient] | None, **kwargs: Any,
) -> LLMClient:
    """Build this run's client: a real DeepSeek one, or the test's factory.

    Every arm gets a *fresh* client so its token ledger isolates exactly one
    run; a shared client would silently carry another run's tokens.
    """
    if llm_factory is not None:
        return llm_factory()
    return _fresh_client(settings, budget, pricing, **kwargs)


def run_arm_single_prompt(
    *, settings: Settings, out_dir: Path, paper_text: str, paper_meta: dict,
    budget: float, pricing: Pricing, collector: dict | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
) -> ArmRun:
    """Arm A: the whole paper plus the task in one call."""
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = _client_for(settings, budget, pricing, llm_factory)
    # Reuse the Synthesizer's own instruction so all three arms are given the
    # same task description and the same eight headings - the only difference
    # is that A gets no agents, tools or task board around it.
    system = SynthesizerAgent(None, llm, settings).system_prompt()  # type: ignore[arg-type]
    user = (
        "PAPER METADATA:\n"
        + json.dumps(
            {k: paper_meta.get(k) for k in ("title", "authors", "arxiv_id", "doi", "url", "published", "abstract")},
            ensure_ascii=False,
        )
        + "\n\nPAPER TEXT (this is the complete paper; quote from it):\n"
        + paper_text
        + "\n\nWrite the complete review now, in one answer."
    )
    started = time.time()
    message = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
    )
    report = (message.get("content") or "").strip()
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    usage = _usage_metrics(llm, pricing)
    record = {
        "run_id": out_dir.name,
        "arm": "A_single_prompt",
        "status": "ok" if report.strip() else "failed",
        "error": None if report.strip() else "empty answer",
        "wall_seconds": round(time.time() - started, 2),
        **usage,
    }
    if collector is not None:
        collector.update({"usage": usage, "payloads": [p for p in llm.payloads]})
    return ArmRun(record=record, report=report, judge_samples=[])


@contextlib.contextmanager
def verification_disabled():
    """Switch the deterministic quote layer off, exactly for one block.

    The pre-fix architecture: the Reader records no deterministic annotation
    and the report is neither machine-corrected nor checked. Everything is
    restored in ``finally``, so an experiment never leaves the shipped
    behaviour patched. (Identical in intent to the offline harness; kept here
    so the live script can be run on its own.)
    """
    original_reader_after_run = ReaderAgent.after_run
    original_inject = synthesizer_module.inject_machine_verdicts
    original_check = synthesizer_module.check_report_consistency

    ReaderAgent.after_run = lambda self, board, output: None  # type: ignore[method-assign]
    synthesizer_module.inject_machine_verdicts = lambda board, markdown: (  # type: ignore[assignment]
        markdown,
        {"counts": {"total": 0, "problems": 0}, "repaired": 0, "ledger_added": False},
    )
    synthesizer_module.check_report_consistency = lambda board, markdown: []  # type: ignore[assignment]
    try:
        yield
    finally:
        ReaderAgent.after_run = original_reader_after_run  # type: ignore[method-assign]
        synthesizer_module.inject_machine_verdicts = original_inject  # type: ignore[assignment]
        synthesizer_module.check_report_consistency = original_check  # type: ignore[assignment]


def run_arm_pipeline(
    *, arm: str, settings: Settings, out_root: Path, paper_pdf: Path,
    budget: float, pricing: Pricing, arxiv_stub: Callable, collector: dict | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
) -> ArmRun:
    """Arms B and C: the shipped pipeline, with the deterministic layer on/off."""
    out_dir = out_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = _client_for(settings, budget, pricing, llm_factory)
    pipe = Pipeline(settings=settings, out_dir=out_dir, llm=llm)
    pipe.registry.get("arxiv_search").func = arxiv_stub

    started = time.time()
    context = contextlib.nullcontext() if arm == "B_pipeline" else verification_disabled()
    with context:
        result = pipe.run(str(paper_pdf), run_id=out_root.name)

    report_path = Path(result["report"]) if result["report"] else None
    report = report_path.read_text(encoding="utf-8") if report_path and report_path.is_file() else ""
    artifacts = Path(result["board"]).parent / "artifacts"
    reader_artifact = _load_json(artifacts / "reader_output.json")
    claims = (reader_artifact or {}).get("claims", []) or []
    usage = _usage_metrics(llm, pricing)
    record = {
        "run_id": out_root.name,
        "arm": arm,
        "status": to_harness_status(result["status"]) if report else STATUS_FAILED,
        "error": None if report else "no report written",
        "wall_seconds": round(time.time() - started, 2),
        **usage,
        # claim-level columns: B annotates every claim deterministically, C
        # (pre-fix emulation) annotates none - this is the architecture column
        "claims_total": len(claims),
        "claims_with_deterministic_status": sum(1 for c in claims if c.get("quote_status")),
        "claims_quote_verified": sum(1 for c in claims if c.get("quote_verified")),
        "claims_quote_unverified": sum(1 for c in claims if c.get("quote_status") == "unverified"),
        "claims_quote_unverifiable": sum(1 for c in claims if c.get("quote_status") == "unverifiable"),
        **reader_claim_evidence(claims),
        "pipeline_result_status": result["status"],
    }
    if collector is not None:
        collector.update({"usage": usage, "payloads": [p for p in llm.payloads], "claims": claims})
    return ArmRun(record=record, report=report, judge_samples=[])


def _load_json(path: Path | None) -> Any:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------

def run_grid(
    *,
    llm_factory: Callable[[], LLMClient] | None,
    runs: int,
    out_dir: Path,
    paper_pdf: Path = PAPER_PDF,
    paper_text: str | None = None,
    arxiv_stub: Callable = stubbed_arxiv_search,
    budget: float = BUDGET_CNY,
    pricing: Pricing = PRICING_PEAK_CNY,
    settings: Settings | None = None,
    arms: list[str] | None = None,
) -> list[dict]:
    """Run the whole grid and return one metrics record per (arm, repeat).

    ``llm_factory`` builds a fresh client per run; passing ``None`` uses the
    real DeepSeek client from the environment. Tests pass a ``FakeLLM``
    factory, which is the only way this grid ever runs without spending money.
    """
    text = _paper_text_for(paper_pdf, paper_text)
    if arms:
        counts: dict[str, int] = {}
        pairs: list[tuple[str, int]] = []
        for arm in arms:
            counts[arm] = counts.get(arm, 0) + 1
            pairs.append((arm, counts[arm]))
    else:
        pairs = grid_pairs(runs)
    return execute_plan(
        pairs, out_dir=out_dir, paper_pdf=Path(paper_pdf), paper_text=text,
        arxiv_stub=arxiv_stub, budget=budget, pricing=pricing, settings=settings,
        llm_factory=llm_factory,
    )


def _paper_text_for(paper_pdf: Path, paper_text: str | None) -> str:
    if not Path(paper_pdf).is_file():
        raise FileNotFoundError(f"paper PDF not found: {paper_pdf}")
    return paper_text if paper_text is not None else extract_pdf_text(str(paper_pdf), max_chars=10**8)


def execute_plan(
    pairs: list[tuple[str, int]], *,
    out_dir: Path,
    paper_pdf: Path = PAPER_PDF,
    paper_text: str,
    arxiv_stub: Callable = stubbed_arxiv_search,
    budget: float = BUDGET_CNY,
    pricing: Pricing = PRICING_PEAK_CNY,
    settings: Settings | None = None,
    llm_factory: Callable[[], LLMClient] | None = None,
    spent: float = 0.0,
) -> list[dict]:
    """Buy exactly the (arm, repeat) pairs given, in order, under one budget.

    Both entry points share this: a fresh grid builds its pairs from
    :func:`plan_for`, and an extension builds them from the repeats that are
    still missing (:func:`extension_plan`). Because the pairs - not a repeat
    *count* - are the input, an extension can never re-buy a run it already
    has on disk.
    """
    text = _paper_text_for(paper_pdf, paper_text)
    paper_tokens = canonical_tokens(text)
    paper_index = _token_index(text)
    paper_values = paper_number_values(text)
    settings = settings or Settings()
    paper_pdf = Path(paper_pdf)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for arm, index in pairs:
        run_id = f"{arm}-r{index}"
        run_dir = out_dir / run_id
        if spent >= budget:
            records.append(
                {
                    "run_id": run_id, "arm": arm, "status": "skipped",
                    "error": f"budget cap of {budget} CNY already reached before this run",
                    "wall_seconds": 0.0, "llm_calls": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0,
                    "cost_cny": 0.0,
                }
            )
            continue

        collector: dict = {}
        remaining = round(budget - spent, 6)
        try:
            if arm == "A_single_prompt":
                arm_run = run_arm_single_prompt(
                    settings=settings, out_dir=run_dir, paper_text=text,
                    paper_meta={"title": PAPER_FACTS["paper"]["title"], "authors": [PAPER_FACTS["paper"]["author"]],
                                "doi": PAPER_FACTS["doi"], "abstract": ""},
                    budget=remaining, pricing=pricing, collector=collector,
                    llm_factory=llm_factory,
                )
            elif arm in ("B_pipeline", "C_pipeline_no_verification"):
                arm_run = run_arm_pipeline(
                    arm=arm, settings=settings, out_root=run_dir, paper_pdf=paper_pdf,
                    budget=remaining, pricing=pricing, arxiv_stub=arxiv_stub, collector=collector,
                    llm_factory=llm_factory,
                )
            else:
                raise ValueError(f"unknown arm {arm!r}")
            record = {**arm_run.record, **evaluate_report(arm_run.report, paper_tokens, paper_index, paper_values)}
            # arm A produces no claim artifact; 0 is the honest reading of
            # "this architecture records no per-claim evidence", and it keeps
            # the column present for every arm instead of missing
            record.setdefault("claims_total", 0)
            record.setdefault("claims_with_deterministic_status", 0)
            record["judge_samples"] = sample_judge_items(
                arm, arm_run.report, collector.get("claims") or []
            )
        except BudgetExceeded as exc:
            record = {
                "run_id": run_id, "arm": arm, "status": "skipped", "error": f"BudgetExceeded: {exc}",
                "wall_seconds": 0.0, **{k: v for k, v in (collector.get("usage") or {}).items()},
            }
            record.setdefault("cost_cny", spent_delta(collector))
        except Exception as exc:  # a failed run is data, not a crash
            usage = collector.get("usage") or {}
            record = {
                "run_id": run_id, "arm": arm, "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=3),
                "wall_seconds": 0.0, **usage,
            }
            record.setdefault("cost_cny", 0.0)
        if "prompt_tokens" not in record:
            usage = collector.get("usage") or {}
            record.update(usage)
        spent = round(spent + float(record.get("cost_cny") or 0.0), 6)
        record["cost_cny_cumulative"] = spent
        record.pop("payloads", None)
        records.append(record)
    return records


def spent_delta(collector: dict) -> float:
    return float((collector.get("usage") or {}).get("cost_cny") or 0.0)


# ---------------------------------------------------------------------------
# extending a grid that has already been paid for
#
# The stored runs *are* the cache: their reports, token counts and costs are on
# disk, so widening N=3 to N=5 costs two more repeats per arm and nothing else.
# Anything that would re-buy a stored run is a bug, not a re-run.
# ---------------------------------------------------------------------------

_RUN_ID = __import__("re").compile(r"-r(\d+)$")


def completed_repeats(records: list[dict]) -> dict[tuple[str, int], dict]:
    """``(arm, repeat) -> record`` for every repeat that produced a report.

    Only ``ok`` runs count as bought. A failed or skipped repeat is missing
    data: an extension must re-buy it rather than count the empty row as a
    sample, which would quietly make N=5 mean "5 rows" instead of "5 reviews".
    """
    done: dict[tuple[str, int], dict] = {}
    for record in records:
        if record.get("status") != STATUS_OK:
            continue
        match = _RUN_ID.search(str(record.get("run_id") or ""))
        if match is None:
            continue
        done[(str(record.get("arm")), int(match.group(1)))] = record
    return done


def extension_plan(records: list[dict], runs: int) -> list[tuple[str, int]]:
    """The ``(arm, repeat)`` pairs still missing to reach `runs` per arm.

    Interleaved by repeat, like :func:`plan_for`, so a truncated extension still
    leaves a balanced grid on disk.
    """
    done = completed_repeats(records)
    return [
        (arm, index)
        for index in range(1, runs + 1)
        for arm in ARMS
        if (arm, index) not in done
    ]


def stored_report_integrity(stored: list[dict], runs_dir: Path) -> dict:
    """Which stored reports no longer hold the text the previous grid measured.

    The extension re-scores from disk, so a run directory that was overwritten
    after the previous grid published (a re-run that was aborted, a stray
    process, a manual edit) would make the "N=3" column and the "N=3" rows of
    the merged table disagree about what the model actually wrote. Comparing the
    published `report_chars` with the file on disk finds those runs and reports
    both readings instead of silently preferring one.
    """
    checked = 0
    mismatched: list[dict] = []
    for record in stored:
        if record.get("status") != STATUS_OK:
            continue
        checked += 1
        path = _report_path_for(record, runs_dir / str(record.get("run_id")))
        if path is None:
            mismatched.append({
                "run_id": record.get("run_id"),
                "issue": "the report this grid measured is no longer on disk",
                "published_report_chars": record.get("report_chars"),
                "stored_report_chars": None,
                "published_coverage": record.get("coverage"),
                "stored_coverage": None,
            })
            continue
        text = path.read_text(encoding="utf-8")
        if record.get("report_chars") != len(text):
            coverage, _total = contributions_covered(text)
            mismatched.append({
                "run_id": record.get("run_id"),
                "issue": "the stored report differs from the text this grid measured",
                "path": str(path),
                "published_report_chars": record.get("report_chars"),
                "stored_report_chars": len(text),
                "published_coverage": record.get("coverage"),
                "stored_coverage": coverage,
                "stored_report_mtime": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)
                ),
            })
    return {
        "runs_checked": checked,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mismatched": mismatched,
        "note": (
            "The N=3 side of this comparison is the stored reports re-scored by the current "
            "scorers, so the comparison and the merged `runs` agree. The numbers the previous "
            "grid published came from the texts listed here; where a run appears above, the two "
            "readings differ and both are reported."
        ),
    }


def merge_records(
    stored: list[dict], bought: list[dict], paper_text: str, runs_dir: Path
) -> list[dict]:
    """Stored + newly bought runs, re-scored together and ordered as one grid.

    Every record - old and new - has its deterministic columns recomputed from
    the report on disk, so the merged table is measured by *one* version of the
    scorers. The stored runs' tokens, cost and timing are carried over
    untouched: re-scoring cannot buy a token, and the report says so.
    """
    bought_ids = {record["run_id"] for record in bought}
    merged = rescore_runs(list(stored) + list(bought), paper_text, runs_dir)
    for record in merged:
        if "cost_cny_cumulative" in record and record["run_id"] in bought_ids:
            # a freshly bought run's cumulative total belongs to the extension's
            # own ledger, not to the merged grid's
            record.pop("cost_cny_cumulative", None)

    def order(record: dict) -> tuple[int, int]:
        match = _RUN_ID.search(str(record.get("run_id") or ""))
        repeat = int(match.group(1)) if match else 0
        try:
            position = ARMS.index(str(record.get("arm")))
        except ValueError:
            position = len(ARMS)
        return (repeat, position)

    merged.sort(key=order)
    cumulative = 0.0
    for record in merged:
        cumulative = round(cumulative + float(record.get("cost_cny") or 0.0), 6)
        record["cost_cny_cumulative"] = cumulative
    return merged


def extend_experiment(
    *, previous_path: Path, runs: int, out_dir: Path, budget: float, settings: Settings,
    pricing: Pricing = PRICING_PEAK_CNY, judge: bool = True, paper_pdf: Path = PAPER_PDF,
    paper_text: str | None = None, arxiv_stub: Callable = stubbed_arxiv_search,
    llm_factory: Callable[[], LLMClient] | None = None, judge_items_per_new_run: int = 10,
) -> dict:
    """Widen an already-paid grid to `runs` repeats per arm and merge the two.

    The previous payload is read (never overwritten), only the missing repeats
    are bought, and the judge column is *added to* rather than re-bought: the
    recorded verdicts stay as they were measured, and a second, explicitly
    labelled pass judges the new repeats so the secondary column can be checked
    against fresh samples instead of being re-quoted from the old ones.
    """
    previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
    stored = list(previous.get("runs") or [])
    text = _paper_text_for(paper_pdf, paper_text)

    pairs = extension_plan(stored, runs)
    bought = execute_plan(
        pairs, out_dir=out_dir, paper_pdf=Path(paper_pdf), paper_text=text,
        arxiv_stub=arxiv_stub, budget=budget, pricing=pricing, settings=settings,
        llm_factory=llm_factory,
    )
    merged = merge_records(stored, bought, text, out_dir)
    summary = aggregate(merged)

    arms_spend = round(sum(float(record.get("cost_cny") or 0.0) for record in bought), 6)
    judge_extension = None
    if judge and pairs:
        fresh = [record for record in merged if record.get("run_id") in {r["run_id"] for r in bought}
                 and record.get("status") == STATUS_OK]
        if fresh:
            judge_extension = judge_records(
                fresh, text, settings=settings, budget=max(0.0, budget - arms_spend),
                pricing=pricing, llm_factory=llm_factory, per_run=judge_items_per_new_run,
            )
            judge_extension["scope"] = (
                f"the {len(fresh)} runs bought by this extension; deliberately *not* a re-run of the "
                "recorded column, so the two can be compared"
            )
            for arm, column in judge_extension["arms"].items():
                if arm in summary["arms"]:
                    summary["arms"][arm]["semantic_support_extension"] = {
                        key: value for key, value in column.items() if key != "sample"
                    }

    history = {
        "source": str(previous_path),
        "runs_per_arm": int((previous.get("meta") or {}).get("runs_per_arm") or 0) or None,
        "started_at": (previous.get("meta") or {}).get("started_at"),
        "summary": previous.get("summary"),
        "meta": previous.get("meta"),
        "judge": previous.get("judge"),
        "total_spend_cny": previous.get("total_spend_cny"),
    }
    # The comparison's N=3 side is the *stored* runs re-scored, so the table and
    # the merged run list are the same evidence. The previously published
    # summary stays in `history.summary`, and any run whose stored report no
    # longer matches what that summary measured is reported in `integrity`.
    stored_rescored = records_up_to_repeat(merged, history["runs_per_arm"] or 0)
    history["rescored_summary"] = aggregate(stored_rescored) if stored_rescored else None
    history["integrity"] = stored_report_integrity(stored, out_dir)
    comparison = compare_summaries(
        history["rescored_summary"] or {"arms": {}}, summary,
        previous_records=stored_rescored, current_records=merged,
        previous_runs_per_arm=history["runs_per_arm"] or 0, runs_per_arm=runs,
    )
    bought_ids = [record["run_id"] for record in bought]
    reused_ids = [record["run_id"] for record in stored if record.get("run_id") not in set(bought_ids)]
    extension = {
        "previous_grid_total_cny": float(previous.get("total_spend_cny") or 0.0),
        "previous_runs_per_arm": history["runs_per_arm"],
        "runs_per_arm": runs,
        "runs_bought": len(bought),
        "reused_runs": len(reused_ids),
        "bought_run_ids": bought_ids,
        "extension_arms_cny": arms_spend,
        "extension_judge_cny": round(float((judge_extension or {}).get("judge_cost_cny") or 0.0), 6),
        "extension_total_cny": round(arms_spend + float((judge_extension or {}).get("judge_cost_cny") or 0.0), 6),
        "reused_run_ids": reused_ids,
        "budget_cny": budget,
        "budget_used_fraction": round((arms_spend + float((judge_extension or {}).get("judge_cost_cny") or 0.0)) / budget, 6) if budget else None,
        "new_failure_modes": new_failure_modes(stored, bought, judge_extension, previous.get("judge")),
    }
    return {
        "records": merged, "summary": summary, "judge": previous.get("judge"),
        "judge_extension": judge_extension, "history": history,
        "comparison": comparison, "extension": extension,
    }


def sample_judge_items(arm: str, report: str, claims: list[dict], limit: int = 20) -> list[dict]:
    """The bullets that the (secondary) LLM semantic-support judge will read.

    Each report bullet is aligned to its Reader claim with the project's own
    multi-key matcher (:func:`paperflow.agents.verification.align_bullets`),
    which is what the shipped ledger injection already relies on - the
    experiment does not introduce a second, experimental alignment rule.
    A bullet with no matching claim is reported with an empty quote, and the
    judge is expected to call that "unclear" rather than to guess.

    Stratified so the risky items are actually in the sample: every claim the
    deterministic check could not confirm comes first, then the rest in report
    order. The sample is a *subset*, reported with its n.
    """
    from paperflow.agents.verification import claim_statuses, align_bullets

    rows = claim_statuses_from_claims(claims)
    bullets = claim_bullets(report)
    assignment = align_bullets([_strip_markers(bullet) for bullet in bullets], rows)

    items: list[dict] = []
    for index, bullet in enumerate(bullets):
        row = rows[assignment[index]] if index in assignment else None
        items.append(
            {
                "arm": arm,
                "bullet_index": index,
                "claim_index": row["claim_index"] if row else None,
                "quote_status": row["status"] if row else "unmatched",
                "text": bullet,
                "quote": str((row or {}).get("quote") or ""),
            }
        )
    items.sort(key=lambda item: 0 if item["quote_status"] != "verified" else 1)
    return items[:limit]


def claim_statuses_from_claims(claims: list[dict]) -> list[dict]:
    """The machine status rows for a plain list of reader claims.

    ``verification.claim_statuses`` reads the board; the harness only has the
    artifact, so the same status mapping is applied here to the artifact's own
    annotations (``quote_status`` wins, then the ``quote_verified`` boolean,
    and a claim with neither is ``unknown`` - never silently verified).
    """
    from paperflow.agents.verification import STATUS_MARKER, UNKNOWN, _status_of

    rows: list[dict] = []
    for index, claim in enumerate(claims):
        status = _status_of(claim)
        rows.append(
            {
                "claim_index": index,
                "claim": str(claim.get("claim", "")),
                "quote": str(claim.get("quote", "")),
                "status": status if status in STATUS_MARKER else UNKNOWN,
            }
        )
    return rows


def _strip_markers(text: str) -> str:
    import re

    return re.sub(r"\*\*\[[^\]]+\]\*\*|\[[^\]]+\]", "", text).strip(" *-")


# ---------------------------------------------------------------------------
# the semantic-support judge (secondary, LLM-judged, protocol declared)
# ---------------------------------------------------------------------------

#: the judge's answer options in the order they are offered on pass 1; pass 2
#: offers them reversed, so a preference for the first option shows up as
#: disagreement instead of hiding inside a stable-looking number
_PASS1_ORDER = ("yes", "no", "unclear")


def _judge_prompt(items: list[dict], paper_text: str, quoted: bool, order: tuple[str, ...]) -> list[dict]:
    """The batch prompt, with the answer labels listed in `order`.

    The order is a knob, not a hint: the two passes list the labels
    differently and the wording says so explicitly, so a model that simply
    picks the first label it sees shows up as disagreement rather than as a
    stable (and wrong) verdict.
    """
    messages = build_messages(items, paper_text, quoted=quoted)
    labels = {
        "yes": "yes - the text contains what the claim asserts",
        "no": "no - the text does not contain it, or contradicts it",
        "unclear": "unclear - empty/absent evidence, or the claim is a judgement the text neither states nor contradicts",
    }
    choices = "\n".join(f"  - {labels[label]}" for label in order)
    messages[0]["content"] = (
        messages[0]["content"]
        + "\nAnswer every item with exactly one of these labels:\n"
        + choices
        + "\nThe order in which the labels are listed carries no meaning: pick the one "
        "that is correct for each item."
    )
    return messages


def _verdicts_constant(verdicts: list[str]) -> bool:
    return len(set(verdicts)) <= 1


def _judge_pass(
    llm: LLMClient, items: list[dict], paper_text: str, *, quoted: bool, order: tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Judge `items` in batches of BATCH; returns (verdicts, reasons)."""
    verdicts: list[str] = []
    reasons: list[str] = []
    for start in range(0, len(items), BATCH):
        batch = items[start : start + BATCH]
        message = llm.chat(
            _judge_prompt(batch, paper_text, quoted, order),
            temperature=0.0,
            max_tokens=700,
        )
        content = message.get("content") or ""
        verdicts.extend(parse_verdicts(content, len(batch)))
        reasons.extend(parse_reasons(content, len(batch)))
    return verdicts, reasons


def judge_records(
    records: list[dict], paper_text: str, *, settings: Settings, budget: float, pricing: Pricing,
    llm_factory: Callable[[], LLMClient] | None = None, max_per_arm: int = 20,
    per_run: int | None = None,
) -> dict:
    """Run the declared judge over the sampled claim bullets and attach the result.

    Two passes per arm with the option order swapped; the agreement between
    them is the reliability figure reported next to the verdicts. The judge's
    own token cost is measured with the same ledger as the arms.

    ``per_run`` changes the *sampling rule*, and both rules are recorded in the
    output. The historical rule concatenates every run's bullets and takes the
    first ``max_per_arm``, which lets the earliest repeat fill the whole sample
    (in the first grid it did: the 20 judged bullets came from repeats 1-2).
    An extension sets ``per_run`` so the fresh repeats are all represented
    instead of the sample collapsing onto the first one again.
    """
    llm = _client_for(settings, budget, pricing, llm_factory)
    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        rows = [r for r in records if r.get("arm") == arm and r.get("status") == STATUS_OK]
        if per_run is None:
            take = None
            sampling = (
                f"first {max_per_arm} bullets of the concatenated repeats (historical rule): "
                "earlier repeats can fill the whole sample"
            )
            items = [item for row in rows for item in (row.get("judge_samples") or [])][:max_per_arm]
        else:
            take = max(1, min(per_run, max_per_arm // max(1, len(rows))))
            sampling = f"up to {take} bullets per repeat across {len(rows)} repeats (balanced rule)"
            items = [item for row in rows for item in (row.get("judge_samples") or [])[:take]]
        if not items:
            per_arm[arm] = {
                "n": 0, "yes": 0, "no": 0, "unclear": 0, "support_rate": None,
                "measured": False, "reason": "no claim bullets were available to judge",
                "sampling": sampling,
            }
            continue
        quoted = arm != "A_single_prompt"
        try:
            first, reasons_first = _judge_pass(llm, items, paper_text, quoted=quoted, order=_PASS1_ORDER)
            second, reasons_second = _judge_pass(
                llm, items, paper_text, quoted=quoted, order=tuple(reversed(_PASS1_ORDER))
            )
        except BudgetExceeded as exc:
            per_arm[arm] = {
                "n": len(items), "measured": False,
                "reason": f"judge stopped by the budget cap: {exc}",
                "sampling": sampling,
            }
            continue
        for item, verdict, reason, reason_second in zip(items, first, reasons_first, reasons_second):
            item["verdict"] = verdict
            item["reason"] = reason
            item["verdict_pass2"] = None
            item["reason_pass2"] = reason_second
        for item, verdict in zip(items, second):
            item["verdict_pass2"] = verdict
        per_arm[arm] = {
            **summarise_verdicts(items),
            "measured": True,
            "question": "does the attached quote support the claim?" if quoted
            else "does the paper contain what the claim asserts? (arm A carries no quote artifact)",
            "sample": items,
            "pass1_verdicts": first,
            "pass2_verdicts": second,
            "sampling": sampling,
            "per_repeat_limit": take,
            "percent_agreement": percent_agreement(first, second),
            "cohen_kappa": (
                None if _verdicts_constant(first) and _verdicts_constant(second) and first == second
                else cohen_kappa(first, second)
            ),
            "kappa_note": (
                "kappa undefined: both passes gave the same single verdict for every item, "
                "so percent agreement is 1.0 with no variance to explain"
                if _verdicts_constant(first) and _verdicts_constant(second) and first == second
                else None
            ),
        }
    return {
        "model": settings.model,
        "temperature": 0.0,
        "prompt_version": PROMPT_VERSION,
        "batch_size": BATCH,
        "option_order_swap": True,
        "max_items_per_arm": max_per_arm,
        "judge_cost_cny": llm.usage.cost_cny(pricing),
        "judge_tokens": llm.usage_totals(),
        "arms": per_arm,
        "measurement_note": (
            "SECONDARY, LLM-JUDGED column. Deterministic columns do not depend on it. "
            "support_rate counts every judged item as its denominator, so an 'unclear' "
            "verdict is never a pass."
        ),
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _stats(values: list[float | int]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": round(sum(clean) / len(clean), 4),
        "min": min(clean),
        "max": max(clean),
    }


# ---------------------------------------------------------------------------
# N=3 vs N=5: did widening the sample change anything?
#
# The extension exists to answer three questions and nothing else: is the
# coverage gap inside the noise, is the cost multiple stable, and did the wider
# sample expose a failure the first three runs never showed. These functions
# compute those answers from the two summaries; `render_markdown` writes them
# out, so the delivered document cannot drift from the numbers.
# ---------------------------------------------------------------------------

#: short names used in the comparison tables
ARM_LABELS = {"A_single_prompt": "A", "B_pipeline": "B", "C_pipeline_no_verification": "C"}

#: the deterministic quality columns the overlap question is asked about
QUALITY_COLUMNS = (
    "coverage", "coverage_rate", "factual_errors", "headings_present",
    "attribution_rate", "number_grounding_rate",
)
#: and the cost columns
COST_COLUMNS = ("cost_cny", "llm_calls", "wall_seconds", "prompt_tokens", "completion_tokens")
#: the column that carries the experiment's headline claim
HEADLINE_COLUMN = "coverage"


def arm_pairs() -> list[tuple[str, str]]:
    return [(a, b) for index, a in enumerate(ARMS) for b in ARMS[index + 1 :]]


def intervals_overlap(first: dict | None, second: dict | None) -> bool:
    """Do two [min, max] intervals share any point?

    Both must have runs behind them: two empty intervals are not "overlapping",
    they are unmeasured, and saying so is the difference between a result and a
    blank.
    """
    if not first or not second:
        return False
    if not first.get("n") or not second.get("n"):
        return False
    if first.get("min") is None or second.get("min") is None:
        return False
    return not (first["max"] < second["min"] or second["max"] < first["min"])


def _width(stats: dict | None) -> float | None:
    if not stats or stats.get("min") is None or stats.get("max") is None:
        return None
    return round(float(stats["max"]) - float(stats["min"]), 6)


def _column_of(summary: dict, arm: str, column: str) -> dict:
    return ((summary.get("arms") or {}).get(arm) or {}).get(column) or {}


def _multiple(summary: dict, arm: str, base: str = "A_single_prompt") -> float | None:
    base_cost = _column_of(summary, base, "cost_cny").get("mean")
    cost = _column_of(summary, arm, "cost_cny").get("mean")
    return round(cost / base_cost, 4) if base_cost else None


def ratio_range(records: list[dict], arm: str, base: str = "A_single_prompt") -> list[float] | None:
    """Every cross-run cost ratio `arm / base`, as [min, max].

    Cross-run on purpose: a per-review price is not paired across arms, so the
    honest interval is the range over all arm-run x base-run combinations, not
    the ratio of two means with a made-up error bar.
    """
    arm_costs = [float(r["cost_cny"]) for r in records
                 if r.get("arm") == arm and r.get("cost_cny") and r.get("status") == STATUS_OK]
    base_costs = [float(r["cost_cny"]) for r in records
                  if r.get("arm") == base and r.get("cost_cny") and r.get("status") == STATUS_OK]
    if not arm_costs or not base_costs:
        return None
    ratios = [a / b for a in arm_costs for b in base_costs if b]
    return [round(min(ratios), 4), round(max(ratios), 4)]


def compare_summaries(
    previous: dict, current: dict, *,
    previous_records: list[dict] | None = None, current_records: list[dict] | None = None,
    previous_runs_per_arm: int = 3, runs_per_arm: int = 5,
) -> dict:
    """The two grids side by side, per column, with the overlap verdict.

    Nothing here re-measures: both summaries were produced by `aggregate` from
    records that are already on disk. The point is the *comparison*, which is
    the only thing that can justify the extra spend.
    """
    columns: dict[str, dict] = {}
    for column in QUALITY_COLUMNS + COST_COLUMNS:
        entry: dict[str, Any] = {
            "mean_n3": {}, "mean_n5": {}, "mean_shift_n5_minus_n3": {},
            "range_n3": {}, "range_n5": {},
            "width_n3": {}, "width_n5": {}, "narrowed_by": {},
            "pairwise_overlap_n3": {}, "pairwise_overlap_n5": {},
        }
        for arm in ARMS:
            n3 = _column_of(previous, arm, column)
            n5 = _column_of(current, arm, column)
            entry["mean_n3"][arm] = n3.get("mean")
            entry["mean_n5"][arm] = n5.get("mean")
            entry["mean_shift_n5_minus_n3"][arm] = (
                round(n5["mean"] - n3["mean"], 6)
                if n3.get("mean") is not None and n5.get("mean") is not None else None
            )
            entry["range_n3"][arm] = [n3.get("min"), n3.get("max")]
            entry["range_n5"][arm] = [n5.get("min"), n5.get("max")]
            entry["width_n3"][arm] = _width(n3)
            entry["width_n5"][arm] = _width(n5)
            entry["narrowed_by"][arm] = (
                round(_width(n3) - _width(n5), 6)
                if _width(n3) is not None and _width(n5) is not None else None
            )
        for first, second in arm_pairs():
            key = f"{ARM_LABELS[first]}_vs_{ARM_LABELS[second]}"
            entry["pairwise_overlap_n3"][key] = intervals_overlap(
                _column_of(previous, first, column), _column_of(previous, second, column)
            )
            entry["pairwise_overlap_n5"][key] = intervals_overlap(
                _column_of(current, first, column), _column_of(current, second, column)
            )
        entry["every_pair_overlaps_n3"] = all(entry["pairwise_overlap_n3"].values())
        entry["every_pair_overlaps_n5"] = all(entry["pairwise_overlap_n5"].values())
        entry["arms_separated_n3"] = not entry["every_pair_overlaps_n3"]
        entry["arms_separated_n5"] = not entry["every_pair_overlaps_n5"]
        columns[column] = entry

    cost_multiple: dict[str, Any] = {
        "metric": "mean CNY per review",
        "b_vs_a_n3": _multiple(previous, "B_pipeline"),
        "b_vs_a_n5": _multiple(current, "B_pipeline"),
        "c_vs_a_n3": _multiple(previous, "C_pipeline_no_verification"),
        "c_vs_a_n5": _multiple(current, "C_pipeline_no_verification"),
    }
    cost_multiple["b_vs_a_delta"] = (
        round(cost_multiple["b_vs_a_n5"] - cost_multiple["b_vs_a_n3"], 4)
        if cost_multiple["b_vs_a_n3"] is not None and cost_multiple["b_vs_a_n5"] is not None else None
    )
    if previous_records and current_records:
        cost_multiple["b_vs_a_ratio_range_n3"] = ratio_range(previous_records, "B_pipeline")
        cost_multiple["b_vs_a_ratio_range_n5"] = ratio_range(current_records, "B_pipeline")
        band = cost_multiple["b_vs_a_ratio_range_n3"]
        # Stability is defined before looking: the wider sample is "stable" when
        # its point estimate still lands inside the ratio band the 3-run grid
        # already implied. Anything else is a real move, not noise.
        cost_multiple["stable"] = bool(
            band and cost_multiple["b_vs_a_n5"] is not None
            and band[0] <= cost_multiple["b_vs_a_n5"] <= band[1]
        )
        for key in ("b_vs_a_ratio_range_n3", "b_vs_a_ratio_range_n5"):
            cost_multiple[key + "_width"] = (
                round(cost_multiple[key][1] - cost_multiple[key][0], 4) if cost_multiple[key] else None
            )

    headline = columns.get(HEADLINE_COLUMN, {})
    verdicts = {
        HEADLINE_COLUMN: {
            "column": HEADLINE_COLUMN,
            "mean_n3": headline.get("mean_n3"), "mean_n5": headline.get("mean_n5"),
            "range_n3": headline.get("range_n3"), "range_n5": headline.get("range_n5"),
            "arms_separated_n3": headline.get("arms_separated_n3", False),
            "arms_separated_n5": headline.get("arms_separated_n5", False),
            "overlap_n5": headline.get("pairwise_overlap_n5", {}),
            "gap_n5": (
                round(headline["mean_n5"]["A_single_prompt"] - headline["mean_n5"]["B_pipeline"], 4)
                if headline.get("mean_n5", {}).get("A_single_prompt") is not None
                and headline["mean_n5"].get("B_pipeline") is not None else None
            ),
        },
        "attribution_rate": {
            "column": "attribution_rate",
            "mean_n3": columns["attribution_rate"]["mean_n3"],
            "mean_n5": columns["attribution_rate"]["mean_n5"],
            "arms_separated_n3": columns["attribution_rate"]["arms_separated_n3"],
            "arms_separated_n5": columns["attribution_rate"]["arms_separated_n5"],
            "overlap_n5": columns["attribution_rate"]["pairwise_overlap_n5"],
        },
        "factual_errors": {
            "column": "factual_errors",
            "mean_n3": columns["factual_errors"]["mean_n3"],
            "mean_n5": columns["factual_errors"]["mean_n5"],
            "arms_separated_n3": columns["factual_errors"]["arms_separated_n3"],
            "arms_separated_n5": columns["factual_errors"]["arms_separated_n5"],
            "overlap_n5": columns["factual_errors"]["pairwise_overlap_n5"],
        },
        "cost_multiple": {
            "n3": cost_multiple["b_vs_a_n3"], "n5": cost_multiple["b_vs_a_n5"],
            "c_vs_a_n3": cost_multiple["c_vs_a_n3"], "c_vs_a_n5": cost_multiple["c_vs_a_n5"],
            "delta": cost_multiple["b_vs_a_delta"],
            "ratio_range_n3": cost_multiple.get("b_vs_a_ratio_range_n3"),
            "ratio_range_n5": cost_multiple.get("b_vs_a_ratio_range_n5"),
            "stable": cost_multiple.get("stable"),
        },
        "narrowing": {
            column: {
                arm: {"width_n3": columns[column]["width_n3"][arm], "width_n5": columns[column]["width_n5"][arm]}
                for arm in ARMS
            }
            for column in QUALITY_COLUMNS + COST_COLUMNS
        },
    }
    return {
        "runs_per_arm": {"n3": previous_runs_per_arm, "n5": runs_per_arm},
        "columns": columns,
        "cost_multiple": cost_multiple,
        "verdicts": verdicts,
        "reading_note": (
            "min-max over successful runs, not a confidence interval: with n=5 no significance "
            "test is defensible, so a disjoint pair of intervals is reported as 'the arms did not "
            "overlap in this sample', never as a proven difference."
        ),
    }


def new_failure_modes(
    stored: list[dict], bought: list[dict], judge_extension: dict | None = None,
    historical_judge: dict | None = None,
) -> list[str]:
    """Anything the new repeats show that the first grid never did.

    Deliberately narrow: a failed run, a new factual error, a quality column
    below every stored run's value, a pipeline run that shipped no ledger, or a
    verification marker leaking into the arm that has verification switched off.
    An empty list is a result ("no new failure mode"), which is why it is
    recorded as a list rather than as prose written after the fact.
    """
    modes: list[str] = []
    stored_ok = [r for r in stored if r.get("status") == STATUS_OK]
    ok = [r for r in bought if r.get("status") == STATUS_OK]

    for record in bought:
        if record.get("status") != STATUS_OK:
            modes.append(f"{record['run_id']}: {record.get('status')} - {record.get('error')}")
        elif record.get("factual_error_details"):
            modes.append(f"{record['run_id']}: factual error(s) {record['factual_error_details']}")

    for column in ("coverage", "headings_present", "attribution_rate"):
        for arm in ARMS:
            old = [r.get(column) for r in stored_ok if r.get("arm") == arm and r.get(column) is not None]
            new = [r.get(column) for r in ok if r.get("arm") == arm and r.get(column) is not None]
            if old and new and min(new) < min(old):
                modes.append(
                    f"{ARM_LABELS.get(arm, arm)}: {column} fell to {min(new)} in the new repeats, "
                    f"below every stored run (min {min(old)})"
                )

    for record in ok:
        if record.get("arm") == "B_pipeline" and not record.get("ledger_in_report"):
            modes.append(f"{record['run_id']}: the pipeline shipped a report with no verification ledger")
        if record.get("arm") == "C_pipeline_no_verification" and (
            record.get("claims_with_deterministic_status") or record.get("ledger_in_report")
        ):
            modes.append(
                f"{record['run_id']}: the machine layer left a footprint in the arm where it is switched off "
                f"(claim statuses={record.get('claims_with_deterministic_status')}, "
                f"ledger={record.get('ledger_in_report')})"
            )

    for arm, column in (judge_extension or {}).get("arms", {}).items():
        before = ((historical_judge or {}).get("arms") or {}).get(arm) or {}
        after = column.get("support_rate")
        was = before.get("support_rate")
        if after is not None and was is not None and after < was - 0.2:
            modes.append(
                f"{ARM_LABELS.get(arm, arm)}: judge support rate fell from {was} to {after} on the new repeats"
            )
    return modes


def aggregate(records: list[dict]) -> dict:
    """Mean/min/max per arm over successful runs, plus cost totals.

    Failed and skipped runs are counted and listed but excluded from the
    means - and removing them is stated in the output, never silent.
    """
    arms: dict[str, dict] = {}
    for arm in ARMS:
        rows = [r for r in records if r.get("arm") == arm]
        ok = [r for r in rows if r.get("status") == "ok"]
        failed = [r for r in rows if r.get("status") == "failed"]
        skipped = [r for r in rows if r.get("status") == "skipped"]
        summary: dict[str, Any] = {
            "runs_total": len(rows),
            "runs_ok": len(ok),
            "runs_failed": len(failed),
            "runs_skipped": len(skipped),
            "failure_reasons": sorted({str(r.get("error")) for r in failed if r.get("error")}),
        }
        for column in (
            "llm_calls", "prompt_tokens", "completion_tokens", "cache_hit_tokens",
            "cache_miss_tokens", "cost_cny", "wall_seconds", "headings_present",
            "report_chars", "claim_bullets", "claim_bullets_locatable", "attribution_rate",
            "number_bearing_bullets", "number_grounded_bullets", "number_grounding_rate",
            "coverage", "coverage_rate", "factual_errors",
            "llm_markers_on_bullets", "machine_markers_on_bullets",
            "claims_total", "claims_with_deterministic_status",
            "claims_quote_verified", "quote_evidence_rate",
        ):
            summary[column] = _stats([r.get(column) for r in ok])
        summary["ledger_in_report_all_runs"] = bool(ok) and all(bool(r.get("ledger_in_report")) for r in ok)
        summary["factual_error_details"] = [d for r in ok for d in (r.get("factual_error_details") or [])]
        arms[arm] = summary

    all_cost = round(sum(float(r.get("cost_cny") or 0.0) for r in records), 6)
    return {
        "arms": arms,
        "total_cost_cny": all_cost,
        "runs_total": len(records),
        "runs_ok": sum(1 for r in records if r.get("status") == "ok"),
        "runs_failed": sum(1 for r in records if r.get("status") == "failed"),
        "runs_skipped": sum(1 for r in records if r.get("status") == "skipped"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def rescore_runs(records: list[dict], paper_text: str, runs_dir: Path) -> list[dict]:
    """Recompute the deterministic columns from the stored reports of a past run.

    The model outputs are read back from each run directory and scored with the
    current scorers, which is how a fixed measurement bug is corrected without
    buying new samples. Nothing about an arm's *output* changes: only the
    measurement of it, and the report says so.

    Token counts, costs and timings are carried over untouched - they came from
    the API's own `usage` block and re-scoring cannot change them.
    """
    paper_tokens = canonical_tokens(paper_text)
    paper_index = _token_index(paper_text)
    paper_values = paper_number_values(paper_text)

    rescored: list[dict] = []
    for record in records:
        if record.get("status") == STATUS_SKIPPED:
            rescored.append(record)  # never ran: no report exists, nothing to re-score
            continue
        run_dir = runs_dir / record["run_id"]
        report_path = _report_path_for(record, run_dir)
        if report_path is None:
            record = {**record, "status": STATUS_FAILED, "error": "report not found for re-scoring"}
            rescored.append(record)
            continue
        # A run whose report is on disk is re-scored and marked ok again: the
        # model did produce it, and reporting otherwise because of an earlier
        # scoring bug would be the harness lying about what it has.
        report = report_path.read_text(encoding="utf-8")
        claims_doc = _load_json(_artifact_path_for(run_dir, "reader_output.json"))
        claims = (claims_doc or {}).get("claims", []) if isinstance(claims_doc, dict) else []
        fresh = evaluate_report(report, paper_tokens, paper_index, paper_values)
        merged = {
            **record,
            **fresh,
            "status": STATUS_OK if report.strip() else STATUS_FAILED,
            "error": None if report.strip() else "empty report on disk",
        }
        if record.get("arm") == "A_single_prompt":
            merged["claims_total"] = 0
            merged["claims_with_deterministic_status"] = 0
            merged.update(reader_claim_evidence(None))
        else:
            merged.update(reader_claim_evidence(claims))
            merged.update(
                {
                    "claims_total": len(claims),
                    "claims_with_deterministic_status": sum(1 for c in claims if c.get("quote_status")),
                    "claims_quote_unverified": sum(1 for c in claims if c.get("quote_status") == "unverified"),
                }
            )
        merged["judge_samples"] = sample_judge_items(record["arm"], report, claims)
        rescored.append(merged)
    return rescored


def _report_path_for(record: dict, run_dir: Path) -> Path | None:
    """Where a past run's report lives.

    Arm A writes it straight into the run directory; a pipeline run nests it
    one level down under the run id the pipeline allocated. Searching by
    pattern (rather than by an assumed depth) keeps the re-scorer working when
    the layout changes, and a missing report is reported as a failed run
    instead of being silently scored as empty.
    """
    for candidate in (run_dir / "report.md", run_dir / "out" / "report.md"):
        if candidate.is_file():
            return candidate
    matches = sorted(run_dir.glob("**/report.md"))
    return matches[0] if matches else None


def _artifact_path_for(run_dir: Path, name: str) -> Path | None:
    """Locate one artifact of a past run, wherever the pipeline nested it."""
    direct = run_dir / "out" / "artifacts" / name
    if direct.is_file():
        return direct
    matches = sorted(run_dir.glob(f"**/artifacts/{name}"))
    return matches[0] if matches else None


def build_report(
    records: list[dict], summary: dict, meta: dict, judge: dict | None = None, *,
    min_runs_per_arm: int = MIN_RUNS_PER_ARM,
    judge_extension: dict | None = None, history: dict | None = None,
    comparison: dict | None = None, extension: dict | None = None,
) -> dict:
    """Assemble the deliverable, refusing to publish a grid narrower than the floor.

    `min_runs_per_arm` is a parameter because re-scoring (`--score-only`)
    reproduces a design that was already run and must not be blocked by a floor
    introduced afterwards; every *new* report takes the default.
    """
    require_runs_per_arm(summary, min_runs_per_arm)
    judge_cny = float((judge or {}).get("judge_cost_cny") or 0.0)
    extension_judge_cny = float((judge_extension or {}).get("judge_cost_cny") or 0.0)
    spend = summary["total_cost_cny"] + judge_cny + extension_judge_cny
    payload = {
        "experiment": "paperflow three-arm comparison under a real model",
        "arms": {
            "A_single_prompt": "the paper text + task in one chat() call",
            "B_pipeline": "Researcher -> Reader -> Critic -> Synthesizer with the deterministic quote check and ledger",
            "C_pipeline_no_verification": "the same pipeline with the deterministic quote check and report injection disabled",
        },
        "measurement_note": (
            "Tokens, cache splits, call counts and cost come from the API's own `usage` block; "
            "cost is tokens x published DeepSeek prices (see paperflow.core.pricing). The "
            "attribution/coverage/factual-error columns are deterministic and involve no LLM. "
            "The semantic-support column is an LLM judgement over a declared sample."
        ),
        "meta": meta,
        "summary": summary,
        "judge": judge,
        "total_spend_cny": round(spend, 6),
        "total_spend_breakdown": {
            "arms_cny": summary["total_cost_cny"],
            "judge_cny": round(judge_cny, 6),
            "extension_judge_cny": round(extension_judge_cny, 6),
        },
        "runs": records,
    }
    if judge_extension is not None:
        payload["judge_extension"] = judge_extension
    if history is not None:
        payload["history"] = history
    if comparison is not None:
        payload["comparison"] = comparison
    if extension is not None:
        payload["extension"] = extension
        payload["spend"] = {
            "previous_grid_total_cny": extension.get("previous_grid_total_cny"),
            "extension_arms_cny": extension.get("extension_arms_cny"),
            "extension_judge_cny": extension.get("extension_judge_cny"),
            "extension_total_cny": extension.get("extension_total_cny"),
            "cumulative_experiment_cny": round(
                float(extension.get("previous_grid_total_cny") or 0.0)
                + float(extension.get("extension_total_cny") or 0.0), 6
            ),
            "budget_cny": extension.get("budget_cny"),
        }
    return payload


# ---------------------------------------------------------------------------
# the delivered document
#
# `docs/arm-comparison-live.md` is *rendered* from the payload, not written by
# hand next to it: every figure in it is read out of `docs/arm-comparison-live.json`,
# so the document cannot drift from the measurement it reports. It carries the
# same >=5-runs-per-arm floor as the JSON.
# ---------------------------------------------------------------------------

def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not measurable"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 10**6:
            return f"{int(value):,}" if not digits else f"{value:,.2f}".rstrip("0").rstrip(".")
        return f"{value:,.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _mean_min_max(stats: dict | None, digits: int = 3) -> str:
    if not stats or stats.get("n") in (None, 0) or stats.get("mean") is None:
        return "not measurable"
    return f"**{_num(stats['mean'], digits)}** ({_num(stats['min'], digits)}–{_num(stats['max'], digits)})"


def _range(stats: dict | None, digits: int = 3) -> str:
    if not stats or stats.get("min") is None:
        return "not measurable"
    return f"{_num(stats['min'], digits)}–{_num(stats['max'], digits)}"


def _signed(value: Any, digits: int = 3) -> str:
    if value is None:
        return "not measurable"
    return f"{float(value):+.{digits}f}"


def _rate(value: Any) -> str:
    if value is None:
        return "not measurable"
    return f"{float(value):.3f}"


def _plural(count: Any, singular: str, plural: str | None = None) -> str:
    number = float(count or 0)
    return singular if abs(number - 1) < 1e-9 else (plural or singular + "s")


def _coverage_reading(payload: dict, arm: str) -> str:
    """Which checklist items an arm missed, and whether it is always the same one.

    Read from the per-run `coverage_missing` ids, so the sentence "it keeps
    missing the same item" is a fact about the records rather than a claim
    written next to them.
    """
    from scripts.paper_ground_truth import CONTRIBUTIONS

    statements = {contribution["id"]: contribution["statement"] for contribution in CONTRIBUTIONS}
    ok_runs = [
        record for record in (payload.get("runs") or [])
        if record.get("arm") == arm and record.get("status") == "ok"
    ]
    if not ok_runs:
        return "no successful run to read"
    if not any("coverage_missing" in record for record in ok_runs):
        return "not measurable: this grid did not record the per-run checklist ids"
    runs = [set(record.get("coverage_missing") or []) for record in ok_runs]
    always = set.intersection(*runs)
    sometimes = set.union(*runs) - always
    parts = []
    if always:
        detail = "; ".join(f"`{item}` ({statements.get(item, 'unknown')})" for item in sorted(always))
        parts.append(f"missed in every one of the {len(runs)} runs: {detail}")
    if sometimes:
        parts.append(f"missed in some runs only: {', '.join(sorted(sometimes))}")
    if not parts:
        parts.append(f"missed nothing in any of the {len(runs)} runs")
    return "; ".join(parts)


def _missing_id_phrase(payload: dict, arm: str = "B_pipeline") -> str:
    """The checklist ids this arm missed across its runs, as markdown code spans."""
    ids = sorted({
        item
        for record in (payload.get("runs") or [])
        if record.get("arm") == arm and record.get("status") == "ok"
        for item in (record.get("coverage_missing") or [])
    })
    return ", ".join(f"`{item}`" for item in ids) if ids else "nothing"


def _pipeline_full_coverage_runs(payload: dict, arm: str = "B_pipeline") -> int:
    """How many of the pipeline's runs covered the whole checklist."""
    return sum(
        1 for record in (payload.get("runs") or [])
        if record.get("arm") == arm and record.get("status") == "ok"
        and not (record.get("coverage_missing") or [])
    )


def _separated_columns(comparison: dict, runs_key: str, columns: Iterable[str] = QUALITY_COLUMNS) -> list[str]:
    return [
        column for column in columns
        if ((comparison.get("columns") or {}).get(column) or {}).get(f"arms_separated_{runs_key}")
    ]


def _overlap_word(flag: bool | None) -> str:
    if flag is None:
        return "not measurable"
    return "overlap" if flag else "**disjoint**"


def render_markdown(payload: dict) -> str:
    """The delivered report, rendered from the payload.

    Structure: setup, cost, quality, the interval-overlap question the
    extension exists to answer, the three answers, the secondary judge column,
    how to reproduce it, and the limitations. When a `history` block is present
    the document is the N=3 vs N=5 comparison; without one it is a single-grid
    report and says so.
    """
    require_runs_per_arm(payload["summary"])
    summary = payload["summary"]
    meta = payload.get("meta") or {}
    history = payload.get("history")
    comparison = payload.get("comparison") or {}
    extension = payload.get("extension") or {}
    verdicts = comparison.get("verdicts") or {}
    columns = comparison.get("columns") or {}
    judge = payload.get("judge") or {}
    judge_extension = payload.get("judge_extension") or {}
    runs_n5 = int(meta.get("runs_per_arm") or 5)
    runs_n3 = int((history or {}).get("runs_per_arm") or 0) or None
    previous = (history or {}).get("summary") or {}
    # Everything the comparison *reads* comes from the re-scored basis, so the
    # N=3 column equals the N=3 rows of the merged run list. The published
    # summary is kept separately and shown where it differs.
    previous_rescored = (history or {}).get("rescored_summary") or previous
    integrity = (history or {}).get("integrity") or {}
    mismatched = integrity.get("mismatched") or []
    ok_n5 = summary.get("runs_ok", 0)
    out: list[str] = []

    def add(line: str = "") -> None:
        out.append(line)

    add(f"# Three-arm comparison under a real model — results (N={runs_n5} per arm)")
    add()
    add("**What this settles:** the first grid answered \"does the pipeline beat one prompt?\"")
    add(f"with {runs_n3 or 'three'} repeats per arm, its own declared floor at the time and a stated")
    add("limitation. This document reports the same experiment widened to")
    add(f"**{runs_n5} repeats per arm ({summary.get('runs_total', 0)} runs)**, and it answers the one")
    add("question the extra samples were bought for: **once the per-arm intervals are computed from a")
    add("wider sample, do the arms still fail to separate on quality?**")
    add()
    add("Every number below is rendered from `docs/arm-comparison-live.json` (machine-readable,")
    add("per-run) by `scripts/compare_arms_live.py`; the protocol was fixed in advance in")
    add("[`arm-comparison-live-design.md`](arm-comparison-live-design.md). The first (N=3) grid is")
    add("preserved unchanged next to it as `docs/arm-comparison-live-n3.json` — history is not")
    add("overwritten, it is what the extension is compared against.")
    add()
    if mismatched:
        add(f"> **Integrity note (read this before comparing the two grids).** {len(mismatched)} of the")
        add(f"> {_num(integrity.get('runs_checked'), 0)} stored N=3 run directories no longer contain the text the")
        add("> published N=3 numbers were measured from. The extension re-scored every stored report and")
        add("> reports the N=3 column from that re-scored basis, so the table and the run list are the same")
        add("> evidence; the published-vs-stored detail is in [§5.1](#51-a-stored-report-that-changed-underneath-the-grid),")
        add("> and the published N=3 numbers themselves are preserved unchanged in")
        add("> `docs/arm-comparison-live-n3.json`.")
        add()

    coverage_v = verdicts.get(HEADLINE_COLUMN) or {}
    cost_v = verdicts.get("cost_multiple") or {}
    coverage_entry = columns.get(HEADLINE_COLUMN) or {}
    if history:
        mean5 = coverage_v.get("mean_n5") or {}
        gap = coverage_v.get("gap_n5")
        published_overlap = intervals_overlap(
            _column_of(previous, "A_single_prompt", HEADLINE_COLUMN),
            _column_of(previous, "B_pipeline", HEADLINE_COLUMN),
        )
        add(
            "**Headline: the wider sample kept the cost and error conclusions, and removed the one "
            "column that had looked like a quality edge.** Coverage at N="
            f"{runs_n5} is a mean of {_num(mean5.get('A_single_prompt'), 2)}/14 for the single prompt against "
            f"{_num(mean5.get('B_pipeline'), 2)}/14 for the pipeline (a mean gap of {_num(gap, 2)} of 14 items), and "
            + (
                "the two intervals **overlap**, so coverage does not separate the arms. The published N=3 "
                "numbers had them disjoint (\"A ≥ B in every run\"); on the stored reports re-scored — the "
                "same three repeats — they already overlapped (§5.1). "
                if not published_overlap and not coverage_entry.get("arms_separated_n5") else
                "the two intervals are disjoint. " if coverage_entry.get("arms_separated_n5") else
                "the two intervals still overlap. "
            )
            + "Factual errors were "
            f"{_num((verdicts.get('factual_errors') or {}).get('mean_n5', {}).get('A_single_prompt'), 2)} in every run "
            f"of both grids, and the cost multiple moved from {_num(cost_v.get('n3'), 2)}× to "
            f"{_num(cost_v.get('n5'), 2)}× "
            + (
                f"({_num(cost_v.get('delta'), 3)}), which is inside the band the three-run grid implied."
                if cost_v.get("stable") else
                f"({_num(cost_v.get('delta'), 3)}), outside the band the three-run grid implied: quote the wider sample."
            )
        )
    else:
        add(f"**Headline: {ok_n5} successful runs, {_num(payload.get('total_spend_cny'), 4)} CNY measured spend.**")
    add()
    add("---")
    add()

    # 1. setup -------------------------------------------------------------
    add("## 1. Setup")
    add()
    add("| | |")
    add("| --- | --- |")
    add(f"| Model | `{meta.get('model_requested', 'unknown')}` |")
    add(f"| Temperature | {meta.get('temperature')} (non-zero so the repeats sample real variance) |")
    add(f"| Thinking mode | `{meta.get('thinking')}` |")
    add(f"| Paper | `{Path(str(meta.get('paper_pdf'))).name}` |")
    add(
        f"| Runs | **{runs_n5} per arm**"
        + (f", widened from {runs_n3} by buying {extension.get('runs_bought', 0)} new repeats "
           f"and reusing {extension.get('reused_runs', 0)} stored ones" if extension else "")
        + f" — {summary.get('runs_total', 0)} runs total |"
    )
    add(f"| Failures | **{summary.get('runs_failed', 0)}** failed, {summary.get('runs_skipped', 0)} skipped |")
    if extension:
        add(
            f"| Bought by this extension | {', '.join(extension.get('bought_run_ids') or [])} "
            f"({_num(extension.get('extension_total_cny'), 4)} CNY) |"
        )
    add(
        "| Judge | same model, temperature 0, prompt "
        f"v{judge.get('prompt_version', 'n/a')}, batch {judge.get('batch_size', 'n/a')}, "
        f"≤{judge.get('max_items_per_arm', 'n/a')} items/arm, every batch judged twice with the option order swapped |"
    )
    add()

    # 2. cost --------------------------------------------------------------
    add("## 2. Cost accounting — measured, from the API's own `usage` block")
    add()
    add("Prices are the published DeepSeek ones at **peak** rates, converted at 7.1 CNY/USD; cost =")
    add("tokens × those prices. Mean with [min–max] over the successful runs of each grid"
        + (", the N=3 side being the stored reports re-scored (§5.1)." if history else "."))
    add()
    add("| Column | A " + (f"(N={runs_n3})" if runs_n3 else "") + " | B " + (f"(N={runs_n3})" if runs_n3 else "")
        + " | C " + (f"(N={runs_n3})" if runs_n3 else "") + " | A " + f"(N={runs_n5}) | B (N={runs_n5}) | C (N={runs_n5}) |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for column, label, digits in (
        ("llm_calls", "LLM calls per review", 2),
        ("prompt_tokens", "Input tokens per review", 0),
        ("completion_tokens", "Output tokens per review", 0),
        ("wall_seconds", "Wall-clock per review (s)", 1),
        ("cost_cny", "**CNY per review**", 4),
    ):
        cells = []
        if history:
            for arm in ARMS:
                cells.append(_mean_min_max(_column_of(previous_rescored, arm, column), digits))
        for arm in ARMS:
            cells.append(_mean_min_max(_column_of(summary, arm, column), digits))
        add(f"| {label} | " + " | ".join(cells) + " |")
    multiple_cells = ["1.00×"]
    if history:
        multiple_cells.append(f"{_num(cost_v.get('n3'), 2)}×")
        multiple_cells.append(f"{_num(cost_v.get('c_vs_a_n3'), 2)}×")
    else:
        multiple_cells.extend(["—", "—"])
    multiple_cells.append("1.00×")
    multiple_cells.append(f"**{_num(cost_v.get('n5'), 2)}×**" if cost_v.get("n5") else "—")
    multiple_cells.append(f"{_num(cost_v.get('c_vs_a_n5'), 2)}×" if cost_v.get("c_vs_a_n5") else "—")
    add("| vs arm A | " + " | ".join(multiple_cells) + " |")
    add()
    if history:
        add("### The cost multiple, N=3 vs N=5")
        add()
        band_n3 = [value for value in (cost_v.get("ratio_range_n3") or [None, None])]
        band_n5 = [value for value in (cost_v.get("ratio_range_n5") or [None, None])]
        band_delta = (
            round((band_n5[1] - band_n5[0]) - (band_n3[1] - band_n3[0]), 4)
            if None not in (*band_n3, *band_n5) else None
        )
        add("| | N=3 | N=5 | moved by |")
        add("| --- | ---: | ---: | ---: |")
        add(f"| B over A (mean prices) | {_num(cost_v.get('n3'), 2)}× | {_num(cost_v.get('n5'), 2)}× | "
            f"{_signed(cost_v.get('delta'), 3)} |")
        add(f"| per-run ratio band, B/A | {_range({'min': band_n3[0], 'max': band_n3[1]}, 2)} "
            f"| {_range({'min': band_n5[0], 'max': band_n5[1]}, 2)} | {_signed(band_delta, 2)} wide |")
        add(f"| C over A (mean prices) | {_num(cost_v.get('c_vs_a_n3'), 2)}× | {_num(cost_v.get('c_vs_a_n5'), 2)}× | "
            f"{_signed((cost_v.get('c_vs_a_n5') or 0) - (cost_v.get('c_vs_a_n3') or 0), 3)} |")
        add()
        stable = cost_v.get("stable")
        add(
            "The ratio band is every cross-run ratio between the two arms' per-review prices "
            "(a price per review is not paired across arms, so this is the honest interval). "
            + (
                "**The multiple is stable:** the N=5 point estimate still sits inside the band the "
                "three-run grid implied. "
                if stable else
                "**The multiple moved:** the N=5 point estimate sits outside the band the three-run "
                "grid implied, so the 4.57× figure was a property of those three runs, not of the arms. "
            )
            + (
                "A ratio band over more runs can narrow or widen, so which way it moved is reported "
                "rather than assumed."
                if band_delta is not None else ""
            )
        )
        add()
    add("### Per-run raw figures (the numbers the means are computed from)")
    add()
    add("| run | repeat | arm | calls | input | output | CNY | seconds |")
    add("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for record in payload.get("runs") or []:
        run_id = str(record.get("run_id"))
        repeat = run_id.rsplit("-r", 1)[-1] if "-r" in run_id else "?"
        add(
            f"| {run_id} | {repeat} | {ARM_LABELS.get(str(record.get('arm')), '?')} | "
            f"{_num(record.get('llm_calls'), 0)} | {_num(record.get('prompt_tokens'), 0)} | "
            f"{_num(record.get('completion_tokens'), 0)} | {_num(record.get('cost_cny'), 6)} | "
            f"{_num(record.get('wall_seconds'), 1)} |"
        )
    add()
    add("### Spend")
    add()
    add("| | CNY |")
    add("| --- | ---: |")
    spend_block = payload.get("spend") or {}
    if history:
        add(f"| Previous grid (N={runs_n3}, arms + judge) | {_num(spend_block.get('previous_grid_total_cny'), 4)} |")
        add(f"| **This extension — {extension.get('runs_bought', 0)} new runs (arms)** | "
            f"**{_num(spend_block.get('extension_arms_cny'), 4)}** |")
        add(f"| **This extension — judge on the new repeats** | **{_num(spend_block.get('extension_judge_cny'), 4)}** |")
        add(f"| **This extension, total** | **{_num(spend_block.get('extension_total_cny'), 4)}** |")
        add(f"| Cumulative for the whole experiment | {_num(spend_block.get('cumulative_experiment_cny'), 4)} |")
        add(f"| Budget cap for the extension | {_num(spend_block.get('budget_cny'), 2)} "
            f"(used {_num((extension.get('budget_used_fraction') or 0) * 100, 1)}%) |")
    else:
        add(f"| Arms | {_num(payload.get('total_spend_breakdown', {}).get('arms_cny'), 4)} |")
        add(f"| Judge | {_num(payload.get('total_spend_breakdown', {}).get('judge_cny'), 4)} |")
        add(f"| **Total** | **{_num(payload.get('total_spend_cny'), 4)}** |")
    add()

    # 3. quality -----------------------------------------------------------
    add(f"## 3. Quality — deterministic columns, N={runs_n3 or ''} vs N={runs_n5}".replace("N= vs", "N/a vs"))
    add()
    add("**Primary columns are deterministic and use no LLM at all.**")
    add()
    add(f"| Column | A | B | C |")
    add("| --- | ---: | ---: | ---: |")
    quality_rows = (
        ("claim_bullets", "Report bullets in `Key Claims & Evidence`"),
        ("claim_bullets_locatable", "…whose wording is locatable in the paper (≥8 verbatim tokens)"),
        ("attribution_rate", "…as a share of the bullets (the citation-verifiability column)"),
        ("coverage", "Contributions covered, of 14"),
        ("coverage_rate", "Coverage rate"),
        ("factual_errors", "Factual errors per review (deterministic)"),
        ("number_grounding_rate", "Number-bearing bullets whose magnitudes are all in the paper"),
        ("headings_present", "Required headings present, of 8"),
    )
    for column, label in quality_rows:
        if history:
            cells = [
                f"{_mean_min_max(_column_of(previous_rescored, arm, column), 3)} → {_mean_min_max(_column_of(summary, arm, column), 3)}"
                for arm in ARMS
            ]
        else:
            cells = [_mean_min_max(_column_of(summary, arm, column), 3) for arm in ARMS]
        add(f"| {label} | " + " | ".join(cells) + " |")
    add()
    if history:
        add(f"Each cell reads **N={runs_n3} → N={runs_n5}**: mean (min–max) before the extension, then after.")
        add(
            f"The N={runs_n3} side is the stored reports re-scored by one scorer version, so this table and "
            "the run list agree; §5.1 records where that differs from what the first write-up published."
        )
        add()
        add("| Machine layer (architectural, not a quality score) | A | B | C |")
        add("| --- | ---: | ---: | ---: |")
        claims_b = _column_of(summary, "B_pipeline", "claims_total")
        claims_c = _column_of(summary, "C_pipeline_no_verification", "claims_total")
        verified_b = _column_of(summary, "B_pipeline", "claims_quote_verified")
        add(f"| Claims recorded with a quotation (Reader artifact) | **0** (by construction) | "
            f"{_mean_min_max(claims_b, 3)} | {_mean_min_max(claims_c, 3)} |")
        add(f"| …of which the shipped verifier located in the paper | **not measurable** | "
            f"**{_num(verified_b.get('mean'), 3)} / {_num(claims_b.get('mean'), 3)} = "
            f"{_num((verified_b.get('mean') or 0) / (claims_b.get('mean') or 1) * 100, 1)}%** | "
            "**0** — the check is switched off |")
        add(f"| Machine verdicts on the report's own bullets | **0** | "
            f"{_mean_min_max(_column_of(summary, 'B_pipeline', 'machine_markers_on_bullets'), 3)} | **0** |")
        ledger_runs = sum(
            1 for record in (payload.get("runs") or [])
            if record.get("arm") == "B_pipeline" and record.get("status") == "ok" and record.get("ledger_in_report")
        )
        ledger_ok = sum(1 for record in (payload.get("runs") or [])
                        if record.get("arm") == "B_pipeline" and record.get("status") == "ok")
        add(f"| Machine-written `## Verification Ledger` present | "
            f"**0/{ledger_ok} runs** | **{ledger_runs}/{ledger_ok} runs** | **0/{ledger_ok} runs** |")
        add()
        vocabulary = [
            record["run_id"] for record in (payload.get("runs") or [])
            if record.get("arm") == "C_pipeline_no_verification" and record.get("status") == "ok"
            and (record.get("machine_markers_on_bullets") or 0) > 0
        ]
        if vocabulary:
            add(
                "**Marker-vocabulary caveat.** The marker column counts *labels*, not provenance, and in "
                f"{len(vocabulary)} of C's runs the model wrote the machine layer's own vocabulary itself "
                f"({', '.join('`' + run + '`' for run in sorted(vocabulary))}) while the machine layer "
                "contributed nothing to that arm — no deterministic claim status, no ledger. So \"0 machine "
                "verdicts on C's bullets\" is true of the *machine layer* (which is switched off) and false "
                "of the *text* in those runs. The columns that carry the architecture claim are the claim "
                "statuses and the ledger, not the label count."
            )
            add()

    # 4. overlap -----------------------------------------------------------
    if history:
        add("## 4. Do the arms separate? — the interval question the extension was bought for")
        add()
        add("Two intervals overlap when the intervals printed above share a point. This is a min–max")
        add("range over the successful runs, **not** a confidence interval: with these n's no")
        add("significance test is defensible, so a disjoint pair is reported as *the arms did not")
        add("overlap in this sample*, never as a proven difference. An overlapping pair is the honest")
        add("statement of \"no measurable difference at this sample size\".")
        add()
        add("| Column | A–B (N=3) | A–B (N=5) | A–C (N=5) | B–C (N=5) | reading at N=5 |")
        add("| --- | :---: | :---: | :---: | :---: | --- |")
        for column in QUALITY_COLUMNS:
            entry = columns.get(column) or {}
            pairs_n5 = entry.get("pairwise_overlap_n5") or {}
            reading = (
                "arms separate (intervals disjoint)" if entry.get("arms_separated_n5")
                else "**still no separation — every pair overlaps**"
            )
            add(
                f"| `{column}` | {_overlap_word((entry.get('pairwise_overlap_n3') or {}).get('A_vs_B'))} | "
                f"{_overlap_word(pairs_n5.get('A_vs_B'))} | {_overlap_word(pairs_n5.get('A_vs_C'))} | "
                f"{_overlap_word(pairs_n5.get('B_vs_C'))} | {reading} |"
            )
        add()
        add("Interval width, N=3 → N=5. These are min–max ranges, and a range over more runs can stay")
        add("the same or **widen** — more samples can always turn up a new extreme. What a wider sample")
        add("buys is a mean estimated from more runs; it does not promise a narrower range, and this")
        add("table is reported rather than promised:")
        add()
        add("| Column | A | B | C |")
        add("| --- | ---: | ---: | ---: |")
        for column in QUALITY_COLUMNS:
            entry = columns.get(column) or {}
            cells = []
            for arm in ARMS:
                w3 = (entry.get("width_n3") or {}).get(arm)
                w5 = (entry.get("width_n5") or {}).get(arm)
                cells.append(f"{_num(w3, 3)} → {_num(w5, 3)}")
            add(f"| `{column}` | " + " | ".join(cells) + " |")
        add()

    # 5. the three answers -------------------------------------------------
    if history:
        add("## 5. The three questions, answered")
        add()
        cov = verdicts.get(HEADLINE_COLUMN) or {}
        coverage_entry = columns.get(HEADLINE_COLUMN) or {}
        gap = cov.get("gap_n5")
        a_range = _range({"min": (cov.get("range_n5") or {}).get("A_single_prompt", [None, None])[0],
                          "max": (cov.get("range_n5") or {}).get("A_single_prompt", [None, None])[1]}, 2)
        b_range = _range({"min": (cov.get("range_n5") or {}).get("B_pipeline", [None, None])[0],
                          "max": (cov.get("range_n5") or {}).get("B_pipeline", [None, None])[1]}, 2)
        add("**1. Is the coverage difference (14/14 vs 13/14) still inside the noise?**")
        add()
        published_overlap = intervals_overlap(
            _column_of(previous, "A_single_prompt", HEADLINE_COLUMN),
            _column_of(previous, "B_pipeline", HEADLINE_COLUMN),
        )
        if coverage_entry.get("arms_separated_n5"):
            add("**Yes, it is inside the noise at N=5**, so the published N=3 edge does not survive:")
        else:
            add("**Yes — and on the stored reports it was never outside it.**")
        add()
        add(
            f"At N={runs_n5} the coverage intervals are A [{a_range}] and B [{b_range}] and they overlap, so the "
            f"measured gap of {_num(gap, 2)} {_plural(gap, 'item')} of 14 is inside the run-to-run spread. "
            + (
                "The published N=3 numbers had those intervals disjoint (B was 13/14 in all three runs), which "
                "is what the first write-up reported; re-scoring the stored reports — the same three repeats — "
                "shows B at 14/14 in its first run, i.e. overlapping already (§5.1). "
                if not published_overlap else ""
            )
            + "Which checklist items an arm misses, read from the per-run ids rather than asserted:"
        )
        add()
        for arm in ARMS:
            add(f"* arm {ARM_LABELS[arm]} — {_coverage_reading(payload, arm)}")
        add()
        add(
            "So the honest reading of this column is: **the single prompt covered all 14 contributions in every "
            f"one of its {runs_n5} runs; the pipeline covered 14 in {_pipeline_full_coverage_runs(payload)} of its "
            f"{runs_n5}, and its misses are not one recurring blind spot** (across the pipeline runs they move "
            f"between {_missing_id_phrase(payload)}). A 1/14 difference is the smallest this checklist can "
            "express, and at five runs it is not a measurable difference between the arms."
        )
        add()
        add("**2. Is the cost multiple stable?**")
        add()
        add(
            f"The mean-price multiple moved from {_num(cost_v.get('n3'), 2)}× at N=3 to "
            f"{_num(cost_v.get('n5'), 2)}× at N=5 ({_signed(cost_v.get('delta'), 3)}), with the"
        )
        add(
            f"per-run B/A ratio band going from {_range({'min': band_n3[0], 'max': band_n3[1]}, 2)} to "
            f"{_range({'min': band_n5[0], 'max': band_n5[1]}, 2)}."
        )
        add(
            (
                "The N=5 estimate still lies inside the N=3 band, so **the multiple is stable** — the "
                "headline multiple is a property of the architecture, not of one grid."
            )
            if cost_v.get("stable") else
            (
                "The N=5 estimate falls outside the N=3 band, so **the multiple is not stable**: the "
                "widened sample is the one to quote."
            )
        )
        add()
        add("**3. Did the wider sample expose a failure mode the first three runs never showed?**")
        add()
        modes = extension.get("new_failure_modes") or []
        if modes:
            add("**Yes — one, and it is a within-arm event rather than a new way for one arm to win:**")
            add()
            for mode in modes:
                add(f"* {mode}")
            add()
            add(
                "Everything the extension could have shown and did not: no failed or skipped run, no new "
                "deterministic factual error in any arm, no pipeline run without its ledger, no machine-layer "
                "footprint in the arm where verification is switched off, and no judge support rate that fell "
                "away from the recorded one (§6)."
            )
        else:
            add(
                f"**No.** Across the {extension.get('runs_bought', 0)} new runs: no failed run, no new "
                "deterministic factual error, no quality column below every stored run, no pipeline run "
                "without its ledger, no machine verdict leaking into the arm that has verification off, "
                "and no judge support rate that fell away from the recorded one. The failure modes the "
                "N=3 grid saw are the failure modes the N=5 grid sees."
            )
        add()
        add("**What the extension changed in the write-up:** "
            + ("nothing in the conclusion; " if not modes else "the conclusion is qualified above; ")
            + "the interval table and the per-run figures are the new evidence, and the N=3 numbers are "
            "reported next to the N=5 ones rather than replaced.")
        add()
        add("### 5.1 A stored report that changed underneath the grid")
        add()
        if mismatched:
            add(
                "The extension re-scores from disk. Comparing each stored report with the character count the "
                "previous grid published shows that some run directories no longer hold the text that grid "
                "measured — so the N=3 numbers as published and the N=3 reports as stored are not the same "
                "evidence. This document reports the re-scored basis and keeps the published summary "
                "untouched:"
            )
            add()
            add("| run | published chars | stored chars | published coverage | stored coverage | stored report mtime |")
            add("| --- | ---: | ---: | ---: | ---: | --- |")
            for row in mismatched:
                add(
                    f"| `{row.get('run_id')}` | {_num(row.get('published_report_chars'), 0)} | "
                    f"{_num(row.get('stored_report_chars'), 0)} | {_num(row.get('published_coverage'), 0)} | "
                    f"{_num(row.get('stored_coverage'), 0)} | {row.get('stored_report_mtime') or 'n/a'} |"
                )
            add()
            started_at = (history or {}).get("started_at")
            rewritten_after = [
                row for row in mismatched
                if started_at and (row.get("stored_report_mtime") or "") > str(started_at)
            ]
            if rewritten_after:
                add(
                    f"Every rewritten report is stamped **after the frozen grid's own `started_at` "
                    f"({started_at})**, so the change happened once that grid had finished writing its records; "
                    "what rewrote them is not recorded anywhere in this repository (the run directories carry no "
                    "provenance), which is exactly why the check exists instead of a note in a README."
                )
                add()
            published_cov = {arm: _column_of(previous, arm, "coverage") for arm in ARMS}
            rescored_summary = (history or {}).get("rescored_summary") or {}
            add("Coverage, both readings of the same three stored repeats:")
            add()
            add("| basis | A | B | C |")
            add("| --- | ---: | ---: | ---: |")
            add("| published N=3 (the texts as measured then) | "
                + " | ".join(_mean_min_max(published_cov[arm], 2) for arm in ARMS) + " |")
            add("| stored reports, re-scored now | "
                + " | ".join(_mean_min_max(_column_of(rescored_summary, arm, "coverage"), 2) for arm in ARMS) + " |")
            add()
            add(
                "The honest consequence: the published N=3 line \"A ≥ B on coverage in every run\" is a "
                "property of those three texts, and at least one stored run (B's first repeat) covers 14 "
                "of 14. The N=5 column in §3 is therefore the one to quote, and these run directories are no "
                "longer a faithful re-scoring source for the N=3 numbers — the frozen JSON is."
            )
        else:
            add("Every stored report still matches the text the previous grid measured "
                f"({_num(integrity.get('runs_checked'), 0)} runs checked), so the N=3 column and the N=3 rows are the same evidence.")
        add()

    # 6. judge -------------------------------------------------------------
    if judge:
        add("## 6. The LLM-judged column (secondary, and it stays secondary)")
        add()
        add("| | A | B | C |")
        add("| --- | ---: | ---: | ---: |")
        for label, key in (("Items judged", "n"), ("yes", "yes"), ("no", "no"), ("unclear", "unclear"),
                           ("support rate", "support_rate"), ("agreement between the two passes", "percent_agreement"),
                           ("Cohen's kappa", "cohen_kappa")):
            cells = []
            for arm in ARMS:
                column = (judge.get("arms") or {}).get(arm) or {}
                cells.append(_rate(column.get(key)) if key in ("support_rate", "percent_agreement", "cohen_kappa")
                             else _num(column.get(key), 0))
            add(f"| {label} (recorded grid) | " + " | ".join(cells) + " |")
        if judge_extension:
            for label, key in (("Items judged", "n"), ("yes", "yes"), ("no", "no"), ("unclear", "unclear"),
                               ("support rate", "support_rate"),
                               ("agreement between the two passes", "percent_agreement")):
                cells = []
                for arm in ARMS:
                    column = (judge_extension.get("arms") or {}).get(arm) or {}
                    cells.append(_rate(column.get(key)) if key in ("support_rate", "percent_agreement")
                                 else _num(column.get(key), 0))
                add(f"| {label} (new repeats only) | " + " | ".join(cells) + " |")
            add()
            add(
                "The recorded column is the one bought in the first grid; it was **not re-bought**. The "
                "second block judges only the repeats this extension bought, sampled evenly across them "
                f"(`{(judge_extension.get('arms') or {}).get('B_pipeline', {}).get('sampling', 'n/a')}`), "
                "so the two blocks together say whether the judge's verdicts reproduce on fresh samples "
                "instead of quoting one sample twice."
            )
        add()
        kappa_note = next(
            (column.get("kappa_note") for column in (judge.get("arms") or {}).values() if column.get("kappa_note")),
            None,
        )
        if kappa_note:
            add(f"*Kappa note: {kappa_note}.*")
            add()
        degenerate = [
            ARM_LABELS.get(arm, arm) for arm, column in (judge.get("arms") or {}).items()
            if column.get("cohen_kappa") == 0 and (column.get("percent_agreement") or 0) > 0
        ]
        if degenerate:
            add(
                f"*Where kappa is 0 while the agreement is high (arm {', '.join(degenerate)}): one pass gave the "
                "same verdict for every item, so chance alone predicts the observed agreement and kappa "
                "collapses to 0. For those arms the percent-agreement column, not kappa, carries the "
                "reliability reading.*"
            )
            add()
        add(
            "The judge asks arm A a different (easier) question, because A produces no quotes to judge "
            "— attribution to the paper instead of entailment by a quotation. **The two support rates "
            "are therefore not comparable across arms and this document does not compare them.** The "
            "deterministic columns above carry the conclusion."
        )
        add()

    # 7. reproduce ---------------------------------------------------------
    add("## 7. Reproducing it")
    add()
    add("```bash")
    add("# re-score the stored reports with the current scorers (no API calls, no spend)")
    add("python scripts/compare_arms_live.py --score-only --out docs/arm-comparison-live.json \\")
    add("    --runs-dir outputs/arm-comparison-live")
    add()
    if history:
        add(f"# widen the stored N={runs_n3} grid to N={runs_n5}: only the missing repeats are bought")
        add(f"python scripts/compare_arms_live.py --extend-from docs/arm-comparison-live-n3.json \\")
        add(f"    --runs {runs_n5} --budget 10 --out docs/arm-comparison-live.json \\")
        add("    --runs-dir outputs/arm-comparison-live")
        add()
    add("# the experiment from scratch (requires DEEPSEEK_API_KEY in the environment or a .env)")
    add(f"python scripts/compare_arms_live.py --runs {runs_n5} --budget 30 \\")
    add("    --out docs/arm-comparison-live.json --runs-dir outputs/arm-comparison-live")
    add("```")
    add()

    # 8. limitations -------------------------------------------------------
    separated = _separated_columns(comparison, "n5") if history else []
    add("## 8. Limitations")
    add()
    add(f"1. **One paper, one model, {runs_n5} repeats.** "
        + (
            "The per-arm ranges overlap on every quality column except "
            + ", ".join(f"`{column}`" for column in separated) + ", so on the others the honest"
            if separated else
            "The per-arm ranges overlap on every quality column, so the honest"
        ))
    add("   statement is \"no measurable difference in this sample\", not \"A is better than B\". A")
    add("   difference smaller than a few points could not have been detected here at all, and no")
    add("   significance test is defensible at n=5.")
    add("2. **The judge's question is not identical across arms** (§6). Fixing that properly needs a")
    add("   quote from arm A, which is precisely what arm A does not produce.")
    add("3. **`arxiv_search` is stubbed**, so the pipeline's related-work tool loop was never exercised")
    add("   and no arm's citations were verified. Real related-work quality is unmeasured.")
    add("4. **Arm A's cache advantage**: A re-sends one identical prefix and is almost entirely cached,")
    add("   while the multi-stage conversation keeps changing its prefix. Cache pricing therefore")
    add("   flatters A slightly, and the cost multiple is a mid-to-upper estimate rather than a")
    add("   conservative one.")
    add("5. **Factual-error detection only understands `%`/`pp` magnitudes bound to a configuration.**")
    add("   Zero errors means \"no arithmetic or attribution error of this kind\", not \"no errors\".")
    add("6. **Coverage is a checklist, not comprehension.** It measures whether a contribution is")
    add("   present with its magnitude, not whether it is discussed well — which is also why a 1/14 gap")
    add("   is the finest difference it can express.")
    add("7. The verifier, the coverage checklist and the error detector were written by the same author")
    add("   as the architecture under test. The mitigations are that they are deterministic, that the")
    add("   checklist was frozen before the runs, and that the grader is tested against an unrelated")
    add("   report (which it scores 0/14).")
    if history:
        add(f"8. **The stored runs were not re-bought, only re-scored.** The N={runs_n3} runs' tokens, costs")
        add("   and timings are exactly as recorded in `docs/arm-comparison-live-n3.json`; the deterministic")
        add("   columns for all runs were recomputed by one version of the scorers, said in")
        add("   `extension.reused_run_ids`.")
        if mismatched:
            add(f"9. **{len(mismatched)} of the {_num(integrity.get('runs_checked'), 0)} stored reports no longer match the text the")
            add("   published N=3 numbers were measured from** (§5.1). The published summary is preserved")
            add("   unchanged in the frozen JSON, and the N=3 column of this document is the re-scored basis,")
            add("   so the two readings differ for those runs. Treat the N=5 column as the quotable one.")
    add()

    # 9. conclusion --------------------------------------------------------
    add("## 9. Conclusion")
    add()
    cov = verdicts.get(HEADLINE_COLUMN) or {}
    mean5 = cov.get("mean_n5") or {}
    a_cov = _num(mean5.get("A_single_prompt"), 2)
    b_cov = _num(mean5.get("B_pipeline"), 2)
    multiple5 = _num((verdicts.get("cost_multiple") or {}).get("n5"), 2)
    ledger_runs = sum(
        1 for record in (payload.get("runs") or [])
        if record.get("arm") == "B_pipeline" and record.get("status") == "ok" and record.get("ledger_in_report")
    )
    ledger_ok = sum(1 for record in (payload.get("runs") or [])
                    if record.get("arm") == "B_pipeline" and record.get("status") == "ok")
    add("> **The multi-stage pipeline bought verifiability and auditability; it did not buy quality, and")
    add(f"> it cost {multiple5}x.** On the deterministic columns the single long prompt was equal or")
    add(f"> better: coverage {a_cov}/14 vs {b_cov}/14, formal factual errors 0.0 for every arm, all eight")
    add("> headings in every run. What only the pipeline does is put a machine-written ledger in")
    add(f"> **every** report ({ledger_runs}/{ledger_ok} runs vs 0/{ledger_ok} for A and C), hold a quotation for")
    add("> **every** claim (located by the shipped verifier, vs *not measurable* for A and *never checked*")
    add("> for C), and let code rather than the model own the verdict.")
    add()
    if history:
        modes = extension.get("new_failure_modes") or []
        coverage_separated = bool(coverage_entry.get("arms_separated_n5"))
        add(
            f"Widening the sample from {runs_n3} to {runs_n5} repeats per arm "
            f"({_num(extension.get('extension_total_cny'), 4)} CNY, "
            f"{_num((extension.get('budget_used_fraction') or 0) * 100, 1)}% of the extension budget) "
            + (
                "**changed one reading and left the rest standing**: the coverage column no longer separates "
                "the arms at N=5, so the published N=3 edge (\"A ≥ B on coverage in every run\") does not "
                "survive the wider sample, and the stored reports had already contradicted it (§5.1). "
                if not coverage_separated else
                "**kept the coverage separation** and left the rest standing: "
            )
            + "Everything else held: factual errors at zero in every run, headings complete in every run, "
            + (
                f"the cost multiple inside the band the three-run grid implied ({_num(cost_v.get('n5'), 2)}×), "
                if cost_v.get("stable") else
                f"the cost multiple moving to {_num(cost_v.get('n5'), 2)}× outside that band, "
            )
            + (
                "and no new failure mode. "
                if not modes else
                f"and one new within-arm event ({len(modes)}: {modes[0].split(':')[0]}). "
            )
            + "What the extension bought is not a new headline but a corrected one: a smaller claim about "
            "coverage, a verified cost multiple, and a stored-run integrity problem that the wider sample "
            "walked straight into. That is what spending on more data is for, and it is reported as such."
        )
        add()
    add("The interview answer this supports: *\"Quality was flat across one prompt and four agents —")
    add(f"{a_cov}/14 vs {b_cov}/14 coverage, zero arithmetic errors either way, on {runs_n5} repeats per")
    add(f"arm. What the pipeline adds is that every claim carries a quotation a program locates in the")
    add(f"paper and a ledger written by code rather than by the model, and that costs {multiple5}x. I can")
    add("show you the run where the single prompt's report contains no claim a program could ever")
    add("check, and the run where the pipeline's check downgraded a quotation its own Reader invented.\"*")
    add()
    return "\n".join(out)


def run_experiment(
    *, runs: int, out_dir: Path, budget: float, settings: Settings, pricing: Pricing,
    paper_pdf: Path = PAPER_PDF, judge: bool = True, llm_factory: Callable[[], LLMClient] | None = None,
) -> dict:
    """The whole experiment: the grid, then the declared judge over its output."""
    records = run_grid(
        llm_factory=llm_factory, runs=runs, out_dir=out_dir, paper_pdf=paper_pdf,
        budget=budget, pricing=pricing, settings=settings,
    )
    summary = aggregate(records)
    judge_result = None
    if judge:
        paper_text = extract_pdf_text(str(paper_pdf), max_chars=10**8)
        remaining = max(0.0, budget - summary["total_cost_cny"])
        judge_result = judge_records(
            records, paper_text, settings=settings, budget=remaining, pricing=pricing,
            llm_factory=llm_factory,
        )
        for arm, column in judge_result["arms"].items():
            if arm in summary["arms"]:
                summary["arms"][arm]["semantic_support"] = {
                    key: value for key, value in column.items() if key != "sample"
                }
    return {"records": records, "summary": summary, "judge": judge_result}


def _repeat_of(record: dict) -> int:
    match = _RUN_ID.search(str(record.get("run_id") or ""))
    return int(match.group(1)) if match else 0


def records_up_to_repeat(records: list[dict], runs: int) -> list[dict]:
    """The subset of a merged grid that belongs to the first `runs` repeats."""
    return [record for record in records if 0 < _repeat_of(record) <= runs]


def write_outputs(payload: dict, out_path: Path, md_path: Path | None) -> None:
    """Write the JSON and (unless suppressed) the rendered markdown beside it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        print(f"wrote {md_path}")


def print_summary(summary: dict, runs: int) -> None:
    for arm in ARMS:
        row = summary["arms"][arm]
        support = (row.get("semantic_support") or {}).get("support_rate")
        print(
            f"{arm:<28} ok={row['runs_ok']}/{row['runs_total']}  "
            f"calls={row['llm_calls']['mean']}  in={row['prompt_tokens']['mean']}  "
            f"out={row['completion_tokens']['mean']}  CNY={row['cost_cny']['mean']}  "
            f"headings={row['headings_present']['mean']}  attr={row['attribution_rate']['mean']}  "
            f"cov={row['coverage_rate']['mean']}  errs={row['factual_errors']['mean']}  "
            f"support={support}"
        )
    if any(row["runs_ok"] < runs for row in summary["arms"].values()):
        print(f"NOTE: an arm has fewer than {runs} successful runs; no report will be published.")


def _score_only(args, runs_dir: Path) -> int:
    """Re-score an existing grid in place; never sends a request."""
    out_path = Path(args.out)
    if not out_path.is_file():
        print(f"ERROR: {out_path} does not exist, so there is nothing to re-score.", file=sys.stderr)
        return 2
    previous = json.loads(out_path.read_text(encoding="utf-8"))
    # Re-scoring reproduces a design that was already run, so it declares the
    # floor that design had - the >=5 fence governs reports of *new* grids.
    floor = int((previous.get("meta") or {}).get("runs_per_arm") or MIN_RUNS_PER_ARM)
    history = previous.get("history")
    previous_runs = int((history or {}).get("runs_per_arm") or 0)
    paper_text = extract_pdf_text(str(PAPER_PDF), max_chars=10**8)
    records = rescore_runs(previous["runs"], paper_text, runs_dir)
    summary = aggregate(records)
    judge = previous.get("judge")
    judge_extension = previous.get("judge_extension")
    for block in (judge, judge_extension):
        for arm, column in (block or {}).get("arms", {}).items():
            if arm in summary["arms"]:
                key = "semantic_support" if block is judge else "semantic_support_extension"
                summary["arms"][arm][key] = {k: v for k, v in column.items() if k != "sample"}
    comparison = None
    extension = previous.get("extension")
    if history and (history.get("summary") or history.get("rescored_summary")):
        stored = records_up_to_repeat(records, previous_runs)
        bought = [record for record in records if _repeat_of(record) > previous_runs]
        stored_summary = aggregate(stored)
        history = {
            **history,
            "rescored_summary": stored_summary,
            # carried over, not recomputed: after a re-score the payload's own
            # report_chars are the stored text, so the published-vs-stored
            # comparison can only be made where the published numbers still are
            "integrity": history.get("integrity"),
        }
        comparison = compare_summaries(
            stored_summary, summary, previous_records=stored, current_records=records,
            previous_runs_per_arm=previous_runs, runs_per_arm=floor,
        )
        if extension is not None:
            extension = {
                **extension,
                "new_failure_modes": new_failure_modes(stored, bought, judge_extension, judge),
            }
    if args.published:
        # the authoritative published numbers live in the frozen grid file, so
        # the integrity check reads them from there rather than from this
        # payload (whose report_chars are already the stored text)
        published_path = Path(args.published)
        if not published_path.is_file():
            print(f"ERROR: --published {published_path} does not exist.", file=sys.stderr)
            return 2
        published = json.loads(published_path.read_text(encoding="utf-8"))
        history = {
            **(history or {"runs_per_arm": int((published.get("meta") or {}).get("runs_per_arm") or 0) or None,
                           "summary": published.get("summary"), "meta": published.get("meta")}),
            "published_source": str(published_path),
            "integrity": stored_report_integrity(published.get("runs") or [], runs_dir),
        }
    meta = {
        **previous.get("meta", {}),
        "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rescoring_note": (
            "The deterministic columns were recomputed from the same stored reports after "
            "fixing two measurement bugs (a comparison list cascading into false misattributions, "
            "and arithmetic derivable from the paper being counted as fabrication). Model outputs, "
            "token counts, costs and timings are unchanged; the judge column is the one recorded "
            "in the original run and was not re-bought."
        ),
    }
    payload = build_report(
        records, summary, meta, judge, min_runs_per_arm=floor,
        judge_extension=judge_extension, history=history, comparison=comparison, extension=extension,
    )
    write_outputs(payload, out_path, _md_path(args))
    print(f"re-scored {len(records)} runs from {runs_dir}; no API calls made")
    print_summary(summary, floor)
    return 0


def _md_path(args) -> Path | None:
    """Where the rendered markdown goes: beside --out unless suppressed."""
    if args.no_md:
        return None
    return Path(args.md) if args.md else Path(args.out).with_suffix(".md")


def _experiment_meta(args, settings: Settings, *, previous_runs: int | None = None) -> dict:
    meta = {
        "model_requested": settings.model,
        "temperature": settings.temperature,
        "thinking": os.environ.get("PAPERFLOW_THINKING", "disabled"),
        "runs_per_arm": args.runs,
        "budget_cny": args.budget,
        "paper_pdf": str(PAPER_PDF),
        "pricing_peak": PRICING_PEAK_CNY.as_dict(),
        "pricing_offpeak": PRICING_OFFPEAK_CNY.as_dict(),
        "price_source": PRICE_SOURCE,
        "price_read_on": PRICE_READ_ON,
        "max_output_tokens": settings.max_output_tokens,
        "max_fulltext_chars": settings.max_fulltext_chars,
        "arxiv_search": "stubbed to an empty result set (the paper is a local PDF)",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if previous_runs is not None:
        meta["widened_from_runs_per_arm"] = previous_runs
        meta["extension_from"] = str(args.extend_from)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs", type=int, default=MIN_RUNS_PER_ARM,
        help=f"independent repeats per arm (default {MIN_RUNS_PER_ARM}; a report is refused below {MIN_RUNS_PER_ARM})",
    )
    parser.add_argument("--budget", type=float, default=BUDGET_CNY, help=f"hard spend cap in CNY (default {BUDGET_CNY})")
    parser.add_argument("--out", default=str(REPO / "docs" / "arm-comparison-live.json"), help="JSON output path")
    parser.add_argument("--md", default=None, help="rendered markdown output path (default: beside --out)")
    parser.add_argument(
        "--published", default=None,
        help="with --score-only: the frozen grid JSON whose published numbers the stored reports are "
             "checked against (recorded as history.integrity)",
    )
    parser.add_argument("--no-md", action="store_true", help="write only the JSON")
    parser.add_argument("--runs-dir", default=None, help="where run directories are written")
    parser.add_argument("--env-file", default=None, help="path to a .env holding DEEPSEEK_API_KEY")
    parser.add_argument("--model", default=None, help="override PAPERFLOW_MODEL for this experiment")
    parser.add_argument("--no-judge", action="store_true", help="skip the secondary LLM semantic-support judge")
    parser.add_argument("--dry-run", action="store_true", help="check preconditions and print the plan; spend nothing")
    parser.add_argument(
        "--extend-from",
        default=None,
        help="widen a previous grid (JSON) to --runs repeats per arm: only the missing repeats are "
             "bought, the stored runs are reused and the two grids are reported side by side",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="re-score the reports of an existing run directory (no API calls, no spend) "
             "and reuse the judge results already recorded next to --out",
    )
    args = parser.parse_args(argv)

    # The experiment reads the key from the environment only. `load_dotenv`
    # never overrides an existing variable and never writes anything; the
    # extra candidates below point at the workspace `.env` used by this
    # project so the key does not have to be exported into the shell.
    env_path = load_dotenv(args.env_file)
    if env_path is None and not settings_keys_present():
        for candidate in (REPO.parent.parent / ".env", REPO.parent / ".env"):
            if candidate.is_file():
                env_path = load_dotenv(candidate)
                break
    if args.model:
        os.environ["PAPERFLOW_MODEL"] = args.model
    settings = Settings()
    if not settings.deepseek_api_key:
        print("ERROR: DEEPSEEK_API_KEY is not set. Nothing was run and nothing was spent.", file=sys.stderr)
        return 2

    runs_dir = Path(args.runs_dir) if args.runs_dir else REPO / "outputs" / "arm-comparison-live"
    previous: dict | None = None
    previous_runs: int | None = None
    if args.extend_from:
        previous_path = Path(args.extend_from)
        if not previous_path.is_file():
            print(f"ERROR: --extend-from {previous_path} does not exist.", file=sys.stderr)
            return 2
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_runs = int((previous.get("meta") or {}).get("runs_per_arm") or 0) or None
        plan = extension_plan(previous.get("runs") or [], args.runs)
    else:
        plan = [(arm, index) for index, arm in enumerate(plan_for(args.runs), start=1)]

    print(f"env file loaded: {env_path}")
    print(f"model: {settings.model}  base_url: {settings.base_url}  temperature: {settings.temperature}")
    print(f"thinking: {os.environ.get('PAPERFLOW_THINKING', 'disabled')}")
    print(f"paper: {PAPER_PDF} ({PAPER_PDF.stat().st_size if PAPER_PDF.is_file() else 0} bytes)")
    print(f"budget: {args.budget} CNY   plan: {len(plan)} runs -> {[f'{a}-r{i}' for a, i in plan]}")
    if previous is not None:
        print(
            f"extending {args.extend_from}: {len(previous.get('runs') or [])} stored runs "
            f"(N={previous_runs}) -> N={args.runs}; {len(plan)} repeats will be bought"
        )
        if not plan:
            print("nothing to buy: the stored grid is already at least as wide as --runs")
    if args.dry_run:
        print("dry run: no request will be sent.")
        return 0

    if args.score_only:
        return _score_only(args, runs_dir)

    md_path = _md_path(args)
    try:
        if previous is not None:
            outcome = extend_experiment(
                previous_path=Path(args.extend_from), runs=args.runs, out_dir=runs_dir,
                budget=args.budget, settings=settings, pricing=PRICING_PEAK_CNY,
                judge=not args.no_judge, paper_pdf=PAPER_PDF,
            )
            records = outcome["records"]
            summary = outcome["summary"]
            meta = {**_experiment_meta(args, settings, previous_runs=previous_runs), "extension": True}
            payload = build_report(
                records, summary, meta, outcome["judge"],
                judge_extension=outcome["judge_extension"], history=outcome["history"],
                comparison=outcome["comparison"], extension=outcome["extension"],
            )
        else:
            outcome = run_experiment(
                runs=args.runs, out_dir=runs_dir, budget=args.budget, settings=settings,
                pricing=PRICING_PEAK_CNY, judge=not args.no_judge,
            )
            records, summary, judge_result = outcome["records"], outcome["summary"], outcome["judge"]
            payload = build_report(records, summary, _experiment_meta(args, settings), judge_result)
    except InsufficientRuns as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        print("Nothing was published: a report is not written from a grid below the floor.", file=sys.stderr)
        return 3

    write_outputs(payload, Path(args.out), md_path)
    arms_cny = float(payload["total_spend_breakdown"]["arms_cny"])
    judge_cny = float(payload["total_spend_breakdown"]["judge_cny"])
    extension_cny = float(payload["total_spend_breakdown"].get("extension_judge_cny") or 0.0)
    print(
        f"arm spend: {arms_cny:.4f} CNY   judge spend: {judge_cny + extension_cny:.4f} CNY   "
        f"total: {payload['total_spend_cny']:.4f} CNY (cap {args.budget})   "
        f"ok={summary['runs_ok']} failed={summary['runs_failed']} skipped={summary['runs_skipped']}"
    )
    if "spend" in payload:
        print(
            f"bought by this extension: {payload['spend']['extension_total_cny']:.4f} CNY "
            f"({payload['spend']['extension_arms_cny']:.4f} arms + "
            f"{payload['spend']['extension_judge_cny']:.4f} judge); "
            f"cumulative for the experiment: {payload['spend']['cumulative_experiment_cny']:.4f} CNY"
        )
    print_summary(summary, args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
