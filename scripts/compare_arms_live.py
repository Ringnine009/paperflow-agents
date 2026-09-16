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
from paperflow.tools.texttools import canonical_tokens, normalize_ws  # noqa: E402
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

#: arm execution order, by repeat: A and B are the comparison the result hinges
#: on, so they complete their repeats before C spends anything - an aborted
#: grid then still contains the rows the conclusion needs. `plan_for(runs)`
#: slices this list, so it must stay interleaved by repeat, not grouped by arm.
RUN_ORDER = (
    ["A_single_prompt", "B_pipeline", "C_pipeline_no_verification"] * 3
)


def plan_for(runs: int) -> list[str]:
    """The (arm, repeat) schedule for `runs` repeats per arm, in spend order.

    A full grid is A r1, B r1, C r1, A r2, ... so that truncating it at any
    point leaves a balanced comparison rather than three copies of one arm.
    """
    return RUN_ORDER[: runs * len(ARMS)]


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
    from paperflow.tools.texttools import canonical_text

    tokens = canonical_tokens(text)  # a list: rescanned, never consumed
    canonical = canonical_text(text)
    covered = 0
    for contribution in CONTRIBUTIONS:
        groups_ok = all(
            any(_subsequence(tokens, canonical_tokens(alternative)) for alternative in group)
            for group in contribution["groups"]
        )
        substrings_ok = all(
            needle in canonical for needle in contribution.get("substrings", [])
        )
        if groups_ok and substrings_ok:
            covered += 1
    return covered, len(CONTRIBUTIONS)


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
    if not Path(paper_pdf).is_file():
        raise FileNotFoundError(f"paper PDF not found: {paper_pdf}")
    text = paper_text if paper_text is not None else extract_pdf_text(str(paper_pdf), max_chars=10**8)
    paper_tokens = canonical_tokens(text)
    paper_index = _token_index(text)
    paper_values = paper_number_values(text)
    settings = settings or Settings()

    # interleaved by repeat so the run ids are unique per (arm, repeat) and a
    # grid truncated by the budget stays balanced (see `plan_for`)
    plan = plan_for(runs) if not arms else list(arms)
    per_arm_count: dict[str, int] = {}

    records: list[dict] = []
    spent = 0.0
    for arm in plan:
        index = per_arm_count.get(arm, 0) + 1
        per_arm_count[arm] = index
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
                    arm=arm, settings=settings, out_root=run_dir, paper_pdf=Path(paper_pdf),
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
) -> dict:
    """Run the declared judge over the sampled claim bullets and attach the result.

    Two passes per arm with the option order swapped; the agreement between
    them is the reliability figure reported next to the verdicts. The judge's
    own token cost is measured with the same ledger as the arms.
    """
    llm = _client_for(settings, budget, pricing, llm_factory)
    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        rows = [r for r in records if r.get("arm") == arm and r.get("status") == STATUS_OK]
        items = [item for row in rows for item in (row.get("judge_samples") or [])][:max_per_arm]
        if not items:
            per_arm[arm] = {
                "n": 0, "yes": 0, "no": 0, "unclear": 0, "support_rate": None,
                "measured": False, "reason": "no claim bullets were available to judge",
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


def build_report(records: list[dict], summary: dict, meta: dict, judge: dict | None = None) -> dict:
    spend = summary["total_cost_cny"] + float((judge or {}).get("judge_cost_cny") or 0.0)
    return {
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
            "judge_cny": round(float((judge or {}).get("judge_cost_cny") or 0.0), 6),
        },
        "runs": records,
    }


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


def _score_only(args, runs_dir: Path) -> int:
    """Re-score an existing grid in place; never sends a request."""
    out_path = Path(args.out)
    if not out_path.is_file():
        print(f"ERROR: {out_path} does not exist, so there is nothing to re-score.", file=sys.stderr)
        return 2
    previous = json.loads(out_path.read_text(encoding="utf-8"))
    paper_text = extract_pdf_text(str(PAPER_PDF), max_chars=10**8)
    records = rescore_runs(previous["runs"], paper_text, runs_dir)
    summary = aggregate(records)
    judge = previous.get("judge")
    for arm, column in (judge or {}).get("arms", {}).items():
        if arm in summary["arms"]:
            summary["arms"][arm]["semantic_support"] = {
                key: value for key, value in column.items() if key != "sample"
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
    payload = build_report(records, summary, meta, judge)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"re-scored {len(records)} runs from {runs_dir}; no API calls made")
    print(f"wrote {out_path}")
    for arm in ARMS:
        row = summary["arms"][arm]
        print(
            f"{arm:<28} ok={row['runs_ok']}/{row['runs_total']}  CNY={row['cost_cny']['mean']}  "
            f"headings={row['headings_present']['mean']}  attr={row['attribution_rate']['mean']}  "
            f"cov={row['coverage_rate']['mean']}  errs={row['factual_errors']['mean']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=3, help="independent repeats per arm (default 3)")
    parser.add_argument("--budget", type=float, default=BUDGET_CNY, help=f"hard spend cap in CNY (default {BUDGET_CNY})")
    parser.add_argument("--out", default=str(REPO / "docs" / "arm-comparison-live.json"), help="JSON output path")
    parser.add_argument("--runs-dir", default=None, help="where run directories are written")
    parser.add_argument("--env-file", default=None, help="path to a .env holding DEEPSEEK_API_KEY")
    parser.add_argument("--model", default=None, help="override PAPERFLOW_MODEL for this experiment")
    parser.add_argument("--no-judge", action="store_true", help="skip the secondary LLM semantic-support judge")
    parser.add_argument("--dry-run", action="store_true", help="check preconditions and print the plan; spend nothing")
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
    plan = plan_for(args.runs)
    print(f"env file loaded: {env_path}")
    print(f"model: {settings.model}  base_url: {settings.base_url}  temperature: {settings.temperature}")
    print(f"thinking: {os.environ.get('PAPERFLOW_THINKING', 'disabled')}")
    print(f"paper: {PAPER_PDF} ({PAPER_PDF.stat().st_size if PAPER_PDF.is_file() else 0} bytes)")
    print(f"budget: {args.budget} CNY   plan: {len(plan)} runs -> {plan}")
    if args.dry_run:
        print("dry run: no request will be sent.")
        return 0

    if args.score_only:
        return _score_only(args, runs_dir)

    outcome = run_experiment(
        runs=args.runs, out_dir=runs_dir, budget=args.budget, settings=settings,
        pricing=PRICING_PEAK_CNY, judge=not args.no_judge,
    )
    records, summary, judge_result = outcome["records"], outcome["summary"], outcome["judge"]
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
    payload = build_report(records, summary, meta, judge_result)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"arm spend: {summary['total_cost_cny']:.4f} CNY   "
          f"judge spend: {float((judge_result or {}).get('judge_cost_cny') or 0):.4f} CNY   "
          f"total: {payload['total_spend_cny']:.4f} CNY (cap {args.budget})   "
          f"ok={summary['runs_ok']} failed={summary['runs_failed']} skipped={summary['runs_skipped']}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
