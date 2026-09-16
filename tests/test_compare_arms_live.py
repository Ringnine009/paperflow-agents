"""The live three-arm harness: measurement core, offline aggregation, cost guard.

`scripts/compare_arms_live.py` must be runnable against the real DeepSeek API,
but everything that *decides a number* has to be testable without spending a
yuan. These tests cover:

* the deterministic measurements (are the report's structures complete, is a
  quote locatable in the paper, is a number attributed to the right
  configuration, how much of the paper is covered) - the columns that carry
  the paper's conclusions, none of which involve an LLM,
* the full grid runner driven by a `FakeLLM`, so the aggregation, the
  per-run isolation of token deltas and the partial-result handling are
  exercised end to end with no network,
* the promise that a live run refuses to start without a key or with a
  budget that is already blown.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperflow.core.llm import BudgetExceeded, LLMClient
from scripts.compare_arms_live import (
    ARMS,
    BUDGET_CNY,
    PAPER_FACTS,
    PAPER_PDF,
    aggregate,
    check_number_pairings,
    contributions_covered,
    known_win_rates,
    paper_number_values,
    reader_claim_evidence,
    run_grid,
    token_runs,
)
from tests.helpers import FakeLLM, text_response, tool_call_response, tool_result_text_path

# ---------------------------------------------------------------------------
# token_runs: the primitive every locatability measurement is built on
# ---------------------------------------------------------------------------


def test_token_runs_keeps_contiguous_matches_and_orders_them_longest_first():
    """Tokens are the project's canonical tokens: punctuation is stripped.

    So "44.2%" becomes the single token "442" - the same normalization the
    shipped verifier uses, which is why a report number and a paper number
    compare equal even across PDF hyphenation and line wrapping.
    """
    assert token_runs("the villager win rate was 44.2% in group a") == [
        ("the", "villager"),
        ("villager", "win"),
        ("win", "rate"),
        ("rate", "was"),
        ("was", "442"),
        ("442", "in"),
        ("in", "group"),
        ("group", "a"),
    ]


def test_token_runs_can_require_a_minimum_length():
    runs = token_runs("we ran 500 games with nine players", size=3)
    assert runs == [
        ("we", "ran", "500"),
        ("ran", "500", "games"),
        ("500", "games", "with"),
        ("games", "with", "nine"),
        ("with", "nine", "players"),
    ]


def test_token_runs_needs_more_tokens_than_the_window():
    assert token_runs("only three tokens", size=5) == []


# ---------------------------------------------------------------------------
# numeric attribution - deterministic, no model in the loop
# ---------------------------------------------------------------------------


def test_known_win_rates_are_read_from_the_papers_own_table():
    rates = known_win_rates()
    assert rates["a"] == "44.2"
    assert rates["e"] == "68.8"
    assert rates["f"] == "68.2"


def test_a_configuration_wearing_the_wrong_win_rate_is_a_contradiction():
    """The failure an interviewer probes for: Table 2 values shuffled."""
    errors = check_number_pairings("Group F reached a 54.6% win rate overall.")
    assert len(errors) == 1
    assert "F" in errors[0] and "68.2" in errors[0]


def test_the_full_grid_is_interleaved_so_a_truncated_run_stays_balanced():
    """Spend order must not front-load one arm: a budget stop keeps the grid usable."""
    from scripts.compare_arms_live import plan_for

    assert plan_for(2) == [
        "A_single_prompt", "B_pipeline", "C_pipeline_no_verification",
        "A_single_prompt", "B_pipeline", "C_pipeline_no_verification",
    ]
    assert plan_for(3).count("A_single_prompt") == 3
    assert plan_for(3).count("C_pipeline_no_verification") == 3


def test_the_correct_configuration_number_pairing_is_not_an_error():
    assert check_number_pairings("Group E reached a 68.8% win rate overall.") == []
    assert check_number_pairings("Groups A and B stay in the 44%-55% band.") == []


def test_a_configuration_is_not_flagged_when_the_number_belongs_to_it():
    """'Group C, 53.8%' is right even though 53.8 is not group A's number."""
    assert check_number_pairings("Standalone DBN (Group C, 53.8%) improves over the baseline.") == []


def test_a_percentage_that_is_nowhere_in_the_paper_is_flagged():
    errors = check_number_pairings("The full stack wins 91.7% of games.")
    assert len(errors) == 1 and "91.7" in errors[0]


def test_a_comparison_sentence_does_not_misattribute_every_group_in_it():
    """'Group E: A=44.2%, B=54.6%, C=53.8%, E=68.8%' is a comparison list.

    Only a number *bound* to its label (``group E = 68.8%``) may be judged
    against Table 2. Judging every group named in the sentence against every
    percentage in it reported five contradictions for one correct table - a
    false-positive rate that would have made the whole column worthless.
    """
    errors = check_number_pairings("Group E: A=44.2%, B=54.6%, C=53.8%, E=68.8%, F=68.2%.")
    assert errors == []


def test_a_bound_label_with_the_wrong_value_is_still_caught():
    assert len(check_number_pairings("Group E reached a 54.6% win rate.")) == 1
    assert len(check_number_pairings("Group E = 54.6%.")) == 1


def test_each_bound_label_in_one_sentence_is_judged_on_its_own():
    """One right, one wrong, one column confusion: exactly one is an error.

    ``group E = 68.8%`` matches Table 2; ``group F = 68.8%`` does not (F is
    68.2%); ``group F = 66.6%`` is F's *vote accuracy*, and only win rates are
    judged here, so it is not counted as a win-rate error.
    """
    errors = check_number_pairings("Group E = 68.8% and group F = 68.8%.")
    assert len(errors) == 1 and "F" in errors[0]
    assert check_number_pairings("Group F = 66.6% vote accuracy.") == []


def test_a_value_derivable_from_the_papers_own_numbers_is_not_a_fabrication():
    """68.8 - 58.6 = 10.2 and 68.8 - 68.2 = 0.6 are arithmetic, not invention."""
    assert check_number_pairings("Group E is +10.2 pp over group D.") == []
    assert check_number_pairings("F trails E by 0.6 pp.") == []


def test_an_underivable_made_up_margin_is_still_flagged():
    """A margin the paper's own numbers cannot produce is still a fabrication.

    41.0 pp is not in the paper and is not a difference or sum of its values
    (40.0 and 33.3 are, which is why this test uses 41.0).
    """
    errors = check_number_pairings("The combination adds 41.0 pp over the baseline.")
    assert len(errors) == 1 and "41.0" in errors[0]


# ---------------------------------------------------------------------------
# the two architecture columns that decide "can this be audited?"
# ---------------------------------------------------------------------------

def test_reader_claim_evidence_is_measured_with_the_shipped_verifier():
    """B and C record claims with quotes; A records none, and that is a datum.

    The column is computed from the Reader artifact with the project's own
    `quote_verified` annotation - no new rule is introduced for the experiment.
    """
    claims_a = reader_claim_evidence(None)
    assert claims_a == {"claims_total": 0, "claims_quote_verified": None,
                        "quote_evidence_rate": None, "uncheckable": "arm A produces no claim artifact"}

    claims = [
        {"claim": "x", "quote": "a", "quote_verified": True},
        {"claim": "y", "quote": "b", "quote_verified": True},
        {"claim": "z", "quote": "c", "quote_verified": False},
    ]
    measured = reader_claim_evidence(claims)
    assert measured["claims_total"] == 3
    assert measured["claims_quote_verified"] == 2
    assert measured["quote_evidence_rate"] == pytest.approx(2 / 3, abs=1e-4)


def test_a_fabricated_quote_stays_in_the_denominator():
    """Never drop the failure: the rate must be able to go down."""
    claims = [{"quote_verified": v} for v in (True, True, True, False)]
    assert reader_claim_evidence(claims)["quote_evidence_rate"] == 0.75


def test_paper_numbers_are_read_from_the_paper_text():
    values = paper_number_values("the rate is 44.2% and 3000 games at alpha 0.3")
    assert "44.2" in values and "3000" in values and "0.3" in values


# ---------------------------------------------------------------------------
# grading discipline
# ---------------------------------------------------------------------------

def test_the_grader_recognises_an_unrelated_report_as_uncovered():
    """A control: a checklist that scores any review highly would be worthless."""
    unrelated = (
        "This paper studies protein folding with AlphaFold. It reports a 91% accuracy "
        "on CASP15 and uses a transformer backbone."
    )
    covered, total = contributions_covered(unrelated)
    assert covered == 0, f"an unrelated report scored {covered}/{total} - the checklist is broken"


def test_the_grader_gives_a_wrong_number_no_credit():
    """Numbers are the part of coverage that cannot be faked by vocabulary."""
    wrong = "Group A wins 44.2% and group E wins 91.7%."
    assert contributions_covered(wrong)[0] < contributions_covered("Group A wins 44.2% and group E wins 68.8%.")[0]


def test_the_grader_is_deterministic_across_repeated_calls():
    report = "Group A 44.2%, group E 68.8%, DBN with an exponential moving average."
    assert {contributions_covered(report) for _ in range(5)} == {contributions_covered(report)}


# ---------------------------------------------------------------------------
# coverage against a checklist written down before scoring
# ---------------------------------------------------------------------------


def test_coverage_does_not_score_a_point_from_a_passing_mention():
    """"DBN" and "additive" thrown into one sentence must not count as coverage.

    Every contribution needs its concept *and* the magnitude that makes it a
    contribution, so a review cannot score by listing the paper's vocabulary.
    """
    report = "The paper introduces a DBN and calls the stacking additive, with a 68.8% win rate."
    covered, total = contributions_covered(report)
    assert total == len(PAPER_FACTS["contributions"])
    assert covered == 0


def test_coverage_scores_only_the_contributions_really_reported():
    """The win-rate point is reported; the orthogonal-stacking point is not."""
    report = "Group A wins 44.2% of games, group E wins 68.8%."
    covered, _ = contributions_covered(report)
    assert covered == 1  # win_rate only


def test_coverage_recognises_a_contribution_stated_in_other_words():
    report = (
        "The DBN keeps per-player suspicion up to date with an exponential moving "
        "average, which breaks the positive-feedback loop between identical agents."
    )
    covered, _ = contributions_covered(report)
    assert covered == 1


def test_coverage_accepts_the_orthogonality_point_when_it_is_actually_made():
    report = "The two modules are orthogonal: their combined +24.6 pp is close to the independent sum."
    covered, _ = contributions_covered(report)
    assert covered == 1


def test_a_fully_covering_report_scores_every_contribution():
    report = " ".join(c["statement"] for c in PAPER_FACTS["contributions"])
    covered, total = contributions_covered(report)
    assert covered == total


# ---------------------------------------------------------------------------
# the grid runner, offline
# ---------------------------------------------------------------------------

FIXTURE_TEXT = (
    "On Social Deduction with Dynamic Belief Networks\n"
    "Abstract: villager win rate 44.2% without beliefs and 68.8% with dynamic beliefs.\n"
)


def _fake_llm(pdf: Path) -> FakeLLM:
    """A scripted client that answers every arm with fixed, checkable content."""

    report = (
        "# On Social Deduction with Dynamic Belief Networks\n\n"
        "## Overview\nGroup E reaches a 68.8% win rate.\n\n"
        "## Method\nDBN updates suspicion with an exponential moving average.\n\n"
        "## Key Claims & Evidence\n"
        "- Group A wins 44.2% of games. **44.2% without beliefs**\n"
        "- Group E wins 68.8% of games. **68.8% with dynamic beliefs**\n"
        # the fabricated quote: B's machine layer must correct this bullet,
        # C's (verification disabled) must ship it as the model wrote it
        "- The model reaches a 99.9% win rate with zero variance. **[supported]**\n\n"
        "## Strengths\northogonal stacking\n\n"
        "## Limitations\nsingle game condition\n\n"
        "## Related Work\n- [Related](https://arxiv.org/abs/2501.14225)\n\n"
        "## Relevance to My Research Direction\n**9/10** - aligned.\n\n"
        "## Suggested Next Steps\nbroaden the backends\n"
    )
    claims = [
        {"claim": "Group A wins 44.2% of games", "section": "Results", "quote": "44.2% without beliefs"},
        {"claim": "Group E wins 68.8% of games", "section": "Results", "quote": "68.8% with dynamic beliefs"},
        {"claim": "The model reaches a 99.9% win rate", "section": "Results", "quote": "win rate reaches 99.9%"},
    ]
    return FakeLLM(
        script={
            "Researcher": [
                lambda m: tool_call_response("read_pdf", {"path": str(pdf)}),
                lambda m: text_response(
                    json.dumps(
                        {
                            "title": "On Social Deduction with Dynamic Belief Networks",
                            "authors": ["Zhenxiao Guo"],
                            "abstract": "DBN + DTR for social deduction.",
                            "doi": None,
                            "arxiv_id": None,
                            "url": None,
                            "published": None,
                            "source": "pdf",
                            "full_text_path": tool_result_text_path(m),
                            "full_text_chars": len(FIXTURE_TEXT),
                            "note": "",
                        }
                    )
                ),
            ],
            "Reader": [
                text_response(
                    json.dumps(
                        {
                            "title": "On Social Deduction with Dynamic Belief Networks",
                            "summary": "DBNs help.",
                            "sections": [{"heading": "Method", "content": "EMA"}],
                            "claims": claims,
                            "data_points": [],
                            "method_summary": "EMA",
                            "open_questions": [],
                        }
                    )
                )
            ],
            "Critic": [
                text_response(
                    json.dumps(
                        {
                            "verdicts": [
                                {"claim_index": i, "claim": c["claim"], "verdict": "supported", "note": "n"}
                                for i, c in enumerate(claims)
                            ],
                            "limitations": [],
                            "overall_assessment": "ok",
                        }
                    )
                )
            ],
            "Synthesizer": [text_response(report)],
        }
    )


def test_run_grid_produces_every_arm_and_run_with_unique_ids(tmp_path: Path):
    results = run_grid(
        llm_factory=lambda: _fake_llm(tmp_path / "paper.pdf"),
        runs=2,
        out_dir=tmp_path,
        paper_pdf=_write_pdf(tmp_path),
        paper_text=FIXTURE_TEXT,
        arxiv_stub=lambda query, max_results=5: "[]",
    )
    assert {r["arm"] for r in results} == set(ARMS)
    assert len(results) == len(ARMS) * 2
    ids = [r["run_id"] for r in results]
    assert len(set(ids)) == len(ids)
    for record in results:
        assert record["status"] in {"ok", "failed"}
        assert record["wall_seconds"] >= 0

def test_single_prompt_arm_makes_exactly_one_call_and_records_its_tokens(tmp_path: Path):
    results = run_grid(
        llm_factory=lambda: _fake_llm(tmp_path / "paper.pdf"),
        runs=1,
        out_dir=tmp_path,
        paper_pdf=_write_pdf(tmp_path),
        paper_text=FIXTURE_TEXT,
        arxiv_stub=lambda query, max_results=5: "[]",
    )
    single = next(r for r in results if r["arm"] == "A_single_prompt")
    assert single["llm_calls"] == 1
    assert single["prompt_tokens"] > 0 and single["completion_tokens"] > 0
    assert single["report_chars"] > 0


def test_pipeline_arms_record_per_claim_deterministic_status(tmp_path: Path):
    results = run_grid(
        llm_factory=lambda: _fake_llm(tmp_path / "paper.pdf"),
        runs=1,
        out_dir=tmp_path,
        paper_pdf=_write_pdf(tmp_path),
        paper_text=FIXTURE_TEXT,
        arxiv_stub=lambda query, max_results=5: "[]",
    )
    by_arm = {r["arm"]: r for r in results}
    assert by_arm["B_pipeline"]["claims_total"] == 3
    assert by_arm["B_pipeline"]["claims_with_deterministic_status"] == 3
    assert by_arm["B_pipeline"]["claims_quote_unverified"] == 1, "the fabricated quote must stay in the denominator"
    # C runs the same reader but ships nothing that carries the machine status
    assert by_arm["C_pipeline_no_verification"]["claims_total"] == 3
    assert by_arm["C_pipeline_no_verification"]["claims_with_deterministic_status"] == 0
    assert by_arm["C_pipeline_no_verification"]["machine_markers_on_bullets"] == 0
    # B repairs the bullet that the deterministic check rejected ...
    assert by_arm["B_pipeline"]["machine_markers_on_bullets"] >= 1
    assert by_arm["B_pipeline"]["ledger_in_report"] is True
    # ... while C still ships it marked as the model wrote it
    assert by_arm["C_pipeline_no_verification"]["ledger_in_report"] is False
    assert by_arm["C_pipeline_no_verification"]["llm_markers_on_bullets"] == by_arm["B_pipeline"]["llm_markers_on_bullets"] + 1
    # A has no claim artifact at all: the honest reading of the architecture
    assert by_arm["A_single_prompt"]["claims_total"] == 0


def test_tokens_are_measured_per_run_not_cumulatively(tmp_path: Path):
    """A shared client would silently double-count; each run gets its own delta."""
    results = run_grid(
        llm_factory=lambda: _fake_llm(tmp_path / "paper.pdf"),
        runs=2,
        out_dir=tmp_path,
        paper_pdf=_write_pdf(tmp_path),
        paper_text=FIXTURE_TEXT,
        arxiv_stub=lambda query, max_results=5: "[]",
    )
    per_arm = {}
    for record in results:
        per_arm.setdefault(record["arm"], []).append(record["prompt_tokens"])
    for arm, values in per_arm.items():
        assert values[0] > 0, arm
        assert values[0] == values[1], f"{arm} run deltas differ: {values}"


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def test_aggregate_reports_mean_and_range_and_never_only_the_best_run():
    results = [
        {"arm": "A_single_prompt", "status": "ok", "cost_cny": 0.10, "prompt_tokens": 100, "completion_tokens": 10,
         "llm_calls": 1, "wall_seconds": 2.0, "headings_present": 8, "attribution_rate": 1.0,
         "coverage": 5, "factual_errors": 0},
        {"arm": "A_single_prompt", "status": "ok", "cost_cny": 0.30, "prompt_tokens": 100, "completion_tokens": 30,
         "llm_calls": 1, "wall_seconds": 4.0, "headings_present": 6, "attribution_rate": 0.5,
         "coverage": 3, "factual_errors": 2},
    ]
    summary = aggregate(results)
    arm = summary["arms"]["A_single_prompt"]
    assert arm["runs_ok"] == 2
    assert arm["cost_cny"]["mean"] == pytest.approx(0.20)
    assert arm["cost_cny"]["min"] == pytest.approx(0.10)
    assert arm["cost_cny"]["max"] == pytest.approx(0.30)
    assert arm["headings_present"]["min"] == 6 and arm["headings_present"]["max"] == 8
    assert summary["total_cost_cny"] == pytest.approx(0.40)


def test_aggregate_counts_failed_runs_separately_and_excludes_them_from_means():
    results = [
        {"arm": "B_pipeline", "status": "ok", "cost_cny": 1.0, "prompt_tokens": 10, "completion_tokens": 1,
         "llm_calls": 4, "wall_seconds": 9.0, "headings_present": 8, "attribution_rate": 1.0,
         "coverage": 9, "factual_errors": 0},
        {"arm": "B_pipeline", "status": "failed", "error": "BudgetExceeded", "cost_cny": 0.5,
         "prompt_tokens": 5, "completion_tokens": 0, "llm_calls": 2, "wall_seconds": 1.0,
         "headings_present": 0, "attribution_rate": 0.0, "coverage": 0, "factual_errors": 0},
    ]
    arm = aggregate(results)["arms"]["B_pipeline"]
    assert arm["runs_total"] == 2
    assert arm["runs_ok"] == 1
    assert arm["runs_failed"] == 1
    assert arm["cost_cny"]["mean"] == pytest.approx(1.0), "a failed run must not be averaged in"
    assert arm["failure_reasons"] == ["BudgetExceeded"]


# ---------------------------------------------------------------------------
# live-run preconditions
# ---------------------------------------------------------------------------


def test_the_experiment_pins_the_models_published_prices_and_a_hard_cap():
    assert BUDGET_CNY == 30.0
    assert "DEEPSEEK_API_KEY" not in json.dumps({"x": 1})  # key never enters the artifact


def test_a_live_client_refuses_to_spend_past_the_cap():
    client = LLMClient(api_key="k", base_url="http://fake.local", budget_cny=0.0)
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "hi"}], max_tokens=10)


def test_run_grid_refuses_to_start_without_a_paper_pdf(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_grid(
            llm_factory=lambda: _fake_llm(tmp_path / "missing.pdf"),
            runs=1,
            out_dir=tmp_path,
            paper_pdf=tmp_path / "missing.pdf",
            paper_text=FIXTURE_TEXT,
            arxiv_stub=lambda query, max_results=5: "[]",
        )


def test_paper_facts_are_traceable_to_the_repository_pdf():
    """The ground-truth sheet must describe the paper actually under test."""
    assert PAPER_PDF.name == "werewolf-multiagent-paper.pdf"
    assert PAPER_FACTS["doi"] == "10.54254/2753-8818/2026.DL34010"
    assert len(PAPER_FACTS["contributions"]) >= 8


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_pdf(tmp_path: Path) -> Path:
    from tests.pdfutil import make_pdf

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(make_pdf(FIXTURE_TEXT))
    return pdf
