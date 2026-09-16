"""The semantic-support judge: parsing, reliability maths, and what it refuses to do.

The judge is the only LLM-judged column in the experiment, so its failure
modes matter more than its happy path: an unparseable answer must not become
agreement, an "unclear" must not count as support, and the agreement figure
between the two passes must be real arithmetic rather than an assertion.
"""

from __future__ import annotations

import pytest

from scripts.judge_semantic_support import (
    BATCH,
    VERDICTS,
    build_messages,
    cohen_kappa,
    parse_reasons,
    parse_verdicts,
    percent_agreement,
    summarise,
)


def test_parse_reads_the_documented_json_shape():
    content = '{"verdicts": [{"index": 0, "verdict": "yes", "reason": "matches"}, {"index": 1, "verdict": "no", "reason": "wrong cell"}]}'
    assert parse_verdicts(content, 2) == ["yes", "no"]


def test_parse_survives_code_fences_and_surrounding_prose():
    content = 'Here you go:\n```json\n{"verdicts": [{"index": 0, "verdict": "unclear"}]}\n```\n'
    assert parse_verdicts(content, 1) == ["unclear"]


def test_unparseable_answer_is_unclear_and_never_silently_agreement():
    assert parse_verdicts("I cannot help with that.", 3) == ["unclear", "unclear", "unclear"]
    assert parse_verdicts("", 2) == ["unclear", "unclear"]


def test_missing_or_malformed_entries_become_unclear():
    content = '{"verdicts": [{"index": 1, "verdict": "MAYBE"}, {"index": 99, "verdict": "yes"}]}'
    assert parse_verdicts(content, 2) == ["unclear", "unclear"]


def test_the_judges_reason_is_kept_for_the_audit_trail():
    """A verdict without its reason is not reviewable by a human later.

    The first live run stored `reason: None` for every item - the parser read
    the verdict and threw the justification away - which made the "no"
    verdicts impossible to characterise after the fact.
    """
    content = (
        '{"verdicts": [{"index": 0, "verdict": "no", "reason": "quote is narrower than claim"},'
        ' {"index": 1, "verdict": "yes", "reason": "numbers match"}]}'
    )
    assert parse_verdicts(content, 2) == ["no", "yes"]
    assert parse_reasons(content, 2) == ["quote is narrower than claim", "numbers match"]


def test_a_missing_reason_is_empty_text_and_never_invented():
    content = '{"verdicts": [{"index": 0, "verdict": "yes"}]}'
    assert parse_reasons(content, 1) == [""]


def test_reasons_tolerate_fences_and_odd_shapes():
    content = '```json\n{"verdicts": [{"index": 0, "verdict": "no", "reason": 42}]}\n```'
    assert parse_reasons(content, 1) == ["42"]


def test_yes_no_unclear_is_the_whole_vocabulary():
    assert VERDICTS == ("yes", "no", "unclear")
    assert BATCH > 1 and BATCH <= 20, "batching is a recorded cost knob, not a free lunch"


# ---------------------------------------------------------------------------
# reliability arithmetic
# ---------------------------------------------------------------------------

def test_percent_agreement_counts_matching_passes():
    assert percent_agreement(["yes", "no", "yes"], ["yes", "yes", "yes"]) == pytest.approx(2 / 3, abs=1e-4)


def test_cohen_kappa_is_undefined_when_both_passes_are_constant():
    """All-yes twice gives 100% agreement but no information: kappa is None."""
    assert cohen_kappa(["yes"] * 5, ["yes"] * 5) is None


def test_cohen_kappa_is_zero_for_chance_level_agreement():
    first = ["yes", "no"] * 25
    second = ["yes"] * 50
    # agreement equals the expected rate -> kappa 0
    assert cohen_kappa(first, second) == pytest.approx(0.0, abs=0.05)


def test_cohen_kappa_is_one_for_perfect_disagreeing_distributions():
    first = ["yes", "no"] * 10
    assert cohen_kappa(first, first) == pytest.approx(1.0, abs=1e-6)


def test_agreement_helpers_return_none_on_empty_or_mismatched_inputs():
    assert percent_agreement([], []) is None
    assert cohen_kappa(["yes"], ["yes", "no"]) is None


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def test_support_rate_uses_every_judged_item_as_the_denominator():
    """"unclear" is not a pass: the conservative denominator is the honest one."""
    items = [
        {"verdict": "yes"}, {"verdict": "yes"}, {"verdict": "no"}, {"verdict": "unclear"},
    ]
    summary = summarise(items)
    assert summary["n"] == 4
    assert summary["yes"] == 2 and summary["no"] == 1 and summary["unclear"] == 1
    assert summary["support_rate"] == 0.5
    assert summary["support_rate_of_decidable"] == pytest.approx(2 / 3, abs=1e-4)


def test_empty_sample_reports_no_rate_rather_than_zero():
    assert summarise([])["support_rate"] is None


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_the_quoted_prompt_carries_the_pair_and_the_paper_prompt_carries_the_text():
    items = [{"text": "Group E wins 68.8%", "quote": "68.8%"}]
    quoted = build_messages(items, "PAPER BODY", quoted=True)[1]["content"]
    attributed = build_messages(items, "PAPER BODY", quoted=False)[1]["content"]
    assert "QUOTE: 68.8%" in quoted and "PAPER BODY" not in quoted
    assert "PAPER: PAPER BODY" in attributed


def test_a_missing_quote_is_shown_as_none_so_the_judge_can_say_unclear():
    content = build_messages([{"text": "c", "quote": ""}], "body", quoted=True)[1]["content"]
    assert "QUOTE: (none)" in content
