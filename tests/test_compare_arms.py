"""The three-arm comparison harness must produce three real arms (offline).

Arms: A = one long prompt that writes the review directly, B = the current
multi-stage pipeline, C = the pipeline with the deterministic quote check
switched off (the pre-fix architecture).

Only *architecture-determined* columns are evidence here - how many LLM
calls an arm makes, whether a deterministic check exists for its claims,
whether the report carries the machine verdict. The text quality of an arm
comes from a scripted FakeLLM, so the harness labels those columns as
scripted instead of presenting them as a result.
"""

from __future__ import annotations

from pathlib import Path

from scripts.compare_arms import ARMS, run_experiment


def test_compare_arms_produces_three_arms(tmp_path: Path):
    result = run_experiment(out_dir=tmp_path)
    assert set(result["arms"]) == set(ARMS)
    for name, arm in result["arms"].items():
        assert arm["report_chars"] > 0, name
        assert arm["llm_calls"] >= 1, name
        assert arm["prompt_chars"] > 0, name


def test_arms_differ_in_what_the_architecture_guarantees(tmp_path: Path):
    arms = run_experiment(out_dir=tmp_path)["arms"]
    single, pipeline, no_verification = arms["A_single_prompt"], arms["B_pipeline"], arms["C_pipeline_no_verification"]

    # A: one call, no artifact, nothing checkable
    assert single["llm_calls"] == 1
    assert single["claims_total"] == 0
    assert single["claims_with_deterministic_check"] == 0

    # B and C: multi-stage, claims are recorded as artifacts
    assert pipeline["llm_calls"] >= 4 and no_verification["llm_calls"] >= 4
    assert pipeline["claims_total"] > 0 and no_verification["claims_total"] == pipeline["claims_total"]

    # B: every claim carries a deterministic status and the report shows it
    assert pipeline["claims_with_deterministic_check"] == pipeline["claims_total"]
    assert pipeline["unverified_markers_in_report"] >= 1
    assert pipeline["ledger_in_report"] is True

    # C: the pre-fix architecture - the same claim is still sold as supported
    assert no_verification["claims_with_deterministic_check"] == 0
    assert no_verification["unverified_markers_in_report"] == 0
    assert no_verification["ledger_in_report"] is False
    assert no_verification["supported_markers_in_report"] == pipeline["supported_markers_in_report"] + 1


def test_harness_declares_what_it_cannot_measure(tmp_path: Path):
    result = run_experiment(out_dir=tmp_path)
    # FakeLLM returns a stub `usage` block, so token counts/cost are not real
    assert result["tokens_measurable"] is False
    assert "scripted" in result["quality_note"].lower()
    assert result["arms"]["B_pipeline"]["verifiable_claim_ratio"] == 1.0
    assert result["arms"]["C_pipeline_no_verification"]["verifiable_claim_ratio"] == 0.0
