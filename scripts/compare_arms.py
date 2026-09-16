"""Offline comparison harness for three review architectures.

Arms
----
``A_single_prompt``
    one long prompt containing the paper text, answered in a single LLM call
    (no task board, no artifacts, no tool loop).
``B_pipeline``
    the shipped four-agent pipeline (Researcher -> Reader -> Critic ->
    Synthesizer) with the deterministic quote check and the machine-generated
    verification ledger enabled.
``C_pipeline_no_verification``
    the same pipeline, orchestrator, board and tools, with the deterministic
    quote check switched off and the report not machine-corrected - an
    emulation of the pre-fix architecture, in which nothing in the code
    decides whether a quote exists.

Evidence columns are the ones decided by the *architecture* rather than by
the model: ``llm_calls``, ``prompt_chars`` (a cost proxy),
``claims_with_deterministic_check`` / ``verifiable_claim_ratio``,
``ledger_in_report`` and the verdict markers found in the report.

Not evidence: ``required_headings_present`` is produced by a scripted
``FakeLLM``, so it says nothing about how a real model would write under each
architecture; token counts are unmeasurable offline because the fake client
returns a stub ``usage`` block. The experiment that would settle the quality
question (and what it needs) is described in ``docs/baseline-plan.md``.

Usage::

    python scripts/compare_arms.py --json docs/arm-comparison.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from paperflow.agents import synthesizer as synthesizer_module  # noqa: E402
from paperflow.agents.reader import ReaderAgent  # noqa: E402
from paperflow.agents.synthesizer import REPORT_HEADINGS  # noqa: E402
from paperflow.config import Settings  # noqa: E402
from paperflow.pipeline import Pipeline  # noqa: E402
from tests.helpers import (  # noqa: E402
    FIXTURE_PAPER_TEXT,
    FakeLLM,
    text_response,
    tool_call_response,
    tool_result_text_path,
)
from tests.pdfutil import make_pdf  # noqa: E402

ARMS = ("A_single_prompt", "B_pipeline", "C_pipeline_no_verification")

SETTINGS = Settings(deepseek_api_key="sk-test")

#: a quote that is really in the fixture paper ...
REAL_QUOTE = "44.2% without beliefs and 68.8% with dynamic beliefs"
#: ... and a fabricated one, so every arm faces a claim that cannot be checked
FAKE_QUOTE = "the win rate reaches 99.9% with zero variance"

READER_CLAIMS = [
    {
        "claim": "The DBN raises the villager win rate from 44.2% to 68.8%",
        "section": "Experiments",
        "quote": REAL_QUOTE,
        "evidence": "abstract",
    },
    {
        "claim": "The model reaches a 99.9% win rate with zero variance",
        "section": "Experiments",
        "quote": FAKE_QUOTE,
        "evidence": "made up",
    },
]

#: canned multi-stage report: every claim marked supported, exactly as the
#: audited runs did, so the machine layer has something to correct
PIPELINE_REPORT = (
    "# On Social Deduction with Dynamic Belief Networks\n\n"
    "## Overview\nA study of DBNs in Werewolf.\n\n"
    "## Method\nbelief updates after each public statement.\n\n"
    "## Key Claims & Evidence\n"
    "- **The DBN raises the villager win rate.** **[supported]** - abstract.\n"
    "- **The model reaches a 99.9% win rate.** **[supported]** - Table 2.\n\n"
    "## Strengths\nnovel model\n\n"
    "## Limitations\nnarrow evaluation\n\n"
    "## Related Work\n- [Related](https://arxiv.org/abs/2601.99999)\n\n"
    "## Relevance to My Research Direction\n**8/10** - aligned.\n\n"
    "## Suggested Next Steps\nreproduce on larger games\n"
)

#: Canned single-prompt answer. Deliberately given the same eight headings as
#: the pipeline arm: the text comes from a script, so a "better" or "worse"
#: structure here would only measure the script. SCRIPTED - not evidence.
SINGLE_PROMPT_REPORT = (
    "# On Social Deduction with Dynamic Belief Networks\n\n"
    "## Overview\nA study of social deduction with dynamic belief networks.\n\n"
    "## Method\nPosterior belief updates after each public statement.\n\n"
    "## Key Claims & Evidence\n"
    "- The villager win rate rises from 44.2% to 68.8%. [supported]\n"
    "- The model reaches a 99.9% win rate with zero variance. [supported]\n\n"
    "## Strengths\nnovel model\n\n"
    "## Limitations\nnarrow evaluation\n\n"
    "## Related Work\n- [Related](https://arxiv.org/abs/2601.99999)\n\n"
    "## Relevance to My Research Direction\n**8/10** - aligned.\n\n"
    "## Suggested Next Steps\nreproduce on larger games\n"
)


# ---------------------------------------------------------------------------
# arm scripts and stubs
# ---------------------------------------------------------------------------

def _researcher_script(pdf: Path) -> dict:
    return {
        "Researcher": [
            lambda m: tool_call_response("read_pdf", {"path": str(pdf)}),
            lambda m: text_response(
                json.dumps(
                    {
                        "title": "On Social Deduction with Dynamic Belief Networks",
                        "authors": ["Tongji Student"],
                        "abstract": "We study multi-agent social deduction in Werewolf with DBNs.",
                        "doi": None,
                        "arxiv_id": None,
                        "url": None,
                        "published": None,
                        "source": "pdf",
                        "full_text_path": tool_result_text_path(m),
                        "full_text_chars": len(FIXTURE_PAPER_TEXT),
                        "note": "",
                    }
                )
            ),
        ]
    }


def _reader_script() -> dict:
    return {
        "Reader": [
            text_response(
                json.dumps(
                    {
                        "title": "On Social Deduction with Dynamic Belief Networks",
                        "summary": "DBNs improve social reasoning in hidden-role games.",
                        "sections": [{"heading": "Method", "content": "DBN posterior updates"}],
                        "claims": READER_CLAIMS,
                        "data_points": [],
                        "method_summary": "posterior updates after statements",
                        "open_questions": [],
                    }
                )
            )
        ]
    }


def _critic_script() -> dict:
    """The LLM calls both claims supported - the machine layer must disagree."""
    return {
        "Critic": [
            text_response(
                json.dumps(
                    {
                        "verdicts": [
                            {"claim_index": i, "claim": c["claim"], "verdict": "supported", "note": "llm note"}
                            for i, c in enumerate(READER_CLAIMS)
                        ],
                        "limitations": [],
                        "overall_assessment": "ok",
                    }
                )
            )
        ]
    }


def _synthesizer_script() -> dict:
    return {
        "Synthesizer": [
            lambda m: tool_call_response("arxiv_search", {"query": "ti:multi-agent werewolf", "max_results": 3}),
            lambda m: text_response(PIPELINE_REPORT),
        ]
    }


def _stub_arxiv(pipe: Pipeline) -> None:
    pipe.registry.get("arxiv_search").func = lambda query, max_results=5: json.dumps(
        [
            {
                "arxiv_id": "2601.99999",
                "title": "Related Multi-Agent Work",
                "authors": ["Someone"],
                "published": "2025-01-01",
                "abstract": "related",
                "pdf_url": "https://arxiv.org/pdf/2601.99999",
                "abs_url": "https://arxiv.org/abs/2601.99999",
                "doi": None,
            }
        ]
    )


@contextlib.contextmanager
def verification_disabled():
    """Switch the deterministic quote layer off, exactly for one block.

    Emulates the pre-fix architecture: the Reader records no deterministic
    annotation, and the report is neither machine-corrected nor checked.
    Everything is restored in ``finally`` - the shipped behaviour is never
    left patched behind an experiment.
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


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _cost_metrics(llm: FakeLLM) -> dict:
    """Calls issued and characters sent (a cost proxy, measured not guessed)."""
    chars = 0
    for payload in llm.payloads:
        for message in payload.get("messages", []):
            content = message.get("content") or ""
            chars += len(content if isinstance(content, str) else json.dumps(content))
        if payload.get("tools"):
            chars += len(json.dumps(payload["tools"]))
    return {"llm_calls": len(llm.payloads), "prompt_chars": chars}


def _report_metrics(report: str) -> dict:
    return {
        "report_chars": len(report),
        "required_headings_present": sum(1 for h in REPORT_HEADINGS if f"## {h}" in report),
        "supported_markers_in_report": report.count("[supported]"),
        "unverified_markers_in_report": report.count("[unverified]"),
        "unverifiable_markers_in_report": report.count("[unverifiable]"),
        "ledger_in_report": "## Verification Ledger" in report,
    }


def _claim_metrics(reader_artifact: dict | None) -> dict:
    claims = (reader_artifact or {}).get("claims", []) or []
    checked = [c for c in claims if c.get("quote_status") or "quote_verified" in c]
    return {
        "claims_total": len(claims),
        "claims_with_deterministic_check": len(checked),
        "verifiable_claim_ratio": round(len(checked) / len(claims), 3) if claims else 0.0,
        "claims_verified_by_quote": sum(1 for c in claims if c.get("quote_verified")),
    }


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------

def arm_a_single_prompt(out_dir: Path) -> dict:
    """One long prompt, one call, no artifacts and nothing checkable."""
    system = (
        "You are a single-pass paper reviewer (Synthesizer role). Read the paper text the "
        "user provides and write the complete structured review in Markdown."
    )
    user = "PAPER TEXT:\n" + FIXTURE_PAPER_TEXT
    llm = FakeLLM({"Synthesizer": [text_response(SINGLE_PROMPT_REPORT)]})
    message = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.2
    )
    report = message.get("content") or ""
    (out_dir / "arm_a_report.md").write_text(report, encoding="utf-8")
    return {"arm": "A_single_prompt", **_cost_metrics(llm), **_report_metrics(report), **_claim_metrics(None)}


def arm_pipeline(out_dir: Path, *, verification: bool) -> dict:
    """The shipped pipeline, with the deterministic layer on or off."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "paper.pdf"
    pdf.write_bytes(make_pdf(FIXTURE_PAPER_TEXT))
    llm = FakeLLM(
        script={
            **_researcher_script(pdf),
            **_reader_script(),
            **_critic_script(),
            **_synthesizer_script(),
        }
    )
    pipe = Pipeline(settings=SETTINGS, out_dir=out_dir / "out", llm=llm)
    _stub_arxiv(pipe)

    context = contextlib.nullcontext() if verification else verification_disabled()
    with context:
        result = pipe.run(str(pdf))

    report = Path(result["report"]).read_text(encoding="utf-8") if result["report"] else ""
    artifact_path = Path(result["board"]).parent / "artifacts" / "reader_output.json"
    reader_artifact = json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path.exists() else None
    return {
        "arm": "B_pipeline" if verification else "C_pipeline_no_verification",
        "status": result["status"],
        **_cost_metrics(llm),
        **_report_metrics(report),
        **_claim_metrics(reader_artifact),
    }


def run_experiment(out_dir: Path | None = None) -> dict:
    """Run all three arms and return their metrics (offline, deterministic)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(out_dir) if out_dir else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        collected = [
            arm_a_single_prompt(root),
            arm_pipeline(root / "b", verification=True),
            arm_pipeline(root / "c", verification=False),
        ]

    arms = {arm["arm"]: arm for arm in collected}
    for arm in arms.values():
        arm["prompt_chars_per_call"] = round(arm["prompt_chars"] / max(1, arm["llm_calls"]))
        arm["estimated_prompt_tokens"] = round(arm["prompt_chars"] / 4)  # ~4 chars/token, English
    return {
        "arms": arms,
        "tokens_measurable": False,
        "tokens_note": (
            "FakeLLM returns a stub `usage` block (1/1/2), so real token counts and cost are "
            "not measurable offline; `estimated_prompt_tokens` is a characters/4 estimate."
        ),
        "quality_note": (
            "SCRIPTED model behaviour: every arm answers from a canned FakeLLM script, so the "
            "quality column (required_headings_present) is not evidence about model quality. "
            "Only architecture-determined columns are evidence. See docs/baseline-plan.md."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write the metrics as JSON")
    args = parser.parse_args()

    result = run_experiment()
    header = (
        f"{'arm':<28}{'llm_calls':>10}{'prompt_chars':>14}{'claims':>8}"
        f"{'checked':>9}{'headings':>10}{'[supported]':>13}{'[unverified]':>14}"
    )
    print(header)
    print("-" * len(header))
    for name in ARMS:
        arm = result["arms"][name]
        print(
            f"{name:<28}{arm['llm_calls']:>10}{arm['prompt_chars']:>14}{arm['claims_total']:>8}"
            f"{arm['claims_with_deterministic_check']:>9}{arm['required_headings_present']:>10}"
            f"{arm['supported_markers_in_report']:>13}{arm['unverified_markers_in_report']:>14}"
        )
    print()
    print("tokens measurable:", result["tokens_measurable"])
    print(result["tokens_note"])
    print(result["quality_note"])
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
