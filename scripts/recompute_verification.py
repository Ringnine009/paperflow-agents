"""Recompute the deterministic quote-verification hit rate over archived runs.

The verifier's false-negative rate was found by replaying every archived run
(``outputs/*/`` plus the committed ``examples/*/``) through the *old* and the
*new* implementation of :func:`paperflow.tools.texttools.verify_quote`.

* the old implementation is frozen below as :func:`verify_quote_legacy`
  (a verbatim copy of the pre-fix code) so the "before" number is measured,
  not quoted from memory;
* the "after" number comes from the shipped implementation.

The script also cross-checks the frozen copy against the ``quote_verified``
values the old code stored in each ``reader_output.json`` - if they disagree,
the frozen copy is not faithful and the before/after comparison is invalid.

Offline: it only reads local files.

Usage::

    python scripts/recompute_verification.py                # markdown table
    python scripts/recompute_verification.py --json out.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from paperflow.agents.verification import check_report_consistency, claim_statuses
from paperflow.tools.texttools import verify_quote  # noqa: E402

_WHITESPACE_LEGACY = re.compile(r"\s+")


def normalize_ws_legacy(text: str) -> str:
    """Frozen pre-fix normalizer (collapse whitespace + lowercase)."""
    return _WHITESPACE_LEGACY.sub(" ", text).strip().lower()


def verify_quote_legacy(full_text: str, quote: str, min_chars: int = 4) -> dict:
    """Frozen pre-fix verifier: substring search on whitespace-normalized text.

    Copied verbatim from ``paperflow/tools/texttools.py`` at commit 6e2fff0 so
    the "before" column is reproducible.
    """
    if not quote or not quote.strip():
        return {"found": False, "reason": "empty quote"}
    needle = normalize_ws_legacy(quote)
    if len(needle) < min_chars:
        return {"found": False, "reason": "quote too short to verify"}
    haystack = normalize_ws_legacy(full_text)
    if not haystack:
        return {"found": False, "reason": "no full text available"}
    loc = haystack.find(needle)
    if loc == -1:
        return {"found": False, "reason": "quote not found in the paper text"}
    return {"found": True, "reason": "quote found verbatim (whitespace-normalized)"}


def find_runs(root: Path) -> list[Path]:
    """Run directories that hold both a full text and reader claims."""
    runs = []
    for pattern in ("outputs/pf-*", "examples/*"):
        for path in sorted(glob.glob(str(root / pattern))):
            run = Path(path)
            if (run / "artifacts" / "full_text.txt").is_file() and (run / "artifacts" / "reader_output.json").is_file():
                runs.append(run)
    return runs


class _BoardView:
    """Minimal board stand-in for auditing archived runs.

    Archived ``board.json`` files may carry artifact paths from another
    machine, so the audit reads the artifacts sitting next to the report
    instead of trusting the recorded paths.
    """

    def __init__(self, run: Path):
        self.artifacts = {
            name: {"path": str(run / "artifacts" / f"{name}.json")}
            for name in ("reader_output", "critic_output")
            if (run / "artifacts" / f"{name}.json").is_file()
        }


def analyze(run: Path) -> dict:
    text = (run / "artifacts" / "full_text.txt").read_text(encoding="utf-8", errors="replace")
    data = json.loads((run / "artifacts" / "reader_output.json").read_text(encoding="utf-8"))
    claims = data.get("claims", [])
    row = {
        "run": run.name,
        "claims": len(claims),
        "before": 0,
        "after": 0,
        "modes": {},
        "fixed": [],
        "still_missing": [],
        "stored_agree": 0,
        "stored_total": 0,
    }
    for index, claim in enumerate(claims):
        quote = claim.get("quote", "")
        legacy = verify_quote_legacy(text, quote)
        current = verify_quote(text, quote)
        row["before"] += int(legacy["found"])
        row["after"] += int(current["found"])
        if current["found"]:
            row["modes"][current["match_mode"]] = row["modes"].get(current["match_mode"], 0) + 1
        if not legacy["found"] and current["found"]:
            row["fixed"].append({"index": index, "mode": current["match_mode"], "quote": quote})
        if not current["found"] and legacy["found"]:
            row["still_missing"].append({"index": index, "quote": quote, "regression": "was found before"})
        if not current["found"]:
            row["still_missing"].append({"index": index, "quote": quote, "regression": None})
        if "quote_verified" in claim:
            row["stored_total"] += 1
            row["stored_agree"] += int(bool(claim["quote_verified"]) == bool(legacy["found"]))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="also write the full results as JSON")
    args = parser.parse_args()

    runs = find_runs(REPO)
    rows = [analyze(run) for run in runs]
    claims = sum(r["claims"] for r in rows)
    before = sum(r["before"] for r in rows)
    after = sum(r["after"] for r in rows)
    stored_total = sum(r["stored_total"] for r in rows)
    stored_agree = sum(r["stored_agree"] for r in rows)
    modes: dict[str, int] = {}
    for row in rows:
        for mode, count in row["modes"].items():
            modes[mode] = modes.get(mode, 0) + count

    print("| run | claims | before (legacy) | after (fixed) | newly verified |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['run']} | {row['claims']} | {row['before']}/{row['claims']} | "
            f"{row['after']}/{row['claims']} | {len([f for f in row['fixed'] if f.get('mode')])} |"
        )
    print(
        f"| **TOTAL** | **{claims}** | **{before}/{claims} ({100.0 * before / claims:.1f}%)** | "
        f"**{after}/{claims} ({100.0 * after / claims:.1f}%)** | **{after - before}** |"
    )
    print()
    print("match modes after the fix:", json.dumps(modes, sort_keys=True))
    print(f"frozen-legacy copy reproduces the stored quote_verified flags: {stored_agree}/{stored_total}")
    remaining = [f for row in rows for f in row["still_missing"] if not f["regression"]]
    print(f"quotes still not found: {len(remaining)}")
    for item in remaining:
        print("  -", item["quote"][:110])

    # -- the report/verification consistency check, applied retroactively ----
    # Archived reports were written before the machine ledger existed, so the
    # check is expected to flag exactly the runs the audit complained about.
    audit = []
    print()
    print("=== report/verification consistency of the archived reports (new check) ===")
    for run in runs:
        report_file = run / "report.md"
        if not report_file.is_file():
            continue
        board = _BoardView(run)
        violations = check_report_consistency(board, report_file.read_text(encoding="utf-8"))
        statuses = [row["status"] for row in claim_statuses(board)]
        problems = sum(1 for s in statuses if s != "verified")
        audit.append({"run": run.name, "claims": len(statuses), "non_verified": problems, "violations": violations})
        if problems or violations:
            print(f"{run.name:<28} claims={len(statuses)} non-verified={problems} violations={len(violations)}")
            for violation in violations[:3]:
                print(f"    - {violation[:130]}")
    flagged = [row for row in audit if row["violations"]]
    print(f"reports with at least one violation: {len(flagged)}/{len(audit)}")

    if args.json:
        payload = {
            "runs": rows,
            "totals": {
                "runs": len(rows),
                "claims": claims,
                "before": before,
                "after": after,
                "before_rate": round(100.0 * before / claims, 2) if claims else 0.0,
                "after_rate": round(100.0 * after / claims, 2) if after or claims else 0.0,
                "modes": modes,
                "legacy_reproduces_stored": f"{stored_agree}/{stored_total}",
            },
            "report_consistency": audit,
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
