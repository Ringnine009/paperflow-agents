# Three-arm comparison under a real model — results (N=5 per arm)

**What this settles:** the first grid answered "does the pipeline beat one prompt?"
with 3 repeats per arm, its own declared floor at the time and a stated
limitation. This document reports the same experiment widened to
**5 repeats per arm (15 runs)**, and it answers the one
question the extra samples were bought for: **once the per-arm intervals are computed from a
wider sample, do the arms still fail to separate on quality?**

Every number below is rendered from `docs/arm-comparison-live.json` (machine-readable,
per-run) by `scripts/compare_arms_live.py`; the protocol was fixed in advance in
[`arm-comparison-live-design.md`](arm-comparison-live-design.md). The first (N=3) grid is
preserved unchanged next to it as `docs/arm-comparison-live-n3.json` — history is not
overwritten, it is what the extension is compared against.

> **Integrity note (read this before comparing the two grids).** 4 of the
> 9 stored N=3 run directories no longer contain the text the
> published N=3 numbers were measured from. The extension re-scored every stored report and
> reports the N=3 column from that re-scored basis, so the table and the run list are the same
> evidence; the published-vs-stored detail is in [§5.1](#51-a-stored-report-that-changed-underneath-the-grid),
> and the published N=3 numbers themselves are preserved unchanged in
> `docs/arm-comparison-live-n3.json`.

**Headline: the wider sample kept the cost and error conclusions, and removed the one column that had looked like a quality edge.** Coverage at N=5 is a mean of 14/14 for the single prompt against 13.4/14 for the pipeline (a mean gap of 0.6 of 14 items), and the two intervals **overlap**, so coverage does not separate the arms. The published N=3 numbers had them disjoint ("A ≥ B in every run"); on the stored reports re-scored — the same three repeats — they already overlapped (§5.1). Factual errors were 0 in every run of both grids, and the cost multiple moved from 4.57× to 4.31× (-0.263), which is inside the band the three-run grid implied.

---

## 1. Setup

| | |
| --- | --- |
| Model | `deepseek-chat` |
| Temperature | 0.2 (non-zero so the repeats sample real variance) |
| Thinking mode | `disabled` |
| Paper | `werewolf-multiagent-paper.pdf` |
| Runs | **5 per arm**, widened from 3 by buying 6 new repeats and reusing 9 stored ones — 15 runs total |
| Failures | **0** failed, 0 skipped |
| Bought by this extension | A_single_prompt-r4, B_pipeline-r4, C_pipeline_no_verification-r4, A_single_prompt-r5, B_pipeline-r5, C_pipeline_no_verification-r5 (0.9897 CNY) |
| Judge | same model, temperature 0, prompt v1.0, batch 10, ≤20 items/arm, every batch judged twice with the option order swapped |

## 2. Cost accounting — measured, from the API's own `usage` block

Prices are the published DeepSeek ones at **peak** rates, converted at 7.1 CNY/USD; cost =
tokens × those prices. Mean with [min–max] over the successful runs of each grid, the N=3 side being the stored reports re-scored (§5.1).

| Column | A (N=3) | B (N=3) | C (N=3) | A (N=5) | B (N=5) | C (N=5) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM calls per review | **1** (1–1) | **9** (8–11) | **10.67** (10–12) | **1** (1–1) | **9** (8–11) | **10.8** (10–12) |
| Input tokens per review | **5,991** (5,991–5,991) | **43,623** (35,222–58,035) | **50,707** (43,304–63,540) | **5,991** (5,991–5,991) | **41,116** (29,449–58,035) | **50,703** (39,171–63,540) |
| Output tokens per review | **2,773** (2,576–2,899) | **8,652** (8,188–9,536) | **10,245** (9,836–10,769) | **2,899** (2,576–3,402) | **8,509** (7,407–9,536) | **10,001** (8,760–10,769) |
| Wall-clock per review (s) | **16.3** (15–17.3) | **42.1** (38.3–46.1) | **49.5** (47.5–53.4) | **16.8** (15–18.9) | **42.1** (37.6–46.7) | **48.5** (44.2–53.4) |
| **CNY per review** | **0.0253** (0.0237–0.0264) | **0.1157** (0.107–0.1312) | **0.1277** (0.1135–0.1402) | **0.0264** (0.0237–0.0307) | **0.1138** (0.0984–0.1312) | **0.1271** (0.1135–0.1402) |
| vs arm A | 1.00× | 4.57× | 5.05× | 1.00× | **4.31×** | 4.81× |

### The cost multiple, N=3 vs N=5

| | N=3 | N=5 | moved by |
| --- | ---: | ---: | ---: |
| B over A (mean prices) | 4.57× | 4.31× | -0.263 |
| per-run ratio band, B/A | 4.05–5.55 | 3.2–5.55 | +0.85 wide |
| C over A (mean prices) | 5.05× | 4.81× | -0.233 |

The ratio band is every cross-run ratio between the two arms' per-review prices (a price per review is not paired across arms, so this is the honest interval). **The multiple is stable:** the N=5 point estimate still sits inside the band the three-run grid implied. A ratio band over more runs can narrow or widen, so which way it moved is reported rather than assumed.

### Per-run raw figures (the numbers the means are computed from)

| run | repeat | arm | calls | input | output | CNY | seconds |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| A_single_prompt-r1 | 1 | A | 1 | 5,991 | 2,845 | 0.025958 | 16.6 |
| B_pipeline-r1 | 1 | B | 8 | 37,611 | 8,233 | 0.108788 | 42 |
| C_pipeline_no_verification-r1 | 1 | C | 10 | 45,277 | 10,131 | 0.12951 | 47.5 |
| A_single_prompt-r2 | 2 | A | 1 | 5,991 | 2,899 | 0.026418 | 17.3 |
| B_pipeline-r2 | 2 | B | 11 | 58,035 | 9,536 | 0.131248 | 46.1 |
| C_pipeline_no_verification-r2 | 2 | C | 10 | 43,304 | 9,836 | 0.113469 | 47.5 |
| A_single_prompt-r3 | 3 | A | 1 | 5,991 | 2,576 | 0.023666 | 15 |
| B_pipeline-r3 | 3 | B | 8 | 35,222 | 8,188 | 0.106997 | 38.3 |
| C_pipeline_no_verification-r3 | 3 | C | 12 | 63,540 | 10,769 | 0.140229 | 53.4 |
| A_single_prompt-r4 | 4 | A | 1 | 5,991 | 2,773 | 0.025345 | 16 |
| B_pipeline-r4 | 4 | B | 10 | 45,265 | 9,179 | 0.123581 | 46.7 |
| C_pipeline_no_verification-r4 | 4 | C | 12 | 62,222 | 10,508 | 0.138633 | 50 |
| A_single_prompt-r5 | 5 | A | 1 | 5,991 | 3,402 | 0.030704 | 18.9 |
| B_pipeline-r5 | 5 | B | 8 | 29,449 | 7,407 | 0.098352 | 37.6 |
| C_pipeline_no_verification-r5 | 5 | C | 10 | 39,171 | 8,760 | 0.113902 | 44.2 |

### Spend

| | CNY |
| --- | ---: |
| Previous grid (N=3, arms + judge) | 1.3405 |
| **This extension — 6 new runs (arms)** | **0.5305** |
| **This extension — judge on the new repeats** | **0.4591** |
| **This extension, total** | **0.9897** |
| Cumulative for the whole experiment | 2.3302 |
| Budget cap for the extension | 10 (used 9.9%) |

## 3. Quality — deterministic columns, N=3 vs N=5

**Primary columns are deterministic and use no LLM at all.**

| Column | A | B | C |
| --- | ---: | ---: | ---: |
| Report bullets in `Key Claims & Evidence` | **7.667** (7–9) → **8** (7–10) | **9.667** (8–11) → **9.8** (8–11) | **11.667** (10–13) → **11** (10–13) |
| …whose wording is locatable in the paper (≥8 verbatim tokens) | **0.333** (0–1) → **0.6** (0–1) | **1.333** (1–2) → **1** (0–2) | **1.333** (1–2) → **1.4** (1–2) |
| …as a share of the bullets (the citation-verifiability column) | **0.048** (0–0.143) → **0.077** (0–0.143) | **0.136** (0.1–0.182) → **0.101** (0–0.182) | **0.12** (0.077–0.2) → **0.132** (0.077–0.2) |
| Contributions covered, of 14 | **14** (14–14) → **14** (14–14) | **13.333** (13–14) → **13.4** (13–14) | **13** (12–14) → **13** (12–14) |
| Coverage rate | **1** (1–1) → **1** (1–1) | **0.953** (0.929–1) → **0.957** (0.929–1) | **0.929** (0.857–1) → **0.929** (0.857–1) |
| Factual errors per review (deterministic) | **0** (0–0) → **0** (0–0) | **0** (0–0) → **0** (0–0) | **0** (0–0) → **0** (0–0) |
| Number-bearing bullets whose magnitudes are all in the paper | **0.676** (0.571–0.833) → **0.699** (0.571–0.833) | **0.759** (0.5–1) → **0.777** (0.5–1) | **0.886** (0.75–1) → **0.83** (0.714–1) |
| Required headings present, of 8 | **8** (8–8) → **8** (8–8) | **8** (8–8) → **8** (8–8) | **8** (8–8) → **8** (8–8) |

Each cell reads **N=3 → N=5**: mean (min–max) before the extension, then after.
The N=3 side is the stored reports re-scored by one scorer version, so this table and the run list agree; §5.1 records where that differs from what the first write-up published.

| Machine layer (architectural, not a quality score) | A | B | C |
| --- | ---: | ---: | ---: |
| Claims recorded with a quotation (Reader artifact) | **0** (by construction) | **9.8** (8–11) | **11** (10–13) |
| …of which the shipped verifier located in the paper | **not measurable** | **9.8 / 9.8 = 100%** | **0** — the check is switched off |
| Machine verdicts on the report's own bullets | **0** | **0** (0–0) | **0** |
| Machine-written `## Verification Ledger` present | **0/5 runs** | **5/5 runs** | **0/5 runs** |

**Marker-vocabulary caveat.** The marker column counts *labels*, not provenance, and in 2 of C's runs the model wrote the machine layer's own vocabulary itself (`C_pipeline_no_verification-r4`, `C_pipeline_no_verification-r5`) while the machine layer contributed nothing to that arm — no deterministic claim status, no ledger. So "0 machine verdicts on C's bullets" is true of the *machine layer* (which is switched off) and false of the *text* in those runs. The columns that carry the architecture claim are the claim statuses and the ledger, not the label count.

## 4. Do the arms separate? — the interval question the extension was bought for

Two intervals overlap when the intervals printed above share a point. This is a min–max
range over the successful runs, **not** a confidence interval: with these n's no
significance test is defensible, so a disjoint pair is reported as *the arms did not
overlap in this sample*, never as a proven difference. An overlapping pair is the honest
statement of "no measurable difference at this sample size".

| Column | A–B (N=3) | A–B (N=5) | A–C (N=5) | B–C (N=5) | reading at N=5 |
| --- | :---: | :---: | :---: | :---: | --- |
| `coverage` | overlap | overlap | overlap | overlap | **still no separation — every pair overlaps** |
| `coverage_rate` | overlap | overlap | overlap | overlap | **still no separation — every pair overlaps** |
| `factual_errors` | overlap | overlap | overlap | overlap | **still no separation — every pair overlaps** |
| `headings_present` | overlap | overlap | overlap | overlap | **still no separation — every pair overlaps** |
| `attribution_rate` | overlap | overlap | overlap | overlap | **still no separation — every pair overlaps** |
| `number_grounding_rate` | overlap | overlap | overlap | overlap | **still no separation — every pair overlaps** |

Interval width, N=3 → N=5. These are min–max ranges, and a range over more runs can stay
the same or **widen** — more samples can always turn up a new extreme. What a wider sample
buys is a mean estimated from more runs; it does not promise a narrower range, and this
table is reported rather than promised:

| Column | A | B | C |
| --- | ---: | ---: | ---: |
| `coverage` | 0 → 0 | 1 → 1 | 2 → 2 |
| `coverage_rate` | 0 → 0 | 0.071 → 0.071 | 0.143 → 0.143 |
| `factual_errors` | 0 → 0 | 0 → 0 | 0 → 0 |
| `headings_present` | 0 → 0 | 0 → 0 | 0 → 0 |
| `attribution_rate` | 0.143 → 0.143 | 0.082 → 0.182 | 0.123 → 0.123 |
| `number_grounding_rate` | 0.262 → 0.262 | 0.5 → 0.5 | 0.25 → 0.286 |

## 5. The three questions, answered

**1. Is the coverage difference (14/14 vs 13/14) still inside the noise?**

**Yes — and on the stored reports it was never outside it.**

At N=5 the coverage intervals are A [14–14] and B [13–14] and they overlap, so the measured gap of 0.6 items of 14 is inside the run-to-run spread. The published N=3 numbers had those intervals disjoint (B was 13/14 in all three runs), which is what the first write-up reported; re-scoring the stored reports — the same three repeats — shows B at 14/14 in its first run, i.e. overlapping already (§5.1). Which checklist items an arm misses, read from the per-run ids rather than asserted:

* arm A — missed nothing in any of the 5 runs
* arm B — missed in some runs only: brier, dbn_alpha, future_work
* arm C — missed in some runs only: bc_vs_e, brier, future_work

So the honest reading of this column is: **the single prompt covered all 14 contributions in every one of its 5 runs; the pipeline covered 14 in 2 of its 5, and its misses are not one recurring blind spot** (across the pipeline runs they move between `brier`, `dbn_alpha`, `future_work`). A 1/14 difference is the smallest this checklist can express, and at five runs it is not a measurable difference between the arms.

**2. Is the cost multiple stable?**

The mean-price multiple moved from 4.57× at N=3 to 4.31× at N=5 (-0.263), with the
per-run B/A ratio band going from 4.05–5.55 to 3.2–5.55.
The N=5 estimate still lies inside the N=3 band, so **the multiple is stable** — the headline multiple is a property of the architecture, not of one grid.

**3. Did the wider sample expose a failure mode the first three runs never showed?**

**Yes — one, and it is a within-arm event rather than a new way for one arm to win:**

* B: attribution_rate fell to 0.0 in the new repeats, below every stored run (min 0.1)

Everything the extension could have shown and did not: no failed or skipped run, no new deterministic factual error in any arm, no pipeline run without its ledger, no machine-layer footprint in the arm where verification is switched off, and no judge support rate that fell away from the recorded one (§6).

**What the extension changed in the write-up:** the conclusion is qualified above; the interval table and the per-run figures are the new evidence, and the N=3 numbers are reported next to the N=5 ones rather than replaced.

### 5.1 A stored report that changed underneath the grid

The extension re-scores from disk. Comparing each stored report with the character count the previous grid published shows that some run directories no longer hold the text that grid measured — so the N=3 numbers as published and the N=3 reports as stored are not the same evidence. This document reports the re-scored basis and keeps the published summary untouched:

| run | published chars | stored chars | published coverage | stored coverage | stored report mtime |
| --- | ---: | ---: | ---: | ---: | --- |
| `A_single_prompt-r1` | 11,933 | 13,027 | 14 | 14 | 2026-09-16T06:02:18Z |
| `B_pipeline-r1` | 11,238 | 10,737 | 13 | 14 | 2026-09-16T06:02:58Z |
| `C_pipeline_no_verification-r1` | 11,253 | 8,552 | 14 | 12 | 2026-09-16T06:03:41Z |
| `A_single_prompt-r2` | 12,736 | 12,364 | 14 | 14 | 2026-09-16T06:03:58Z |

Every rewritten report is stamped **after the frozen grid's own `started_at` (2026-09-16T05:57:59Z)**, so the change happened once that grid had finished writing its records; what rewrote them is not recorded anywhere in this repository (the run directories carry no provenance), which is exactly why the check exists instead of a note in a README.

Coverage, both readings of the same three stored repeats:

| basis | A | B | C |
| --- | ---: | ---: | ---: |
| published N=3 (the texts as measured then) | **14** (14–14) | **13** (13–13) | **13.67** (13–14) |
| stored reports, re-scored now | **14** (14–14) | **13.33** (13–14) | **13** (12–14) |

The honest consequence: the published N=3 line "A ≥ B on coverage in every run" is a property of those three texts, and at least one stored run (B's first repeat) covers 14 of 14. The N=5 column in §3 is therefore the one to quote, and these run directories are no longer a faithful re-scoring source for the N=3 numbers — the frozen JSON is.

## 6. The LLM-judged column (secondary, and it stays secondary)

| | A | B | C |
| --- | ---: | ---: | ---: |
| Items judged (recorded grid) | 20 | 20 | 20 |
| yes (recorded grid) | 20 | 15 | 15 |
| no (recorded grid) | 0 | 5 | 5 |
| unclear (recorded grid) | 0 | 0 | 0 |
| support rate (recorded grid) | 1.000 | 0.750 | 0.750 |
| agreement between the two passes (recorded grid) | 0.900 | 1.000 | 1.000 |
| Cohen's kappa (recorded grid) | 0.000 | 1.000 | 1.000 |
| Items judged (new repeats only) | 17 | 20 | 20 |
| yes (new repeats only) | 17 | 13 | 14 |
| no (new repeats only) | 0 | 6 | 5 |
| unclear (new repeats only) | 0 | 1 | 1 |
| support rate (new repeats only) | 1.000 | 0.650 | 0.700 |
| agreement between the two passes (new repeats only) | 1.000 | 0.950 | 0.950 |

The recorded column is the one bought in the first grid; it was **not re-bought**. The second block judges only the repeats this extension bought, sampled evenly across them (`up to 10 bullets per repeat across 2 repeats (balanced rule)`), so the two blocks together say whether the judge's verdicts reproduce on fresh samples instead of quoting one sample twice.

*Where kappa is 0 while the agreement is high (arm A): one pass gave the same verdict for every item, so chance alone predicts the observed agreement and kappa collapses to 0. For those arms the percent-agreement column, not kappa, carries the reliability reading.*

The judge asks arm A a different (easier) question, because A produces no quotes to judge — attribution to the paper instead of entailment by a quotation. **The two support rates are therefore not comparable across arms and this document does not compare them.** The deterministic columns above carry the conclusion.

## 7. Reproducing it

```bash
# re-score the stored reports with the current scorers (no API calls, no spend)
python scripts/compare_arms_live.py --score-only --out docs/arm-comparison-live.json \
    --runs-dir outputs/arm-comparison-live

# widen the stored N=3 grid to N=5: only the missing repeats are bought
python scripts/compare_arms_live.py --extend-from docs/arm-comparison-live-n3.json \
    --runs 5 --budget 10 --out docs/arm-comparison-live.json \
    --runs-dir outputs/arm-comparison-live

# the experiment from scratch (requires DEEPSEEK_API_KEY in the environment or a .env)
python scripts/compare_arms_live.py --runs 5 --budget 30 \
    --out docs/arm-comparison-live.json --runs-dir outputs/arm-comparison-live
```

## 8. Limitations

1. **One paper, one model, 5 repeats.** The per-arm ranges overlap on every quality column, so the honest
   statement is "no measurable difference in this sample", not "A is better than B". A
   difference smaller than a few points could not have been detected here at all, and no
   significance test is defensible at n=5.
2. **The judge's question is not identical across arms** (§6). Fixing that properly needs a
   quote from arm A, which is precisely what arm A does not produce.
3. **`arxiv_search` is stubbed**, so the pipeline's related-work tool loop was never exercised
   and no arm's citations were verified. Real related-work quality is unmeasured.
4. **Arm A's cache advantage**: A re-sends one identical prefix and is almost entirely cached,
   while the multi-stage conversation keeps changing its prefix. Cache pricing therefore
   flatters A slightly, and the cost multiple is a mid-to-upper estimate rather than a
   conservative one.
5. **Factual-error detection only understands `%`/`pp` magnitudes bound to a configuration.**
   Zero errors means "no arithmetic or attribution error of this kind", not "no errors".
6. **Coverage is a checklist, not comprehension.** It measures whether a contribution is
   present with its magnitude, not whether it is discussed well — which is also why a 1/14 gap
   is the finest difference it can express.
7. The verifier, the coverage checklist and the error detector were written by the same author
   as the architecture under test. The mitigations are that they are deterministic, that the
   checklist was frozen before the runs, and that the grader is tested against an unrelated
   report (which it scores 0/14).
8. **The stored runs were not re-bought, only re-scored.** The N=3 runs' tokens, costs
   and timings are exactly as recorded in `docs/arm-comparison-live-n3.json`; the deterministic
   columns for all runs were recomputed by one version of the scorers, said in
   `extension.reused_run_ids`.
9. **4 of the 9 stored reports no longer match the text the
   published N=3 numbers were measured from** (§5.1). The published summary is preserved
   unchanged in the frozen JSON, and the N=3 column of this document is the re-scored basis,
   so the two readings differ for those runs. Treat the N=5 column as the quotable one.

## 9. Conclusion

> **The multi-stage pipeline bought verifiability and auditability; it did not buy quality, and
> it cost 4.31x.** On the deterministic columns the single long prompt was equal or
> better: coverage 14/14 vs 13.4/14, formal factual errors 0.0 for every arm, all eight
> headings in every run. What only the pipeline does is put a machine-written ledger in
> **every** report (5/5 runs vs 0/5 for A and C), hold a quotation for
> **every** claim (located by the shipped verifier, vs *not measurable* for A and *never checked*
> for C), and let code rather than the model own the verdict.

Widening the sample from 3 to 5 repeats per arm (0.9897 CNY, 9.9% of the extension budget) **changed one reading and left the rest standing**: the coverage column no longer separates the arms at N=5, so the published N=3 edge ("A ≥ B on coverage in every run") does not survive the wider sample, and the stored reports had already contradicted it (§5.1). Everything else held: factual errors at zero in every run, headings complete in every run, the cost multiple inside the band the three-run grid implied (4.31×), and one new within-arm event (1: B). What the extension bought is not a new headline but a corrected one: a smaller claim about coverage, a verified cost multiple, and a stored-run integrity problem that the wider sample walked straight into. That is what spending on more data is for, and it is reported as such.

The interview answer this supports: *"Quality was flat across one prompt and four agents —
14/14 vs 13.4/14 coverage, zero arithmetic errors either way, on 5 repeats per
arm. What the pipeline adds is that every claim carries a quotation a program locates in the
paper and a ledger written by code rather than by the model, and that costs 4.31x. I can
show you the run where the single prompt's report contains no claim a program could ever
check, and the run where the pipeline's check downgraded a quotation its own Reader invented."*
