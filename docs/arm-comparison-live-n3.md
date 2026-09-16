# Three-arm comparison under a real model — results

**What this settles:** `docs/arm-comparison.json` was a `FakeLLM` script, so
PaperFlow had data about its *architecture* and none about review *quality*.
This document reports the experiment that fills that gap: the same paper,
the same task text, the same model, three architectures, three repeats each,
with real tokens and real money.

Everything below comes from `docs/arm-comparison-live.json` (machine-readable,
per-run) and the run directories under `outputs/arm-comparison-live/`. The
protocol was fixed in advance in
[`arm-comparison-live-design.md`](arm-comparison-live-design.md).

**Headline: the pipeline did not buy quality. It bought verifiability and
auditability, at 4.6x the cost.**

---

## 1. Setup

| | |
| --- | --- |
| Model | `deepseek-chat`, served and billed as **`deepseek-flash`** (the API reports the served name) |
| Temperature | 0.2 (the project default; non-zero so the repeats sample real variance) |
| Thinking mode | explicitly `disabled` — the project has never used a reasoning model and thinking tokens bill as output |
| Paper | `research/werewolf-multiagent-paper.pdf`, 21,902 chars extracted, **no truncation** in any arm |
| Runs | 3 per arm, 9 total, interleaved A→B→C by repeat |
| Failures | **0** — 9/9 runs produced a report |
| Judge | same model, temperature 0, fixed prompt v1.0, batch 10, ≤20 items/arm, every batch judged twice with the option order swapped |

`arxiv_search` was stubbed to an empty result set so the only paid dependency
was DeepSeek and every arm saw byte-identical input. See §7 for what that
costs the experiment.

## 2. Cost accounting — measured, from the API's own `usage` block

DeepSeek bills input twice (cache hit / cache miss) and output once, so the
ledger keeps those three buckets apart. Prices are the published ones
(`api-docs.deepseek.com`, read 2026-09-16) at **peak** rates — the more
expensive of the two — converted at 7.1 CNY/USD. Cost = tokens × those prices.

| | A single prompt | B pipeline | C pipeline, verification off |
| --- | ---: | ---: | ---: |
| LLM calls per review | **1.0** (1–1) | 9.0 (8–11) | 10.7 (10–12) |
| Input tokens per review | **5,991** (5,991–5,991) | 43,623 (35,428–53,901) | 50,707 (36,143–65,889) |
| Output tokens per review | **2,773** (2,539–2,888) | 8,652 (7,592–9,395) | 10,245 (8,978–11,423) |
| of which cache hits (input) | 5,760 (96.1%) | 26,581 (60.9%) | 35,243 (69.5%) |
| Wall-clock per review | **16.3 s** (15.0–17.3) | 42.1 s (38.3–46.1) | 49.5 s (47.5–53.4) |
| **CNY per review** | **0.0253** | **0.1157** | **0.1277** |
| vs arm A | 1.00x | **4.57x** | 5.05x |

Per-run raw figures (the numbers the means are computed from):

| run | arm | calls | input | output | CNY | seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| r1 | A | 1 | 5,991 | 2,845 | 0.025958 | 16.6 |
| r1 | B | 8 | 37,611 | 8,233 | 0.108788 | 42.0 |
| r1 | C | 10 | 45,277 | 10,131 | 0.129510 | 47.5 |
| r2 | A | 1 | 5,991 | 2,899 | 0.026418 | 17.3 |
| r2 | B | 11 | 58,035 | 9,536 | 0.131248 | 46.1 |
| r2 | C | 10 | 43,304 | 9,836 | 0.113469 | 47.5 |
| r3 | A | 1 | 5,991 | 2,576 | 0.023666 | 15.0 |
| r3 | B | 8 | 35,222 | 8,188 | 0.106997 | 38.3 |
| r3 | C | 12 | 63,540 | 10,769 | 0.140229 | 53.4 |

### The cost section on its own

> Producing one structured review of this paper costs **¥0.0253 with one long
> prompt** and **¥0.1157 with the four-agent pipeline** — **4.57x**, or +¥0.0904
> per review. What that 4.57x buys is: **① every claim carries a quotation a
> program can locate in the paper; ② the verdict on those quotations is decided
> by code, not by the model's own summary of itself; ③ the report and the
> machine verdict cannot silently disagree — a machine-written ledger is
> appended to every report and a consistency check logs any disagreement.**
> It does **not** buy better coverage, fewer factual errors, or more supported
> claims (see §4).

Two details that make the multiple larger than it looks:

* Arm B's input is 7.3x arm A's, but B also gets a much **worse cache-hit
  rate** (60.9% vs A's 96.1%) because the multi-stage conversation keeps
  changing its prefix, while arm A re-sends one identical prefix and is almost
  entirely cached. Input at a cache miss costs 10x input at a cache hit, so B
  pays twice for the same growth. C, which shares B's architecture but skips
  the verification stage, lands in between at 69.5%. On a paper with a colder
  cache the gap would be smaller.
* The pipeline's call count is the whole story of the architecture: 1 vs 9.

### Cumulative spend

| | CNY |
| --- | ---: |
| First grid (9 runs + judge), superseded | 1.3265 |
| **Final grid — 9 runs (arms)** | **0.8063** |
| **Final grid — semantic judge (12 calls)** | **0.5342** |
| **Final grid total** | **1.3405** |
| **Cumulative measured spend, all of this session** | **≈ 2.67** |
| Budget cap | 30.00 |
| Headroom remaining | 95.5% |

The cap was enforced *before* each request, on the worst case of that request
(its input + its `max_tokens` of output), so no run could discover an overrun
after paying for it. It never triggered: the whole experiment cost 4.5% of the
budget.

## 3. The three arms

| Arm | Definition |
| --- | --- |
| **A** `A_single_prompt` | The paper's full text plus the task, in **one** `chat()` call. Given the Synthesizer's own system prompt, so all three arms are asked for the same eight headings. No board, no artifacts, no tools, no per-claim record. |
| **B** `B_pipeline` | The shipped Researcher → Reader → Critic → Synthesizer pipeline with the deterministic quote check, the report repair and the machine-written `## Verification Ledger`. |
| **C** `C_pipeline_no_verification` | The identical pipeline with the quote check, report repair and consistency check switched off — the pre-fix architecture, reproduced deliberately. |

## 4. Quality — the three metric families

**Primary columns are deterministic and use no LLM at all.** Means with
[min–max] over 3 runs per arm.

### 4.1 Citation verifiability

| Column | A | B | C |
| --- | ---: | ---: | ---: |
| Claims recorded with a quotation (Reader artifact) | **0** (by construction) | 11.0 (10–12) | 12.3 (12–13) |
| …of which the shipped verifier located in the paper | **not measurable** | **11.0 / 11.0 = 100%** | **0** — the check is switched off |
| Report bullets whose wording is locatable in the paper (≥8 consecutive verbatim tokens) | 5.6% (0–16.7%) | **17.7%** (10–25%) | 10.9% (7.7–16.7%) |
| Machine verdicts on the report's own bullets | **0** | 0 | **0** |
| Machine-written `## Verification Ledger` present | **0/3 runs** | **3/3 runs** | **0/3 runs** |

Read this carefully, because the tempting reading is wrong. Arm A's
"not measurable" is not a zero: A never records a claim, so there is nothing
to check — that *is* the architectural difference, and it is why A's
verifiability cannot be improved by a better model. C records quotations just
like B, but nothing in C **decides** whether they exist.

The "machine verdicts on bullets = 0" for B needs a word, because the ledger
is right there. In these three runs B's Reader fabricated no quotation, so
nothing had to be downgraded, and the repair step correctly changed nothing.
The guarantee is what matters, and it fired in the harness's offline fixture:
with a fabricated quote in the artifact, B's report bullet is rewritten to
`[unverified]` and listed in the ledger, while C ships the same bullet still
marked `[supported]` (covered by
`tests/test_compare_arms_live.py::test_pipeline_arms_record_per_claim_deterministic_status`
and `tests/test_compare_arms.py`). So: the ledger is always present in B, and
the repair is proven, but in this particular sample there was nothing to
repair.

### 4.2 Factual errors (deterministic)

A number **bound** to a configuration that the paper's Table 2 gives
differently (e.g. "Group E reached 54.6%", where E is 68.8%), or a magnitude
that is neither printed in the paper nor derivable from the paper's own
numbers.

| | A | B | C |
| --- | ---: | ---: | ---: |
| Factual errors per review | **0.0** | **0.0** | **0.0** |
| Number-bearing bullets whose every magnitude is in the paper | 77.8% | 71.8% | 72.0% |

All nine reports were clean on the strict check. An earlier scoring pass
flagged 1.7–2.3 errors per arm; **those were measurement bugs, not model
errors** — a comparison list ("Group E: A=44.2%, B=54.6%, C=53.8%, E=68.8%")
was cascading into five false misattributions, and arithmetic derivable from
the table ("E is +10.2 pp over D") was counted as fabrication. Both were fixed,
and every stored report was re-scored without buying new samples, in one
deliberate pass (§6).

### 4.3 Coverage

14 contributions were listed from the paper's own Table 1, Table 2, Results
and Conclusions **before** any arm ran, each requiring its concept *and* a
magnitude so that listing vocabulary cannot score a point
(`scripts/paper_ground_truth.py`).

| | A | B | C |
| --- | ---: | ---: | ---: |
| Contributions covered, of 14 | **14.0** (14–14) | 13.0 (13–13) | 13.7 (13–14) |
| Coverage rate | **1.00** | 0.929 | 0.976 |
| Required headings present, of 8 | 8/8 | 8/8 | 8/8 |

**A ≥ B on every run.** Arm B covered 13 of 14 in all three runs; A covered 14
of 14 in all three. The single missing point for B is the same one each time.
This is the experiment's least flattering column for the pipeline and it is the
column least distorted by the harness, so it carries the conclusion.

## 5. The LLM-judged column, and why it cannot crown a winner

Secondary by design: one model judged the report's claim bullets, twice per
batch with the option order swapped.

| | A | B | C |
| --- | ---: | ---: | ---: |
| Question asked | does the **paper** contain what the claim asserts? | does the **attached quote** support the claim? | does the **attached quote** support the claim? |
| Items judged | 20 | 20 | 20 |
| yes / no / unclear | **20 / 0 / 0** | 15 / 5 / 0 | 15 / 5 / 0 |
| support rate | **1.00** | 0.75 | 0.75 |
| agreement between the two passes | 0.90 | **1.00** | **1.00** |
| Cohen's kappa | 0.0 (both passes constant → undefined discrimination) | 1.00 | 1.00 |

**Do not read A = 1.00 as "A's evidence is better".** A was asked the *easier*
question, because A produces no quotes to judge: attribution to the paper is a
weaker test than "does this specific quotation entail this claim". The two
support rates are therefore not comparable across arms, and this document does
not compare them. The honest use of this column is *within* an arm.

The judged verdicts are highly consistent (kappa 1.00 for B and C), which
means the judge is measuring something real and repeatable, not noise. What it
measures is strictness about **compound claims**: every one of the ten "no"
verdicts across B and C is the same pattern — a bullet that asserts two or
three things while its quotation contains only one of them. The judge's own
recorded reasons say so verbatim: *"Quote lacks loop-breaking rationale and
alpha value"*, *"Quote omits cross-architecture and cognitive load"*,
*"Quote gives 24.6pp, not sum of gains"*. A human reviewer would accept most
of those quotations as supporting evidence; a strict literal reader does not.
Both readings are defensible, which is exactly why this column is reported as
context and the deterministic columns carry the conclusion.

## 6. Reproducing it

```bash
# re-score the stored reports with the current scorers (no API calls, no spend)
python scripts/compare_arms_live.py --score-only --out docs/arm-comparison-live.json \
    --runs-dir outputs/arm-comparison-live

# the experiment itself (requires DEEPSEEK_API_KEY in the environment or a .env)
python scripts/compare_arms_live.py --runs 3 --budget 30 \
    --out docs/arm-comparison-live.json --runs-dir outputs/arm-comparison-live
```

`--score-only` exists because the deterministic scorers were fixed after the
first grid: it recomputes the deterministic columns from the **same stored
reports** and was verified idempotent. Model outputs, token counts, costs and
timings are untouched by it. The judge column in the final JSON was bought
once, in the final grid, and was not re-run.

## 7. Limitations

1. **One paper, one model, three repeats.** The per-arm ranges overlap on every
   quality column, so the honest statement is "no measurable quality
   difference in this sample", not "A is 7% better than B". A difference
   smaller than a few points could not have been detected here at all.
2. **The judge's question is not identical across arms** (§5). Fixing that
   properly needs a quote from arm A, which is precisely what arm A does not
   produce.
3. **`arxiv_search` is stubbed**, so the pipeline's related-work tool loop was
   never exercised and no arm's citations were verified. Real related-work
   quality is unmeasured.
4. **Arm A's English-only tokenization**: A's input was 96.1% cache hits
   because it re-sends one identical prefix; B's 60.9% is hurt by the
   multi-stage conversation changing its prefix. Cache pricing therefore
   flatters A slightly, and the 4.57x is a mid-to-upper estimate of the true
   cost ratio rather than a conservative one.
5. **Factual-error detection only understands `%`/`pp` magnitudes bound to a
   configuration.** A report that misstates a method, an author or a
   mechanism is invisible to it. Zero errors means "no arithmetic or
   attribution error of this kind", not "no errors".
6. **Coverage is a checklist, not comprehension.** It measures whether a
   contribution is present with its magnitude, not whether it is discussed
   well.
7. The verifier, the coverage checklist and the error detector were written by
   the same author as the architecture under test. The mitigations are that
   they are deterministic, that the checklist was frozen before the runs, and
   that the grader is tested against an unrelated report (which it scores 0/14)
   and against an arm it could easily have flattered.

## 8. Conclusion

> **The multi-stage pipeline bought verifiability and auditability; it did not
> buy quality, and it cost 4.57x.** On the deterministic columns the single
> long prompt was equal or better: coverage 14.0/14 vs 13.0/14, formal
> factual errors 0.0 vs 0.0, all eight headings in every run. What only the
> pipeline does is put a machine-written ledger in **every** report (3/3 runs
> vs 0/3 for A and C), hold a quotation for **every** claim (11.0/11.0
> located, vs *not measurable* for A and *never checked* for C), and let code
> rather than the model own the verdict — proven by the offline fixture where
> B rewrites a fabricated quotation to `[unverified]` while C ships it as
> `[supported]`. For a research-review tool whose value proposition is
> "check the claims before you trust them", that is the product; it is worth
> saying plainly that it is not a quality gain, and that at this paper's size
> a single long prompt is the better buy for quality alone.

The interview answer this supports: *"For one paper of this size, quality was
flat across one prompt and four agents — 14/14 vs 13/14 coverage, zero
arithmetic errors either way. What the pipeline adds is that every claim
carries a quotation a program locates in the paper and a ledger written by
code rather than by the model, and that costs 4.6x, or about nine cents a
review. I can show you the run where the single prompt's report contains no
claim a program could ever check, and the run where the pipeline's check
downgraded a quotation its own Reader invented."*
