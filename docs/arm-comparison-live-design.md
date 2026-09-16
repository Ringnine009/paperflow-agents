# Three-arm comparison under a real model — design (pre-registered)

**Status: protocol fixed before the grid was run.** The numbers live in
`docs/arm-comparison-live.json` and are written up in
`docs/arm-comparison-live.md`. This file records what was decided *in advance*
so the report cannot be accused of choosing its success criteria afterwards.

Date: 2026-09-16. Model: `deepseek-chat` (served as `deepseek-flash`, which is
what the API reports and what is billed). Temperature: 0.2 (the project
default). Thinking mode: explicitly disabled, because the project has never
used a reasoning model and thinking tokens are billed as output.

---

## 1. Question

The offline harness (`scripts/compare_arms.py`) measured only what the
*architecture* decides. It could not answer the question an interviewer asks
first: **"your four agents versus one long prompt — is the review better, and
is whatever it buys worth its cost?"**

That needs a real model, real tokens, real money. This experiment supplies
them.

## 2. Arms (same paper, same task text, same model, same temperature)

| Arm | What runs |
| --- | --- |
| **A** `A_single_prompt` | The paper's full text + the task in **one** `chat()` call. No board, no artifacts, no tools, no per-claim record. |
| **B** `B_pipeline` | The shipped Researcher → Reader → Critic → Synthesizer pipeline, with the deterministic quote check and the machine-written `## Verification Ledger` enabled. |
| **C** `C_pipeline_no_verification` | The identical pipeline with the deterministic quote check, the report repair and the consistency check switched off — the pre-fix architecture. |

Arm A is given the *Synthesizer's own system prompt*, so all three arms are
asked for the same eight headings and the same deliverable. The single
difference is that A has no agents, tools, artifacts or machine layer around
it.

## 3. Paper and inputs

* `research/werewolf-multiagent-paper.pdf` — 21,902 chars of extracted text,
  307 lines, well under the 60,000-char `max_fulltext_chars`, so **no
  truncation occurs in any arm**.
* The ground-truth key (`scripts/paper_ground_truth.py`) was transcribed from
  the paper's own Table 1, Table 2 and Conclusions **before** any arm ran.
* `arxiv_search` is stubbed to return an empty result set. The paper is a
  local PDF, so the only paid dependency is DeepSeek and the input is
  byte-identical for every arm and repeat. **This is a limitation**: the
  pipeline's related-work tool loop is therefore not exercised, and C's
  "Related Work" section cannot cite anything.

## 4. Repetition

N = 3 independent runs per arm, 9 runs total, interleaved A→B→C by repeat so
that a budget stop leaves a balanced grid rather than three copies of one arm.
Temperature is 0.2, not 0, so the repeats genuinely sample the model's
variance; mean **and** min/max are reported, never the best run alone.

## 5. Metrics, and which ones are allowed to carry the conclusion

**Primary (deterministic; no LLM anywhere in the loop)**

| Column | Definition |
| --- | --- |
| heading completeness | how many of the 8 required `##` headings the report contains |
| report attribution | share of the report's claim bullets containing a run of ≥8 consecutive tokens that appear verbatim in the paper (the project's own canonical-token normalization) |
| quote evidence | share of the Reader's claims whose quote the shipped verifier located in the paper text. **Not measurable for A**: it produces no claim artifact |
| auditability in the deliverable | whether the report carries machine verdicts (`[verified]`/`[unverified]`/`[unverifiable]`) and the `## Verification Ledger` |
| number grounding | share of claim bullets whose every magnitude occurs in the paper text |
| factual errors | a number attributed to a configuration the paper's Table 2 attributes differently, or a percentage that appears nowhere in the paper |
| coverage | how many of 14 pre-listed contributions the report covers, each requiring its concept **and** magnitude |

**Secondary (LLM-judged, declared protocol)**

"Does the attached quote actually support the claim?" — same model,
temperature 0, fixed prompt (version recorded), items = the report's claim
bullets, batched at 10, ≤20 items per arm, **each batch judged twice with the
option order swapped**, and the agreement between the two passes reported
alongside. An `unclear` verdict is never counted as support.

The deterministic columns decide the conclusion. The judge column is context.

## 6. Cost accounting and the stop rule

* Tokens come from the API's own `usage` block, including DeepSeek's cache
  hit/miss split, because that split is what is actually billed.
* Cost = tokens × the prices published at `api-docs.deepseek.com`
  (`paperflow.core.pricing`), converted at 7.1 CNY/USD and
  **quoted at peak rates** — the more expensive of the two, so the reported
  figure is never flattering.
* The cap is **¥30**, enforced *before* each request on the worst case of
  that request (its input + its `max_tokens` of output), so the experiment
  cannot discover an overrun after paying for it.

## 7. Pre-registered success criteria

1. **Quality** — B is better than A on the primary columns if it is ahead on
   coverage *and* has no more factual errors, beyond run-to-run spread.
2. **Verifiability** — B beats A and C on auditability if its reports carry
   machine verdicts and a ledger in every run while A's and C's carry none.
   This is expected to hold by construction; it is measured, not assumed.
3. **Cost** — B's cost per review over A's, reported as a multiple.
4. **If A wins or ties on quality**, that is the finding. It is written up as
   the finding, with the reason, and no re-running to find a better sample.

## 8. What this experiment still cannot show

* Whether the reviews are *useful to a human* — that needs reader studies.
* Anything about other papers, models or temperatures: one paper, one model,
  three repeats. The per-arm spread is reported so the reader can see how much
  of a difference the sample could even support.
* Related-work quality, since `arxiv_search` is stubbed.
