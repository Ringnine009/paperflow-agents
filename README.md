# PaperFlow — Multi-Agent Paper Review

> Drop a paper link / DOI / PDF into the pipeline and a small team of LLM
> agents collaborates through a JSON task board to produce a **structured
> review with related-work links**.

PaperFlow is a lightweight, dependency-minimal **multi-agent research
collaboration system** built on DeepSeek function calling. It was designed as
a portfolio piece demonstrating how to build agent orchestration from
scratch — no LangChain, no agent frameworks — with an explicit, inspectable
tool-calling loop, a persisted task board, and a deterministic scheduler.

It echoes the author's published multi-agent work —
[*Dynamic Belief Networks and Deep-Thinking Probes for Multi-agent Social
Reasoning*](https://doi.org/10.54254/2753-8818/2026.DL34010) (Tongji
University, 2026) — applying the same "team of specialized reasoners" idea
to literature review.

---

## 中文摘要

PaperFlow 是一个**自研轻量多智能体科研协作系统**：输入论文链接 / DOI / PDF，
四个专职 Agent（Researcher 检索、Reader 结构化解析、Critic 对照原文事实核查、
Synthesizer 整合综述）通过 **JSON 任务板**协作，产出结构化审阅 + 相关工作链接。

- **不依赖 LangChain 等重框架**：Agent 基类 + 工具注册机制 + 任务板（JSON 状态文件）+ 确定性编排器，代码清晰可扩展
- **DeepSeek function calling**（`deepseek-chat`），工具调用循环显式可见、可测试
- 三种入口：arXiv URL / DOI / 本地 PDF；CLI 一键跑 + 轻量 FastAPI Web 看板（实时任务板进度 + 最终报告）
- 该项目的多智能体编排思路与本人在狼人杀多智能体论文
  （DOI `10.54254/2753-8818/2026.DL34010`，DBN 动态信念网络 + DTR 探针）一脉相承

---

## Why

Literature review has two pain points: **reading** (understanding a paper's
structure, claims and numbers) and **contextualizing** (linking it to related
work and judging its relevance). PaperFlow splits the job across four agents
with distinct responsibilities and lets them hand off artifacts through a
shared, human-readable task board:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Task Board (JSON)                         │
│  input ──► agents{researcher, reader, critic, synthesizer}       │
│            artifacts{paper_info, full_text, reader_output,       │
│                      critic_output}   report{path, preview}      │
└──────────────────────────────────────────────────────────────────┘
        ▲                                                           │
        │ persist after every transition                            │
┌───────┴───────────────────────────────────────────────────────────┐
│ Orchestrator  (deterministic scheduler, dependency-aware)         │
│                                                                   │
│  Researcher ──► Reader ──► Critic ──► Synthesizer ──► report.md   │
│      │               │          │            │                    │
│   arxiv_search   (no tools)  search_text   arxiv_search          │
│   resolve_doi    reads full  verifies      (related work)        │
│   fetch_url      text +      quotes vs     + relevance score     │
│   fetch_pdf_text metadata    the paper     vs research focus      │
│   read_pdf                          text                         │
└───────────────────────────────────────────────────────────────────┘
        ▲                                                           │
        │ function-calling loop (explicit, max-round guarded)       │
┌───────┴───────────────────────────────────────────────────────────┐
│ DeepSeek chat/completions (deepseek-chat) + ToolRegistry          │
└───────────────────────────────────────────────────────────────────┘
```

**Design notes**

- **Tool registry** — tools are plain functions wrapped in a `ToolSpec`
  (name / description / JSON Schema / callable). The registry serves both
  the LLM (OpenAI-compatible schemas) and the runtime (execution with
  `call_safe`, so tool failures become LLM-readable `TOOL ERROR` messages).
- **Task board** — a JSON file is the single source of truth for a run.
  The orchestrator persists it after every agent transition, which is what
  makes the web dashboard "live" and crashed runs inspectable.
- **Function-calling loop** — `LLMClient.solve()` is a visible, testable
  loop: send conversation + schemas → execute requested tools via the
  registry → feed results back → repeat until a final answer or the round
  budget is exhausted.
- **Full text is extracted once into a content-addressed file** — the
  Researcher turns the PDF into plain text a single time and stores it on
  the board. The Reader receives the (truncated) full text in its context
  to ground its claims; the Researcher, Critic and Synthesizer only ever
  exchange *paths and previews*, keeping token cost low and quotes
  checkable. Every Reader quote is then verified deterministically against
  the stored text (see below).
- **Deterministic quote verification, and it reaches the deliverable** — for
  every claim the Reader records a machine status (`verified` /
  `unverified` / `unverifiable`), the Critic's verdict is overridden by it,
  and *code* (not the model) rewrites the report: claim bullets that the
  machine downgraded are corrected and a generated `## Verification Ledger`
  section lists every claim that could not be verified. A post-generation
  check compares the report text with the machine status and logs any
  disagreement to the board. Measured on the archived runs, the verifier
  went from **102/131 (77.9%)** to **130/131 (99.2%)** quotes located —
  see [docs/upgrade-notes.md](docs/upgrade-notes.md).
- **Graceful degradation** — the Critic is optional: if fact-checking fails,
  the pipeline continues and the report says so honestly.

---

## Quick start

Requires Python ≥ 3.10 and a [DeepSeek](https://platform.deepseek.com) API key.

```bash
git clone <this-repo> && cd paperflow
python -m venv .venv
.venv\Scripts\activate                # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# Windows:  copy .env.example .env
# macOS/Linux: cp .env.example .env
# then put your DEEPSEEK_API_KEY in .env
```

### CLI

```bash
# arXiv URL
python -m paperflow run "https://arxiv.org/abs/1706.03762"

# DOI (the author's own paper, with a local PDF for full text)
python -m paperflow run "10.54254/2753-8818/2026.DL34010" \
    --pdf path/to/werewolf-paper.pdf

# local PDF
python -m paperflow run path/to/paper.pdf

# results
#   outputs/<run-id>/board.json      <- task board (live state machine)
#   outputs/<run-id>/report.md       <- final structured review
```

The key can also come from any `.env` file:

```bash
python -m paperflow run "https://arxiv.org/abs/1706.03762" \
    --env-file "D:\secrets\.env" --out outputs
```

### Web dashboard

```bash
python -m paperflow serve --port 8080
# open http://127.0.0.1:8080 — paste an entry, watch the task board update
# live, read the rendered report
```

![PaperFlow dashboard with result cards (light theme)](docs/dashboard-v5.png)
![PaperFlow board view: report summary + grouped agent log](docs/board-v5.png)
![PaperFlow verification view](docs/verification-v5.png)

The dashboard is **bilingual (EN / 中文, toggle top-right, default English)**,
with three tabs: **Task board** (live agent states, an **execution timeline**
of the four agents, and a color-coded per-agent activity log), **Report**
(rendered review; claims that appear in the verification data get clickable
`[#n]` markers that jump to their verification card) and **Verification** — a
per-claim view of the deterministic quote check: every claim's quote, the
code's verdict, the reason string it recorded (verbatim search / PDF
hyphenation repaired / token-aligned), where it was found, the surrounding
passage, and the critic's verdict.

Finished runs show their **results directly in the run list**: a report
preview, a "✓ n/m quotes verified" badge and (for failed runs) the failing
stage and reason. Selecting a run whose report is ready opens a **results
summary card** at the top of the task board.

> The dashboard binds `127.0.0.1` by default and the fetch tools refuse
> private/loopback/link-local URLs (SSRF guard) — including on **every
> redirect hop**, since `requests` would otherwise follow a public
> `302 -> http://169.254.169.254/` straight past the check. Not covered: a
> hostname that resolves to a public address when checked and a private one
> when fetched (DNS rebinding) — that needs connection-level IP pinning.
> It is a local tool — do not expose it publicly with `--host 0.0.0.0`.
> Runs interrupted by a server restart are shown as **interrupted**
> (covered by tests).

### Tests

```bash
python -m pytest                      # offline suite (no network, fake LLM) - 149 tests
python -m pytest -m smoke             # real arXiv / Crossref API smoke tests (3)
node tests/test_markdown.mjs          # dashboard markdown renderer (tables) — not part of pytest
node tests/test_timeline.mjs          # execution timeline + report/claim linking — not part of pytest
```

### Verification quality (measured, not claimed)

Replaying every archived run (131 claims) through the old and the new
verifier, plus the report-consistency check applied retroactively:

```bash
python scripts/recompute_verification.py     # before/after + archived report audit
python scripts/compare_arms.py               # single-prompt vs pipeline vs pipeline-without-checks
node scripts/measure_report_linking.mjs      # report bullet -> claim linking coverage
python scripts/demo_ssrf_guard.py            # live SSRF reproduction on 127.0.0.1
```

| check | before | after |
| --- | ---: | ---: |
| quotes located in the paper (131 claims) | 102 (77.9%) | 130 (99.2%) |
| archived reports carrying the machine verdict | 0/17 | new runs: code-injected ledger + consistency check |
| claim bullets contradicting the machine verdict | 38 (in 11 archived reports) | detected, logged and repaired by code |
| report bullets linked to their claim card (118 bullets) | 50 (42%) | 112 (95%) |
| `fetch_url` reachable from loopback | yes (reproduced live) | refused, per redirect hop |

The single remaining miss is a quote that really is not in that run's text
(the Reader invented it) — it must keep failing. Details, including the
match-tier breakdown and the residual risks, are in
[docs/upgrade-notes.md](docs/upgrade-notes.md).

### Does the four-agent pipeline beat one long prompt? (measured, with a real model)

This was the project's one real evidence gap: `scripts/compare_arms.py` runs
on a scripted `FakeLLM`, so it could prove what the *architecture* guarantees
and nothing about review *quality*. The live experiment closes it — 3 arms ×
**5 repeats** on the author's own werewolf paper, one model, one temperature,
real tokens and real money (**¥2.33 all-in**: ¥1.34 for the first 3 repeats,
**¥0.99** to widen them to 5):

```bash
# widen the stored N=3 grid: only the missing repeats are bought
python scripts/compare_arms_live.py --extend-from docs/arm-comparison-live-n3.json \
    --runs 5 --budget 10 --out docs/arm-comparison-live.json \
    --runs-dir outputs/arm-comparison-live
```

| per review | A one long prompt | B pipeline | C pipeline, verification off |
| --- | ---: | ---: | ---: |
| LLM calls | **1** | 9 | 10.8 |
| tokens in / out | **5,991 / 2,899** | 41,116 / 8,509 | 50,703 / 10,001 |
| wall clock | **16.8 s** | 42.1 s | 48.5 s |
| **cost** | **¥0.0264** | ¥0.1138 (**4.31×**) | ¥0.1271 (4.81×) |
| contributions covered (of 14, deterministic) | **14.0** (14–14) | 13.4 (13–14) | 13.0 (12–14) |
| formal factual errors | 0.0 | 0.0 | 0.0 |
| required headings present (of 8) | 8/8 | 8/8 | 8/8 |
| claim bullets a program can locate in the paper | 7.7% | 10.1% | 13.2% |
| claims carrying a machine-checked quote | **none exist** | **9.8/9.8 (100%)** | 0 — check disabled |
| machine-written ledger in the report | **0/5 runs** | **5/5 runs** | 0/5 runs |

**Result: the pipeline did not buy quality — it bought verifiability and
auditability, at 4.31× the cost.** The single long prompt covered the whole
checklist in every one of its five runs (14/14), the pipeline covered 14/14 in
two of five and 13/14 in three, and the two intervals **overlap** — so on this
paper the coverage column does not separate the arms either. No arm made a
formal factual error, and every arm produced all eight headings. What only the
pipeline provides is a quotation per claim that a *program* locates in the
paper and a ledger written by *code* rather than by the model — verified in the
offline fixture, where a fabricated quotation is rewritten to `[unverified]`
in arm B's report while arm C ships it as `[supported]`.

Widening 3 → 5 repeats was worth doing for what it corrected, not for a new
headline: the N=3 write-up claimed "A ≥ B on coverage in every run", and at
N=5 that edge is gone (B hit 14/14 twice). The extension also found that **4 of
the 9 stored N=3 run directories no longer hold the reports the published N=3
numbers were measured from**, so the report now publishes the N=3 column from
a single re-scored basis and keeps the published numbers frozen in
[docs/arm-comparison-live-n3.json](docs/arm-comparison-live-n3.json).

The LLM-judged "does the quote support the claim" column is **secondary and
not comparable across arms** (arm A produces no quote to judge, so it was
asked an easier question); it is reported with its model, prompt, sample size
and a two-pass order-swap reliability check (κ = 1.00 for B and C on the first
grid, 0.90/0.88 on the repeats added by the extension). Full
protocol in [docs/arm-comparison-live-design.md](docs/arm-comparison-live-design.md),
raw numbers and limitations in
[docs/arm-comparison-live.md](docs/arm-comparison-live.md) and
[docs/arm-comparison-live.json](docs/arm-comparison-live.json).

`docs/baseline-plan.md` records the earlier offline-only stage of this work.

---

## Example output (excerpt)

Running the **author's own werewolf paper** (DOI `10.54254/2753-8818/2026.DL34010`)
produces a report like this — the first 20 lines are copied verbatim from
[`examples/werewolf-dbn/report.md`](examples/werewolf-dbn/report.md).
That file (like the other committed samples) is an **archived run from before
the verification ledger existed**, kept unedited as audit evidence — a fresh
run additionally carries the machine-generated `## Verification Ledger`.

```markdown
# Dynamic Belief Networks and Deep-Thinking Probes for Multi-agent Social Reasoning

## Overview

This paper proposes a framework combining a **Dynamic Belief Network (DBN)** and a **Deep-Thinking Token Ratio (DTR) probe** to address two longstanding problems in LLM-powered multi-agent social deduction: recursive agreement between homogeneous agents and stable detection of deception. The framework is evaluated in nine-player Werewolf across 3000 simulated games (six configurations × 500 games) using DeepSeek-V3.2 agents. The DBN maintains per-player suspicion estimates via exponential moving average (EMA) smoothing (α=0.3), while the DTR probe uses logit-lens to measure Jensen-Shannon divergence across transformer layers of a separate probe model (Qwen2.5-3B-Instruct) as a proxy for cognitive load. Results show that combining DBN with MaKTO-Proxy reasoning raises villager win rate from 44.2% to 68.8%, and adding DTR further improves vote accuracy (to 66.6%) but reduces survival, revealing a non-monotonic relationship between individual capability and collective payoff, interpreted through the lens of the handicap principle.

## Method

The framework consists of three modules combined across six configurations (A–F):

1. **MaKTO-Proxy**: few-shot chain-of-thought prompting to elicit reasoning.
2. **Dynamic Belief Network (DBN)**: per-player suspicion estimates updated each round via EMA smoothing with α=0.3, designed to dampen the positive-feedback loop between homogeneous models.
3. **Deep-Thinking Token Ratio (DTR) probe**: computes the average Jensen-Shannon divergence between layerwise logit-lens distributions of a separate probe model (Qwen2.5-3B-Instruct) to estimate cognitive load of utterances, relying on cross-architecture inference.

Experiments run 500 games per configuration in nine-player Werewolf with DeepSeek-V3.2 agents. Metrics include villager win rate, vote accuracy, mean survival rounds, and Brier Score convergence. Sensitivity analysis on α (0.1/0.3/0.5) is conducted under Configuration C.

## Key Claims & Evidence

- **Unassisted baseline achieves 44.2% villager win rate, lower than random.** **[supported]** — Table 2 confirms Group A win rate 44.2%; quote found verbatim.
- **Combining MaKTO-Proxy and DBN increases win rate to 68.8%.** **[supported]** — Table 2 confirms Group E win rate 68.8%, a ~24.6 pp increase over baseline.
```

---

## Project layout

```
paperflow/
├── paperflow/
│   ├── config.py            # .env loading, Settings (secrets stay out of git)
│   ├── ingest.py            # entry normalization: arxiv | doi | pdf | url | title
│   ├── pipeline.py          # one-call Pipeline: board + registry + team
│   ├── cli.py               # `paperflow run|serve|version`
│   ├── core/
│   │   ├── agent.py         # Agent base class (prompt / tools / artifacts)
│   │   ├── board.py         # JSON task board (atomic saves, agent states)
│   │   ├── llm.py           # DeepSeek client + explicit function-calling loop
│   │   ├── orchestrator.py  # dependency-aware scheduler, critical/optional
│   │   └── tools.py         # ToolSpec + ToolRegistry
│   ├── tools/               # arxiv_search, resolve_doi, fetch_url,
│   │                        # fetch_pdf_text, read_pdf, search_text
│   │                        # net.py: throttle + SSRF guard (per hop)
│   ├── agents/              # Researcher · Reader · Critic · Synthesizer
│   │                        # verification.py: machine ledger + report check
│   └── web/                 # FastAPI dashboard + static page
├── tests/                   # offline suite (fake LLM) + smoke tests
├── scripts/                 # offline evidence: verification recompute,
│                            # three-arm comparison, SSRF / ledger demos
└── examples/                # committed sample runs
```

---

## Limitations (known)

- **Single-pass reading**: the Reader processes the full text in one context
  window (default cap 60k chars). Very long papers are truncated; chunked /
  hierarchical reading is future work.
- **DeepSeek `deepseek-chat` only**: no reasoning-model or multi-provider
  abstraction yet (the `LLMClient` interface makes this straightforward).
  `usage` from the API is currently dropped, so a run reports **no token or
  cost accounting**.
- **Sequential orchestration**: agents run in a fixed pipeline order; a
  parallel / map-reduce mode (multiple Readers per section) is not built.
- **Abstract-only mode**: when no full text can be obtained (e.g. a DOI
  without an open-access PDF), the review is based on the abstract and every
  claim is explicitly marked *unverifiable* — it is never presented as
  supported.
- **What the quote verifier does not catch**: it tolerates PDF artifacts
  (line-break and in-word hyphens, page numbers extracted into a sentence,
  punctuation differences) and quotes with an elided middle. It does not
  catch a quote that was *paraphrased*, nor a claim whose number was changed
  while the surrounding wording matches, and the hyphen/punctuation
  normalization assumes the quote and the paper are the same language. The
  match tier is reported per claim (`quote_match_mode`) so a repaired match
  is never presented as a verbatim one.
- **Report ↔ verification linking**: the dashboard links a report bullet to
  its claim card by exact text containment first, then by claim-token overlap
  (≥60% of the claim's tokens, ≥4 tokens), so bullets the Synthesizer
  rephrased still link. Measured over the archived runs: **50/118 links with
  exact matching only (42%, and 0/7 on the `attention-is-all-you-need`
  sample) → 112/118 (95%)** with the overlap fallback. It is a navigation
  aid, not a verification: the machine ledger in the report is what carries
  the verdict.
- **Network dependence**: arXiv/Crossref/DeepSeek calls need internet;
  arXiv's public API is rate-limited (the client enforces a ≥3s interval).
  The offline suite never touches the network; the 3 smoke tests do.

---

## Credits

- [arXiv API](https://info.arxiv.org/help/api/index.html) — public metadata/PDF source (no key required)
- [Crossref REST API](https://api.crossref.org) — DOI → metadata resolution
- [DeepSeek API](https://platform.deepseek.com) — LLM + function calling
- [pypdf](https://pypi.org/project/pypdf/) — PDF text extraction
- [FastAPI](https://fastapi.tiangolo.com/) / [Uvicorn](https://www.uvicorn.org/) — web dashboard
- Sample: the author's own open-access paper (CC-BY 4.0), DOI `10.54254/2753-8818/2026.DL34010`

## License

MIT — see [LICENSE](LICENSE).
