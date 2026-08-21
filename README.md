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
- **Full text never round-trips through the LLM** — PDF text is extracted
  to a content-addressed file and only *paths + previews* go into the
  conversation, keeping token cost low and quoting verifiable.
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

cp .env.example .env                 # put your DEEPSEEK_API_KEY in .env
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

### Tests

```bash
python -m pytest                      # offline suite (no network, fake LLM)
python -m pytest -m smoke             # real arXiv / Crossref API smoke tests
```

---

## Example output (excerpt)

Running the **author's own werewolf paper** (DOI `10.54254/2753-8818/2026.DL34010`)
produces a report like this (full sample under `examples/`):

```markdown
# Dynamic Belief Networks and Deep-Thinking Probes for Multi-agent Social Reasoning

> Zhenxiao Guo · Tongji University · 2026-06 · DOI: 10.54254/2753-8818/2026.DL34010

## Overview
...

## Key Claims & Evidence
- DBN + prompted reasoning raises villager win rate 44.2% → 68.8%
  **[supported]** (quote verified in full text: "...44.2% without beliefs...")

## Related Work
- [The Rise and Potential of Large Language Model Based Agents](https://arxiv.org/abs/2309.07864)
- [Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924)
- ...

## Relevance to My Research Direction
**9/10** — directly aligned with multi-agent social reasoning ...
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
│   ├── agents/              # Researcher · Reader · Critic · Synthesizer
│   └── web/                 # FastAPI dashboard + static page
├── tests/                   # offline suite (fake LLM) + smoke tests
└── examples/                # committed sample runs
```

---

## Limitations (known)

- **Single-pass reading**: the Reader processes the full text in one context
  window (default cap 60k chars). Very long papers are truncated; chunked /
  hierarchical reading is future work.
- **DeepSeek `deepseek-chat` only**: no reasoning-model or multi-provider
  abstraction yet (the `LLMClient` interface makes this straightforward).
- **Sequential orchestration**: agents run in a fixed pipeline order; a
  parallel / map-reduce mode (multiple Readers per section) is not built.
- **Abstract-only mode**: when no full text can be obtained (e.g. a DOI
  without an open-access PDF), the review is based on the abstract and the
  Critic marks claims *unverifiable*.
- **Network dependence**: arXiv/Crossref/DeepSeek calls need internet;
  arXiv's public API is rate-limited (the client enforces a ≥3s interval).

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
