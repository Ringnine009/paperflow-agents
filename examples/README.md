# Sample runs

Each folder contains a **committed sample run** produced by the real
pipeline (DeepSeek function calling + arXiv/Crossref APIs):

| Folder | Entry | Why it matters |
|---|---|---|
| [`werewolf-dbn/`](werewolf-dbn/report.md) | DOI `10.54254/2753-8818/2026.DL34010` + local PDF | the author's own published multi-agent paper — reviews **DBN dynamic belief networks** for social deduction, the direction this project echoes |
| [`attention-is-all-you-need/`](attention-is-all-you-need/report.md) | arXiv `1706.03762` | a classic LLM paper, exercising the full arXiv fetch path |

What is in each folder:

- `report.md` — the final structured review written by the Synthesizer
- `board.json` — the persisted task board (agent states, artifacts, log);
  the `artifacts[]` and `report` paths are **relative to this directory** so
  the sample is self-contained after cloning (the actual JSON/text artifacts
  live in `artifacts/` and are committed alongside)
- `artifacts/` — the run's stored outputs: `researcher_output.json`,
  `full_text.txt` (paper body), `reader_output.json` (claims with
  deterministic `quote_verified` annotations), `critic_output.json`
  (verdicts)

To reproduce a run yourself:

```bash
# werewolf paper (needs the PDF locally)
python -m paperflow run "10.54254/2753-8818/2026.DL34010" --pdf path/to/werewolf-paper.pdf

# attention paper
python -m paperflow run "https://arxiv.org/abs/1706.03762"
```
