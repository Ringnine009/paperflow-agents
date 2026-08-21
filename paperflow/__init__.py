"""PaperFlow - a lightweight multi-agent research collaboration system.

Drop a paper link / DOI / PDF into the pipeline and a small team of agents
(Researcher, Reader, Critic, Synthesizer) collaborates through a JSON task
board to produce a structured review plus related-work links.

Highlights:
  * self-built orchestration (no LangChain): Agent base class, tool registry,
    JSON task board, and a deterministic orchestrator
  * DeepSeek function calling (deepseek-chat) with an explicit tool loop
  * three entry points: arXiv URL / DOI / local PDF path
  * CLI pipeline + lightweight FastAPI dashboard
"""

__version__ = "0.1.0"
