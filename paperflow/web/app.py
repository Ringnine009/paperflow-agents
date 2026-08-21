"""FastAPI dashboard for PaperFlow.

Endpoints:
  GET  /                     dashboard page
  GET  /api/boards           summaries of every run (persisted board.json files)
  POST /api/runs             start a run  {entry, pdf_override?}
  GET  /api/runs/{id}        live board state for a run
  GET  /api/runs/{id}/report final review markdown

Runs execute in background threads; the board JSON file is the single source
of truth, so the dashboard survives restarts and shows live progress.
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from paperflow.config import get_settings
from paperflow.core.board import TaskBoard, make_run_id
from paperflow.pipeline import Pipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"


class RunRequest(BaseModel):
    entry: str
    pdf_override: str | None = None


class RunManager:
    """Tracks active runs; persisted runs are discovered from disk."""

    def __init__(self, out_dir: str | Path | None = None):
        self.out_dir = Path(out_dir) if out_dir else Path.cwd() / "outputs"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, threading.Thread] = {}

    # -- run discovery -----------------------------------------------------
    def boards(self) -> list[dict]:
        """Summaries of all runs, newest first."""
        summaries = []
        for board_file in sorted(self.out_dir.glob("*/board.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                board = TaskBoard.load(board_file)
            except Exception:  # noqa: BLE001 - skip corrupt boards
                continue
            summaries.append(
                {
                    "id": board.id,
                    "input": board.input.to_dict(),
                    "status": board.status,
                    "current_stage": board.current_stage,
                    "agents": board.agents,
                    "created_at": board.created_at,
                    "updated_at": board.updated_at,
                    "active": board.id in self._threads,
                }
            )
        return summaries

    def board(self, run_id: str) -> TaskBoard:
        path = self.out_dir / run_id / "board.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        return TaskBoard.load(path)

    def report(self, run_id: str) -> str:
        board = self.board(run_id)
        if not board.report or not Path(board.report["path"]).is_file():
            raise HTTPException(status_code=404, detail="no report yet for this run")
        return Path(board.report["path"]).read_text(encoding="utf-8")

    # -- execution ---------------------------------------------------------
    def start(self, request: RunRequest) -> dict:
        run_id = make_run_id()
        pipeline = Pipeline(settings=get_settings(), out_dir=self.out_dir)

        def _worker() -> None:
            pipeline.run(request.entry, pdf_override=request.pdf_override)

        thread = threading.Thread(target=_worker, name=f"paperflow-{run_id}", daemon=True)
        self._threads[run_id] = thread
        thread.start()
        return {"run_id": run_id}


def create_app(out_dir: str | Path | None = None) -> FastAPI:
    manager = RunManager(out_dir)
    app = FastAPI(title="PaperFlow Dashboard", version="0.1.0")
    app.state.manager = manager

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/boards")
    def list_boards() -> dict:
        return {"runs": manager.boards()}

    @app.post("/api/runs")
    def start_run(request: RunRequest) -> dict:
        return manager.start(request)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return manager.board(run_id).to_dict()

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str):
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(manager.report(run_id))

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
