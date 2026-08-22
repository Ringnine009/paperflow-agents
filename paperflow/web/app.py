"""FastAPI dashboard for PaperFlow.

Endpoints:
  GET  /                           dashboard page
  GET  /api/boards                 summaries of every run (persisted board.json files)
  POST /api/runs                   start a run  {entry, pdf_override?}
  GET  /api/runs/{id}              live board state for a run
  GET  /api/runs/{id}/report       final review markdown
  GET  /api/runs/{id}/verification deterministic quote-verification detail

Runs execute in background threads; the board JSON file is the single source
of truth, so the dashboard survives restarts and shows live progress.

Failure discipline: the board is created *before* the worker starts (so the
run is visible immediately), and ANY worker failure - missing API key,
network error, LLM failure - is recorded on the board as ``failed`` with the
error message. A run can never silently stall at "pending".
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from paperflow.config import get_settings
from paperflow.core.board import TaskBoard, make_run_id
from paperflow.ingest import parse_entry
from paperflow.pipeline import Pipeline

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def align_claims(reader_output: dict | None, critic_output: dict | None) -> dict:
    """Merge reader claims with critic verdicts for the Verification view.

    Reader claims are the rows; critic verdicts are joined by ``claim_index``
    (mandatory in the critic's output contract) with a fallback to exact
    claim wording for older artifacts. Every row carries the deterministic
    quote-verification annotations (``quote_verified`` / reason / loc /
    context) plus the critic's verdict and note.
    """
    claims = (reader_output or {}).get("claims", []) or []
    verdicts = (critic_output or {}).get("verdicts", []) or []

    by_index: dict[int, dict] = {}
    by_text: dict[str, dict] = {}
    for verdict in verdicts:
        index = verdict.get("claim_index")
        if isinstance(index, str) and index.strip().isdigit():
            index = int(index)
        if isinstance(index, int):
            by_index[index] = verdict
        else:
            by_text[str(verdict.get("claim", "")).strip().lower()] = verdict

    rows: list[dict[str, Any]] = []
    verified = 0
    for index, claim in enumerate(claims):
        verdict = by_index.get(index) or by_text.get(str(claim.get("claim", "")).strip().lower())
        quote_verified = claim.get("quote_verified")
        if quote_verified is True:
            verified += 1
        rows.append(
            {
                "claim_index": index,
                "claim": claim.get("claim", ""),
                "quote": claim.get("quote", ""),
                "quote_verified": quote_verified,
                "quote_verification": claim.get("quote_verification"),
                "quote_loc": claim.get("quote_loc"),
                "quote_context": claim.get("quote_context"),
                "verdict": (verdict or {}).get("verdict"),
                "verdict_note": (verdict or {}).get("note"),
            }
        )
    return {"total_count": len(rows), "verified_count": verified, "claims": rows}


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
    def _effective_status(self, run_id: str, board: TaskBoard) -> str:
        """Read-time status resolution.

        A run whose board says "running" but has no live worker thread (the
        server was restarted mid-run) is reported as "interrupted" instead of
        spinning forever. Marking is read-time only; the file is untouched.
        """
        if board.status == "running" and run_id not in self._threads:
            return "interrupted"
        return board.status

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
                    "status": self._effective_status(board.id, board),
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
        board = TaskBoard.load(path)
        board.status = self._effective_status(run_id, board)
        return board

    def report(self, run_id: str) -> str:
        board = self.board(run_id)
        if not board.report or not Path(board.report["path"]).is_file():
            raise HTTPException(status_code=404, detail="no report yet for this run")
        return Path(board.report["path"]).read_text(encoding="utf-8")

    def verification(self, run_id: str) -> dict:
        """Deterministic quote-verification detail for the Verification view."""
        board = self.board(run_id)
        reader_output = self._artifact_json(board, "reader_output")
        critic_output = self._artifact_json(board, "critic_output")
        payload = align_claims(reader_output, critic_output)
        payload["run_id"] = run_id
        return payload

    @staticmethod
    def _artifact_json(board: TaskBoard, name: str) -> dict | None:
        entry = board.artifacts.get(name)
        if not entry or not Path(entry["path"]).is_file():
            return None
        try:
            return json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # -- execution ---------------------------------------------------------
    def start(self, request: RunRequest) -> dict:
        """Create the run (visible immediately) and launch the worker."""
        run_id = make_run_id()
        try:
            spec = parse_entry(request.entry, pdf_override=request.pdf_override)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # pre-create the board so the run is visible at once, even if the
        # worker dies before Pipeline.run() gets to create its own
        TaskBoard.create(self.out_dir / run_id / "board.json", spec, run_id=run_id).save()

        thread = threading.Thread(
            target=self._worker, args=(run_id, request), name=f"paperflow-{run_id}", daemon=True
        )
        self._threads[run_id] = thread
        thread.start()
        return {"run_id": run_id}

    def _worker(self, run_id: str, request: RunRequest) -> None:
        """Background runner: Pipeline is built here so a missing API key or
        any other startup error becomes a visible board failure, not a 500."""
        try:
            pipeline = Pipeline(settings=get_settings(), out_dir=self.out_dir)
            pipeline.run(request.entry, pdf_override=request.pdf_override, run_id=run_id)
        except Exception as exc:  # noqa: BLE001 - record every worker failure
            logger.error("run %s failed: %s", run_id, exc)
            try:
                board = TaskBoard.load(self.out_dir / run_id / "board.json")
                board.set_status("failed", current_stage="pipeline")
                board.add_log(f"pipeline failed: {type(exc).__name__}: {exc}", level="error")
                board.save()
            except Exception:  # noqa: BLE001 - the failure state itself must not crash
                logger.exception("could not persist failure state for run %s", run_id)
        finally:
            self._threads.pop(run_id, None)


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

    @app.get("/api/runs/{run_id}/verification")
    def get_verification(run_id: str) -> dict:
        return manager.verification(run_id)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app
