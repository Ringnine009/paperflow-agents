"""Task board: the shared JSON state file the whole team writes to.

The board is the single source of truth for a run. The orchestrator updates
it before/after every agent so that the CLI and the web dashboard can show
live progress, and so a crashed run can be inspected afterwards.

Layout (all timestamps are UTC ISO-8601)::

    {
      "id": "pf-20260815-103000-a1b2c3",
      "input": {"raw": ..., "kind": "arxiv", "value": ..., "pdf_override": null},
      "created_at": ..., "updated_at": ...,
      "status": "running",              # pending | running | done | failed
      "current_stage": "reader",
      "log": [{"ts": ..., "level": "info", "agent": null, "message": ...}],
      "agents": {
        "researcher": {"status": "done", "started_at": ..., "finished_at": ...,
                       "summary": ..., "error": null}
      },
      "artifacts": {
        "full_text": {"path": "run/artifacts/full_text.txt", "description": ...}
      },
      "report": {"path": "run/reports/review.md", "preview": "# ..."}
    }
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperflow.ingest import InputSpec


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"pf-{stamp}-{uuid.uuid4().hex[:6]}"


class TaskBoard:
    """Persistent task board backed by a JSON file."""

    def __init__(self, path: str | Path, input_spec: InputSpec, run_id: str | None = None, created_at: str | None = None):
        self.path = Path(path)
        self.id = run_id or make_run_id()
        self.input = input_spec
        self.created_at = created_at or _utcnow()
        self.updated_at = self.created_at
        self.status = "pending"
        self.current_stage: str | None = None
        self.log: list[dict] = []
        self.agents: dict[str, dict] = {}
        self.artifacts: dict[str, dict] = {}
        self.report: dict | None = None

    # -- constructors -----------------------------------------------------
    @classmethod
    def create(cls, path: str | Path, input_spec: InputSpec) -> "TaskBoard":
        """Create a fresh board (call :meth:`save` to persist it)."""
        return cls(path=path, input_spec=input_spec)

    @classmethod
    def load(cls, path: str | Path) -> "TaskBoard":
        """Rehydrate a board from its JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        board = cls(
            path=path,
            input_spec=InputSpec.from_dict(data["input"]),
            run_id=data["id"],
            created_at=data["created_at"],
        )
        board.updated_at = data.get("updated_at", board.created_at)
        board.status = data.get("status", "pending")
        board.current_stage = data.get("current_stage")
        board.log = data.get("log", [])
        board.agents = data.get("agents", {})
        board.artifacts = data.get("artifacts", {})
        board.report = data.get("report")
        return board

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        """Atomically persist the board (write temp file, then rename)."""
        self.updated_at = _utcnow()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input": self.input.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "current_stage": self.current_stage,
            "log": self.log,
            "agents": self.agents,
            "artifacts": self.artifacts,
            "report": self.report,
        }

    # -- mutations --------------------------------------------------------
    def set_status(self, status: str, current_stage: str | None = None) -> None:
        self.status = status
        if current_stage is not None:
            self.current_stage = current_stage

    def add_log(self, message: str, level: str = "info", agent: str | None = None) -> None:
        self.log.append({"ts": _utcnow(), "level": level, "agent": agent, "message": message})

    def record_agent_start(self, name: str) -> None:
        self.agents[name] = {
            "status": "running",
            "started_at": _utcnow(),
            "finished_at": None,
            "summary": None,
            "error": None,
        }
        self.set_status("running", current_stage=name)
        self.add_log(f"agent '{name}' started", agent=name)

    def record_agent_end(self, name: str, status: str = "done", summary: str | None = None, error: str | None = None) -> None:
        entry = self.agents.setdefault(name, {"started_at": _utcnow()})
        entry.update({"status": status, "finished_at": _utcnow(), "summary": summary, "error": error})
        self.add_log(f"agent '{name}' finished: {status}", level="error" if status != "done" else "info", agent=name)

    def set_artifact(self, name: str, path: str | Path, description: str = "") -> None:
        self.artifacts[name] = {"path": str(path), "description": description}
        self.add_log(f"artifact '{name}' saved: {Path(path).name}", agent="orchestrator")

    def set_report(self, path: str | Path, markdown: str) -> None:
        self.report = {"path": str(path), "preview": markdown[:400]}
        self.add_log(f"final report written: {Path(path).name}", agent="orchestrator")

    def artifacts_dir(self) -> Path:
        return self.path.parent / "artifacts"

    def report_path(self) -> Path | None:
        return Path(self.report["path"]) if self.report else None
