"""Web dashboard regression tests (acceptance findings).

Two real defects found during acceptance:
  1. `serve --env-file` must apply the env file (key loading) — and any
     worker-thread failure (e.g. missing key) must surface on the task
     board as "failed" instead of leaving the run stuck at "pending".
  2. The run_id returned by POST /api/runs must match the run directory and
     board id, so the frontend can auto-follow a fresh run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paperflow.web.app import RunManager, RunRequest


class FakePipeline:
    """Stand-in that mimics Pipeline.run() but is fully offline."""

    instances: list[tuple] = []

    def __init__(self, *args, **kwargs):
        self.out_dir = Path(kwargs.get("out_dir"))

    def run(self, entry, pdf_override=None, run_id=None):
        from paperflow.core.board import TaskBoard
        from paperflow.ingest import parse_entry

        FakePipeline.instances.append((entry, pdf_override, run_id))
        run_id = run_id or "no-explicit-run-id"
        board = TaskBoard.create(
            self.out_dir / run_id / "board.json", parse_entry(entry), run_id=run_id
        )
        board.set_status("done")
        board.save()
        return {"run_id": run_id, "status": "done", "board": str(board.path), "report": None}


class ExplodingPipeline:
    """Pipeline whose run() always fails, e.g. when the API key is missing."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")


def _join_thread(manager: RunManager, run_id: str) -> None:
    thread = manager._threads[run_id]
    thread.join(timeout=15)
    assert not thread.is_alive(), "worker thread did not finish"


def test_web_start_run_id_matches_directory(tmp_path: Path, monkeypatch):
    """POST /api/runs must return the id the pipeline actually uses."""
    monkeypatch.setattr("paperflow.web.app.Pipeline", FakePipeline)
    manager = RunManager(tmp_path / "out")

    response = manager.start(RunRequest(entry="https://arxiv.org/abs/1706.03762"))
    run_id = response["run_id"]
    _join_thread(manager, run_id)

    # pipeline received the SAME id the manager generated
    entry, _pdf, used_run_id = FakePipeline.instances[-1]
    assert used_run_id == run_id
    # board lives at <out>/<run_id>/board.json and carries that id
    board_path = tmp_path / "out" / run_id / "board.json"
    assert board_path.is_file(), f"board not found at {board_path}"
    board = json.loads(board_path.read_text(encoding="utf-8"))
    assert board["id"] == run_id
    # the dashboard resolves the run by that id
    assert manager.board(run_id).status == "done"


def test_worker_failure_marks_board_failed(tmp_path: Path, monkeypatch):
    """A crashed worker (e.g. missing API key) must not leave a silent pending run."""
    monkeypatch.setattr("paperflow.web.app.Pipeline", ExplodingPipeline)
    manager = RunManager(tmp_path / "out")

    response = manager.start(RunRequest(entry="https://arxiv.org/abs/1706.03762"))
    run_id = response["run_id"]
    _join_thread(manager, run_id)

    board = manager.board(run_id)
    assert board.status == "failed"
    assert any("DEEPSEEK_API_KEY" in entry["message"] for entry in board.log)


def test_serve_applies_env_file(tmp_path: Path, monkeypatch):
    """`paperflow serve --env-file X` must load X into the process environment."""
    from paperflow.cli import main

    captured: dict = {}
    env_file = tmp_path / "env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-serve-test-key\n", encoding="utf-8")

    def fake_serve(args) -> int:
        captured["env_file"] = args.env_file
        captured["key"] = os.environ.get("DEEPSEEK_API_KEY")
        return 0

    monkeypatch.setattr("paperflow.cli._cmd_serve", fake_serve)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    code = main(["serve", "--env-file", str(env_file), "--port", "0"])
    assert code == 0
    assert captured["env_file"] == str(env_file)
    assert captured["key"] == "sk-serve-test-key"


def test_orphan_running_marked_interrupted(tmp_path: Path):
    """A run left 'running' by a dead process must read as 'interrupted'.

    Simulates a server restart: a board.json on disk says running, but no
    worker thread exists for it.
    """
    from paperflow.core.board import TaskBoard, make_run_id
    from paperflow.ingest import parse_entry

    manager = RunManager(tmp_path / "out")
    run_id = make_run_id()
    board = TaskBoard.create(
        tmp_path / "out" / run_id / "board.json",
        parse_entry("https://arxiv.org/abs/1706.03762"),
        run_id=run_id,
    )
    board.set_status("running", current_stage="reader")
    board.save()

    runs = manager.boards()
    assert runs[0]["id"] == run_id
    assert runs[0]["status"] == "interrupted"
    assert runs[0]["active"] is False

    # the detail endpoint agrees (read-time marking, not persisted)
    detail = manager.board(run_id)
    assert detail.status == "interrupted"
