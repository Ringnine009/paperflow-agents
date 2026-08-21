"""Tests for paperflow.core.board: the JSON task board state machine."""

from __future__ import annotations

import json
from pathlib import Path

from paperflow.core.board import TaskBoard
from paperflow.ingest import parse_entry


def make_board(path: Path) -> TaskBoard:
    return TaskBoard.create(path, parse_entry("https://arxiv.org/abs/1706.03762"))


def test_create_and_persist(tmp_path: Path):
    path = tmp_path / "board.json"
    board = make_board(path)
    board.record_agent_start("researcher")
    board.save()

    assert path.exists()
    loaded = TaskBoard.load(path)
    assert loaded.id == board.id
    assert loaded.status == "running"
    assert loaded.input.kind == "arxiv"
    assert loaded.agents["researcher"]["status"] == "running"


def test_status_transitions(tmp_path: Path):
    path = tmp_path / "board.json"
    board = make_board(path)
    board.record_agent_start("researcher")
    board.record_agent_end("researcher", status="done", summary="found the paper")
    assert board.agents["researcher"]["status"] == "done"
    assert board.agents["researcher"]["summary"] == "found the paper"

    board.record_agent_start("reader")
    board.record_agent_end("reader", status="failed", error="LLM timeout")
    assert board.agents["reader"]["status"] == "failed"
    assert board.agents["reader"]["error"] == "LLM timeout"
    assert board.agents["reader"]["summary"] is None


def test_artifacts_and_report(tmp_path: Path):
    path = tmp_path / "board.json"
    board = make_board(path)
    artifact = tmp_path / "full_text.txt"
    artifact.write_text("hello")
    board.set_artifact("full_text", str(artifact), "paper body text")

    report = tmp_path / "report.md"
    report.write_text("# review")
    board.set_report(str(report), "# review")

    d = board.to_dict()
    assert d["artifacts"]["full_text"]["path"] == str(artifact)
    assert d["report"]["path"] == str(report)
    assert d["report"]["preview"] == "# review"


def test_log_timeline(tmp_path: Path):
    path = tmp_path / "board.json"
    board = make_board(path)
    board.add_log("pipeline started", agent=None)
    board.add_log("researcher running", agent="researcher")
    entries = board.log
    assert len(entries) == 2
    assert entries[0]["message"] == "pipeline started"
    assert entries[1]["agent"] == "researcher"
    assert all("ts" in e for e in entries)


def test_save_is_atomic_no_temp_leftovers(tmp_path: Path):
    path = tmp_path / "board.json"
    board = make_board(path)
    board.save()
    leftovers = [p.name for p in tmp_path.iterdir() if p != path]
    assert leftovers == []


def test_json_shape_is_serializable(tmp_path: Path):
    path = tmp_path / "board.json"
    board = make_board(path)
    board.record_agent_start("researcher")
    board.set_status("running", current_stage="researcher")
    board.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["status"] == "running"
    assert raw["current_stage"] == "researcher"
    assert raw["input"]["value"] == "1706.03762"
