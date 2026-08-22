"""Shared helpers for agents: reading artifacts off the task board."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paperflow.core.jsonutil import json_dumps


def load_artifact_json(board: Any, name: str) -> dict | None:
    """Read an agent's JSON output artifact from the board, if present."""
    path = artifact_path(board, name)
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_artifact_text(board: Any, name: str) -> str | None:
    """Read a text artifact from the board, if present."""
    path = artifact_path(board, name)
    if not path:
        return None
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def artifact_path(board: Any, name: str) -> str | None:
    entry = board.artifacts.get(name)
    return entry["path"] if entry else None


def compact_json(obj: Any) -> str:
    """Compact JSON for embedding in prompts."""
    return json_dumps(obj)[:4000]
