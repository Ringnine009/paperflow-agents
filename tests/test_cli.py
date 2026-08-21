"""CLI tests: argument handling and the version command."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from paperflow import __version__
from paperflow.cli import build_parser, main


def test_env_file_accepted_after_subcommand():
    """`paperflow run <entry> --env-file path` must parse (common real usage)."""
    parser = build_parser()
    args = parser.parse_args(
        ["run", "10.1234/abc", "--env-file", "C:\\secrets\\.env", "--out", "out"]
    )
    assert args.command == "run"
    assert args.env_file == "C:\\secrets\\.env"
    assert args.out == "out"


def test_env_file_accepted_before_subcommand():
    parser = build_parser()
    args = parser.parse_args(["--env-file", "C:\\secrets\\.env", "run", "10.1234/abc"])
    assert args.env_file == "C:\\secrets\\.env"


def test_version_command(capsys):
    code = main(["version"])
    out = capsys.readouterr().out
    assert code == 0
    assert __version__ in out


def test_cli_run_output_is_gbk_safe(tmp_path: Path, monkeypatch):
    """On Windows (GBK console) CLI output must not contain non-encodable glyphs."""
    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        out_dir = tmp_path

        def run(self, entry, pdf_override=None):
            return {
                "status": "done",
                "board": str(tmp_path / "board.json"),
                "report": str(tmp_path / "report.md"),
            }

    monkeypatch.setattr("paperflow.pipeline.Pipeline", FakePipeline)
    buf = io.BytesIO()
    writer = io.TextIOWrapper(buf, encoding="gbk")
    monkeypatch.setattr(sys, "stdout", writer)

    code = main(["run", "10.1234/abc"])
    writer.flush()
    assert code == 0
    out = buf.getvalue().decode("gbk")
    assert "done" in out
    assert "board.json" in out
