"""Tests for paperflow.config: .env loading and Settings."""

from __future__ import annotations

from pathlib import Path

from paperflow.config import Settings, load_dotenv


def test_load_dotenv_parses_key_value(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n\nDEEPSEEK_API_KEY=sk-test-123\nPAPERFLOW_MODEL=deepseek-chat\n"
    )
    loaded = load_dotenv(env_file)
    assert loaded == env_file
    import os

    assert os.environ["DEEPSEEK_API_KEY"] == "sk-test-123"


def test_load_dotenv_ignores_comments_and_blank_lines(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("  \n# only a comment\n\n")
    loaded = load_dotenv(env_file)
    assert loaded == env_file


def test_load_dotenv_missing_file_returns_none(tmp_path: Path):
    assert load_dotenv(tmp_path / "nope.env") is None


def test_load_dotenv_does_not_override_existing_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "already-set")
    env_file = tmp_path / ".env"
    env_file.write_text("DEEPSEEK_API_KEY=sk-other\n")
    load_dotenv(env_file)
    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "already-set"


def test_load_dotenv_handles_utf8_bom(tmp_path: Path, monkeypatch):
    """Windows editors often save .env with a UTF-8 BOM on the first line."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\xef\xbb\xbfDEEPSEEK_API_KEY=sk-bom-key\n")
    load_dotenv(env_file)
    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "sk-bom-key"


def test_settings_reads_env_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-defaults")
    monkeypatch.delenv("PAPERFLOW_MODEL", raising=False)
    s = Settings()
    assert s.deepseek_api_key == "sk-defaults"
    assert s.model == "deepseek-chat"
    assert s.base_url == "https://api.deepseek.com/v1"
    assert s.max_fulltext_chars > 0
    assert s.max_tool_rounds >= 1


def test_settings_honors_optional_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.setenv("PAPERFLOW_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("PAPERFLOW_MAX_FULLTEXT_CHARS", "1000")
    s = Settings()
    assert s.model == "deepseek-reasoner"
    assert s.max_fulltext_chars == 1000
