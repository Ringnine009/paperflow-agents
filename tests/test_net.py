"""Tests for paperflow.tools.net: throttling, downloads, PDF extraction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperflow.tools import net


class FakeTime:
    """Injectable clock that records sleep calls."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_throttle_enforces_min_interval(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(net, "time", fake)
    monkeypatch.setattr(net, "_LAST_REQUEST", {})
    net.throttle("export.arxiv.org", min_interval=10.0)
    assert fake.sleeps == []  # first call never sleeps
    net.throttle("export.arxiv.org", min_interval=10.0)
    assert fake.sleeps == [10.0]  # second call waits the remainder
    # after waiting, a call made later than the interval does not sleep
    fake.now += 30
    net.throttle("export.arxiv.org", min_interval=10.0)
    assert fake.sleeps == [10.0]


def test_throttle_is_per_host(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(net, "time", fake)
    monkeypatch.setattr(net, "_LAST_REQUEST", {})
    net.throttle("host-a", min_interval=10.0)
    net.throttle("host-b", min_interval=10.0)
    assert fake.sleeps == []


def test_extract_pdf_text_roundtrip(tmp_path: Path, fixture_pdf: Path):
    text = net.extract_pdf_text(str(fixture_pdf), max_chars=500)
    assert "Social Deduction" in text


def test_extract_pdf_text_bad_file(tmp_path: Path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(net.PDFError):
        net.extract_pdf_text(str(bad), max_chars=500)


def test_download_pdf_uses_cache(tmp_path: Path, monkeypatch):
    payload = b"%PDF-1.4 cached-content"
    called = {"n": 0}

    def fake_get_bytes(url, timeout, headers=None):
        called["n"] += 1
        return payload

    monkeypatch.setattr(net, "http_get_bytes", fake_get_bytes)
    dest1 = net.download_pdf("https://x.example/a.pdf", tmp_path / "cache", timeout=10)
    dest2 = net.download_pdf("https://x.example/a.pdf", tmp_path / "cache", timeout=10)
    assert dest1 == dest2
    assert called["n"] == 1  # second call served from cache
    assert dest1.read_bytes() == payload


def test_download_pdf_invalid_url():
    from paperflow.tools.net import ToolNetError

    with pytest.raises(ToolNetError):
        net.download_pdf("not-a-url", Path("."), timeout=5)
