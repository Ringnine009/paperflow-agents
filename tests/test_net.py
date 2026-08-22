"""Tests for paperflow.tools.net: throttling, downloads, PDF extraction, retry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from paperflow.tools import net


def make_response(status: int, payload: bytes = b"ok"):
    class FakeResponse:
        status_code = status
        content = payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} error", response=self)

    return FakeResponse()


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
    dest1 = net.download_pdf("https://example.com/a.pdf", tmp_path / "cache", timeout=10)
    dest2 = net.download_pdf("https://example.com/a.pdf", tmp_path / "cache", timeout=10)
    assert dest1 == dest2
    assert called["n"] == 1  # second call served from cache
    assert dest1.read_bytes() == payload


def test_download_pdf_invalid_url():
    from paperflow.tools.net import ToolNetError

    with pytest.raises(ToolNetError):
        net.download_pdf("not-a-url", Path("."), timeout=5)


def test_http_get_bytes_retries_on_429_then_succeeds(monkeypatch):
    """arXiv rate limits surface as 429; the client must retry with backoff."""
    fake = FakeTime()
    monkeypatch.setattr(net, "time", fake)
    calls = {"n": 0}

    def fake_get(url, timeout, headers):
        calls["n"] += 1
        return make_response(429 if calls["n"] < 3 else 200)

    monkeypatch.setattr(net.requests, "get", fake_get)
    data = net.http_get_bytes("https://export.arxiv.org/api/query?x=1", timeout=5)
    assert data == b"ok"
    assert calls["n"] == 3
    assert fake.sleeps == [8.0, 16.0]  # backoff: 8s, 16s


def test_http_get_bytes_does_not_retry_client_errors(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(net, "time", fake)
    calls = {"n": 0}

    def fake_get(url, timeout, headers):
        calls["n"] += 1
        return make_response(404)

    monkeypatch.setattr(net.requests, "get", fake_get)
    with pytest.raises(requests.HTTPError):
        net.http_get_bytes("https://x.example/404", timeout=5)
    assert calls["n"] == 1  # 404 is not retryable
    assert fake.sleeps == []


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/admin",
        "http://127.0.0.1:8000/secret",
        "http://10.0.0.5/x",
        "http://192.168.1.1/x",
        "http://172.16.0.1/x",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://[::1]/x",
        "http://0.0.0.0/x",
    ],
)
def test_assert_http_url_blocks_private_and_link_local(url: str):
    """SSRF guard: fetch tools must never touch loopback/private/link-local."""
    with pytest.raises(net.ToolNetError):
        net._assert_http_url(url)


def test_assert_http_url_allows_public_hosts():
    net._assert_http_url("https://export.arxiv.org/api/query?id_list=1706.03762")
    net._assert_http_url("https://api.crossref.org/works/10.1234/x")
    net._assert_http_url("https://example.com/paper.pdf")
