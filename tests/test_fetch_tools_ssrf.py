"""SSRF containment for the fetch tools (P0).

An audit reproduced this against the running tool registry: a local service
(started on 127.0.0.1) could be read back through ``fetch_url`` and its body
landed in the agent's context - even though ``fetch_pdf_text`` / ``download_pdf``
already refused private hosts, and the README advertised the guard for "the
fetch tools". Redirects were a second hole: ``requests`` follows them by
default, so a public URL answering ``302 Location: http://169.254.169.254/``
reached the cloud metadata endpoint without any check.
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from pathlib import Path

import pytest
import requests

from paperflow.config import Settings
from paperflow.tools import build_default_registry, net

SETTINGS = Settings(deepseek_api_key="sk-test")
SECRET = "PAPERFLOW-SSRF-CANARY-DO-NOT-LEAK"


class _Handler(http.server.BaseHTTPRequestHandler):
    """A local service that would leak its body if it were reachable."""

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/secret")
            self.end_headers()
            return
        body = SECRET.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the test output
        return


@pytest.fixture
def local_service():
    """A real local HTTP server (offline): the SSRF target the audit used."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def registry(tmp_path: Path):
    return build_default_registry(SETTINGS, tmp_path / "cache")


# ---------------------------------------------------------------------------
# the hole itself
# ---------------------------------------------------------------------------

def test_local_service_really_is_reachable(local_service: str):
    """Control: the target is live, so the guard - not the network - must refuse."""
    assert SECRET in requests.get(f"{local_service}/secret", timeout=5).text


def test_fetch_url_refuses_a_loopback_service(registry, local_service: str):
    with pytest.raises(net.ToolNetError):
        registry.call("fetch_url", url=f"{local_service}/secret")


def test_fetch_url_refusal_is_visible_to_the_llm(registry, local_service: str):
    """The agent loop uses call_safe: the refusal must come back as TOOL ERROR."""
    result = registry.call_safe("fetch_url", url=f"{local_service}/secret")
    assert result.startswith("TOOL ERROR:")
    assert SECRET not in result


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/secret",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/secret",
        "http://[::1]/secret",
        "http://0.0.0.0/secret",
    ],
)
def test_fetch_url_refuses_private_and_link_local(registry, url: str):
    with pytest.raises(net.ToolNetError):
        registry.call("fetch_url", url=url)


@pytest.mark.parametrize("url", ["file:///C:/Windows/win.ini", "ftp://example.com/x", "gopher://example.com/"])
def test_fetch_url_refuses_non_http_schemes(registry, url: str):
    with pytest.raises(net.ToolNetError):
        registry.call("fetch_url", url=url)


# ---------------------------------------------------------------------------
# redirects must be validated per hop
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status: int, body: bytes = b"", location: str | None = None):
        self.status_code = status
        self.content = body
        self.headers = {"location": location} if location else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


def test_fetch_url_validates_every_redirect_hop(registry, monkeypatch):
    """A public URL answering 302 -> metadata must not be followed."""
    public = net._non_public_reason
    monkeypatch.setattr(
        net, "_non_public_reason", lambda host: None if host == "public.example" else public(host)
    )
    requested: list[str] = []

    def fake_get(url, **kwargs):
        requested.append(url)
        assert kwargs.get("allow_redirects") is False, "redirects must be followed manually"
        if "public.example" in url:
            return _FakeResponse(302, location="http://169.254.169.254/latest/meta-data/")
        return _FakeResponse(200, body=b"METADATA-LEAK")

    monkeypatch.setattr(net.requests, "get", fake_get)
    with pytest.raises(net.ToolNetError):
        registry.call("fetch_url", url="http://public.example/start")
    assert not any("169.254" in url for url in requested), requested


def test_fetch_url_validates_redirect_hops_to_loopback(registry, monkeypatch, local_service: str):
    public = net._non_public_reason
    monkeypatch.setattr(
        net, "_non_public_reason", lambda host: None if host == "public.example" else public(host)
    )
    requested: list[str] = []

    def fake_get(url, **kwargs):
        requested.append(url)
        if "public.example" in url:
            return _FakeResponse(302, location=f"{local_service}/secret")
        return _FakeResponse(200, body=SECRET.encode())

    monkeypatch.setattr(net.requests, "get", fake_get)
    with pytest.raises(net.ToolNetError):
        registry.call("fetch_url", url="http://public.example/start")
    assert all("public.example" in url for url in requested), requested


def test_download_pdf_also_validates_redirect_hops(monkeypatch, tmp_path: Path):
    public = net._non_public_reason
    monkeypatch.setattr(
        net, "_non_public_reason", lambda host: None if host == "public.example" else public(host)
    )

    def fake_get(url, **kwargs):
        return _FakeResponse(302, location="http://169.254.169.254/meta.pdf")

    monkeypatch.setattr(net.requests, "get", fake_get)
    with pytest.raises(net.ToolNetError):
        net.download_pdf("http://public.example/paper.pdf", tmp_path / "cache", timeout=5)


# ---------------------------------------------------------------------------
# the guard must not break legitimate fetches
# ---------------------------------------------------------------------------

def test_fetch_url_still_reads_a_public_page(registry, monkeypatch):
    monkeypatch.setattr(net, "_non_public_reason", lambda host: None)
    seen: dict = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["allow_redirects"] = kwargs.get("allow_redirects")
        return _FakeResponse(200, body=b"<html>landing page</html>")

    monkeypatch.setattr(net.requests, "get", fake_get)
    result = json.loads(json.dumps(registry.call("fetch_url", url="https://example.com/paper", max_chars=100)))
    assert "landing page" in result
    assert seen["url"] == "https://example.com/paper"
    assert seen["allow_redirects"] is False


def test_fetch_url_follows_a_public_redirect(registry, monkeypatch):
    """Redirects still work - they are just checked one hop at a time."""
    monkeypatch.setattr(net, "_non_public_reason", lambda host: None)

    def fake_get(url, **kwargs):
        if url.endswith("/old"):
            return _FakeResponse(301, location="https://example.com/new")
        return _FakeResponse(200, body=b"moved content")

    monkeypatch.setattr(net.requests, "get", fake_get)
    assert "moved content" in registry.call("fetch_url", url="https://example.com/old")


def test_fetch_url_stops_on_a_redirect_loop(registry, monkeypatch):
    """A redirect loop must terminate with an error instead of spinning."""
    monkeypatch.setattr(net, "_non_public_reason", lambda host: None)
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return _FakeResponse(302, location="https://example.com/loop")

    monkeypatch.setattr(net.requests, "get", fake_get)
    with pytest.raises(net.ToolNetError):
        registry.call("fetch_url", url="https://example.com/loop")
    assert calls["n"] <= 6
