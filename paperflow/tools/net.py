"""Networking helpers shared by the fetch tools.

* :func:`throttle` - per-host minimum interval between requests. arXiv's
  API guidelines ask for >= 3s between calls to export.arxiv.org, and we
  honor that globally so heavy use cannot hammer the service.
* :func:`download_pdf` / :func:`extract_pdf_text` - fetch a PDF once into a
  content-addressed cache and pull its text with pypdf.
* :func:`_assert_http_url` - SSRF guard: the fetch tools refuse any URL
  whose host is localhost, loopback, private, link-local (incl. the
  ``169.254.169.254`` cloud-metadata address) or otherwise non-public.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from pypdf import PdfReader

USER_AGENT = "PaperFlow/0.1 (portfolio research assistant; arXiv/Crossref API user)"


class ToolNetError(Exception):
    """A network-tool failure that should be surfaced to the LLM."""


class PDFError(ToolNetError):
    """A PDF could not be downloaded or parsed."""


#: last request timestamp per host, used by :func:`throttle`
_LAST_REQUEST: dict[str, float] = {}


def throttle(host: str, min_interval: float = 3.0) -> None:
    """Sleep so that consecutive requests to `host` are >= `min_interval` apart."""
    now = time.time()
    wait = _LAST_REQUEST.get(host, 0.0) + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST[host] = time.time()


#: statuses worth retrying - arXiv rate limits surface as 429, and 5xx are
#: transient server errors; everything else (404, 403, ...) is final
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def http_get_bytes(
    url: str,
    timeout: int = 30,
    headers: dict | None = None,
    max_retries: int = 3,
    retry_delay: float = 8.0,
) -> bytes:
    """GET a URL and return raw bytes, retrying rate limits / 5xx with backoff.

    arXiv's public API blocks IPs that burst too many requests (HTTP 429);
    waiting out the window with exponential backoff is the polite recovery.
    """
    response = None
    for attempt in range(max_retries + 1):
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT, **(headers or {})}
        )
        if response.status_code not in RETRYABLE_STATUS or attempt >= max_retries:
            break
        time.sleep(retry_delay * (attempt + 1))
    response.raise_for_status()
    return response.content


def http_get(url: str, timeout: int = 30, headers: dict | None = None) -> str:
    """GET a URL and return decoded text."""
    return http_get_bytes(url, timeout=timeout, headers=headers).decode("utf-8", errors="replace")


def _non_public_reason(host: str) -> str | None:
    """Return why `host` is unsafe to fetch, or None if it is public.

    Resolves the host and rejects it if ANY resolved address is loopback,
    private, link-local (covers the 169.254.169.254 metadata endpoint),
    reserved, multicast or unspecified.
    """
    lowered = host.lower().rstrip(".")
    if lowered == "localhost":
        return "localhost"
    try:
        infos = socket.getaddrinfo(lowered, None)
    except socket.gaierror:
        return f"unresolvable host: {host!r}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return f"{host!r} resolves to non-public address {ip}"
    return None


def _assert_http_url(url: str) -> None:
    """Reject non-http(s) URLs and any URL targeting a non-public host.

    SSRF guard: the web dashboard accepts user-supplied entries, so the
    fetch tools must never be tricked into hitting localhost, the local
    network or cloud metadata endpoints (169.254.169.254).
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ToolNetError(f"not an http(s) URL: {url!r}")
    host = (urlparse(url).hostname or "").strip()
    if not host:
        raise ToolNetError(f"URL has no host: {url!r}")
    reason = _non_public_reason(host)
    if reason:
        raise ToolNetError(
            f"blocked host ({reason}) - SSRF guard rejects private/loopback/link-local addresses"
        )


def download_pdf(url: str, cache_dir: str | Path, timeout: int = 60) -> Path:
    """Download a PDF into a content-addressed cache; reuse on repeat calls."""
    _assert_http_url(url)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    dest = cache / f"{digest}.pdf"
    if dest.exists():
        return dest
    dest.write_bytes(http_get_bytes(url, timeout=timeout))
    return dest


def extract_pdf_text(path: str | Path, max_chars: int = 60000) -> str:
    """Extract text from a local PDF with pypdf, truncated to `max_chars`."""
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pypdf raises assorted exceptions on bad files
        raise PDFError(f"cannot parse PDF {Path(path).name}: {exc}") from exc
    text = "\n".join(pages).replace("\x00", "")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:max_chars]
