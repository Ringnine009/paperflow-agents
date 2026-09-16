"""Live reproduction of the fetch_url SSRF hole and its fix (docs evidence).

Starts a local HTTP server holding a canary string and shows, with real
requests:

1. the server really is reachable (control),
2. ``net.http_get`` - the call ``fetch_url`` used before the fix - reads the
   canary back into the process (the old behaviour, still reproducible),
3. ``registry.call("fetch_url", ...)`` now refuses it, and
4. the refusal the agent sees (``call_safe`` turns it into ``TOOL ERROR``).

Offline: everything happens on 127.0.0.1.
"""

from __future__ import annotations

import http.server
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from paperflow.config import Settings  # noqa: E402
from paperflow.tools import build_default_registry  # noqa: E402
from paperflow.tools import net  # noqa: E402

CANARY = "INTERNAL-SECRET-CANARY"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server API
        body = f"{CANARY} (internal admin page)".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def main() -> int:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/admin"
    registry = build_default_registry(Settings(deepseek_api_key="sk-test"), "outputs/_cache")

    print(f"target: {url}")
    print("1. control - the service is live:",
          CANARY in requests.get(url, timeout=5).text)
    print("2. pre-fix call path (net.http_get, what fetch_url used):",
          repr(net.http_get(url, timeout=5)[:40]))
    try:
        registry.call("fetch_url", url=url)
        print("3. fetch_url: STILL LEAKS")
    except net.ToolNetError as exc:
        print(f"3. fetch_url: refused -> {exc}")
    print("4. what the agent sees:", registry.call_safe("fetch_url", url=url)[:90])

    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
