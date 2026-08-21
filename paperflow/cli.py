"""Command-line interface.

Usage::

    paperflow run "https://arxiv.org/abs/1706.03762"          # arXiv entry
    paperflow run "10.54254/2753-8818/2026.DL34010"           # DOI entry
    paperflow run path/to/paper.pdf --pdf path/to/paper.pdf   # local PDF
    paperflow serve --port 8080                               # web dashboard
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from paperflow import __version__
from paperflow.config import get_settings, load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperflow",
        description="PaperFlow - multi-agent paper review (arXiv/DOI/PDF -> structured review).",
    )
    parser.add_argument("--env-file", default=None, help="path to a .env file with DEEPSEEK_API_KEY")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the agent team on one paper entry")
    # --env-file is accepted both before and after the subcommand (SUPPRESS
    # keeps the main parser's value when the subparser does not see it)
    run_p.add_argument("--env-file", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    run_p.add_argument("entry", help="arXiv URL / DOI / local PDF path / free-text title")
    run_p.add_argument("--pdf", default=None, help="local PDF to use as full text (for DOI/URL entries)")
    run_p.add_argument("--out", default=None, help="output directory (default ./outputs)")

    serve_p = sub.add_parser("serve", help="start the web dashboard")
    serve_p.add_argument("--env-file", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8080)
    serve_p.add_argument("--out", default=None, help="run output directory (default ./outputs)")

    sub.add_parser("version", help="print version")
    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from paperflow.pipeline import Pipeline

    settings = get_settings()
    pipeline = Pipeline(settings=settings, out_dir=args.out)
    # ASCII-only output: Windows consoles may use GBK and cannot encode ▶/✖/✔
    print(f"==> paperflow {__version__} - entry: {args.entry}")
    print(f"  run output will be written under: {pipeline.out_dir.resolve()}")
    try:
        result = pipeline.run(args.entry, pdf_override=args.pdf)
    except Exception as exc:  # noqa: BLE001 - CLI should surface failures cleanly
        print(f"[error] pipeline failed: {exc}", file=sys.stderr)
        return 1
    print(f"[ok] status: {result['status']}")
    print(f"  task board : {result['board']}")
    if result["report"]:
        print(f"  review     : {result['report']}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from paperflow.web.app import create_app

    app = create_app(out_dir=args.out)
    print(f"==> PaperFlow dashboard at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # secrets discipline: the API key only enters the process via env / .env
    load_dotenv(args.env_file)

    if args.command == "version":
        print(f"paperflow {__version__}")
        return 0
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "serve":
        return _cmd_serve(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
