"""Command line: `paperboard run` (daily, CI or laptop), `paperboard build-library` (laptop only)."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .paths import LOCAL


def log(msg: str) -> None:
    print(msg, flush=True)


def cmd_run(args) -> None:
    from . import arxiv_source, profile, render, scoring
    from .library_model import Library

    cfg = profile.config()
    prof = profile.load(resolve_missing=not args.offline, log=log)
    log(f"profile: {len(prof['own'])} own, {len(prof['favorites'])} favorites, "
        f"{len(prof['interests'])} interests, {len(prof['avoid'])} avoid")
    ax = cfg["sources"]["arxiv"]
    t0 = time.time()
    cache = LOCAL / "last_fetch.json"
    if args.reuse_fetch and cache.exists():
        papers = json.loads(cache.read_text())
        log(f"reusing {len(papers)} papers from {cache}")
    else:
        papers = arxiv_source.fetch(ax["categories"], ax["window_days"], log=log)
        log(f"fetched {len(papers)} papers in {time.time() - t0:.0f}s")
        cache.parent.mkdir(exist_ok=True)
        cache.write_text(json.dumps(papers))
    if not papers:
        sys.exit("no papers fetched")
    t0 = time.time()
    scoring.score(papers, prof, cfg, log=log)
    log(f"scored in {time.time() - t0:.0f}s")
    meta = Library().meta if Library.available() else None
    render.render(papers, prof, cfg, meta)
    log("site written")


def cmd_build_library(args) -> None:
    from . import library_build
    library_build.build(skip_export=args.skip_export, skip_background=args.skip_background, log=log)


def main() -> None:
    ap = argparse.ArgumentParser(prog="paperboard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fetch, score and render the site")
    r.add_argument("--offline", action="store_true", help="don't look up new profile entries")
    r.add_argument("--reuse-fetch", action="store_true", help="reuse the last arXiv fetch (for tuning)")
    r.set_defaults(func=cmd_run)
    b = sub.add_parser("build-library", help="(laptop) refit the reading-library model from Zotero")
    b.add_argument("--skip-export", action="store_true", help="reuse local/library.jsonl.gz")
    b.add_argument("--skip-background", action="store_true", help="reuse local/background.jsonl.gz")
    b.set_defaults(func=cmd_build_library)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
