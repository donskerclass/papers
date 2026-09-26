"""Render the ranked papers to a self-contained static site."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter

from jinja2 import Environment, PackageLoader, select_autoescape

from .paths import SITE

SECTIONS = [
    ("own", "Closest to your work", "nearest to one of your own papers"),
    ("favorites", "Closest to your favorites", "nearest to a paper you liked"),
    ("interests", "Matching your interests", "best match to an interest statement"),
    ("library", "From your reading", "most like your Zotero library as a whole"),
]


def _human(d: str) -> str:
    return dt.date.fromisoformat(d).strftime("%a %-d %b")


def build_sections(papers: list[dict], per_section: int) -> list[dict]:
    new = [p for p in papers if p["is_new"]]
    out = []
    for key, title, blurb in SECTIONS:
        if not any(key in p["ranks"] for p in papers):
            continue
        mine = sorted((p for p in new if p["best"] == key), key=lambda p: -p["ranks"][key])
        out.append({"key": key, "title": title, "blurb": blurb, "papers": mine[:per_section]})
    return out


def render(papers: list[dict], profile: dict, cfg: dict, library_meta: dict | None) -> None:
    latest = max(p["announced"] for p in papers)
    for p in papers:
        p["is_new"] = p["announced"] == latest
        p["announced_human"] = _human(p["announced"])
        # Reasons ordered by how highly the paper ranks on each criterion.
        p["why_order"] = sorted((c for c in p["why"] if c in p["ranks"]), key=lambda c: -p["ranks"][c])[:3]
    papers.sort(key=lambda p: -p["score"])
    for i, p in enumerate(papers, 1):
        p["rank"] = i
        p["surnames"] = [a.split()[-1] for a in p["authors"]]

    counts = Counter(p["announced"] for p in papers)
    now = dt.datetime.now(dt.timezone.utc)
    sc = cfg["scoring"]
    run = {
        "latest": latest,
        "latest_human": dt.date.fromisoformat(latest).strftime("%A %-d %B %Y"),
        "built_human": now.strftime("%-d %b %Y, %H:%M UTC"),
        "n_total": len(papers),
        "n_new": counts[latest],
        "dates": [(d, _human(d), counts[d]) for d in sorted(counts, reverse=True)],
        "categories": cfg["sources"]["arxiv"]["categories"],
        "window_days": cfg["sources"]["arxiv"]["window_days"],
        "model": cfg["embedding"]["model"],
        "weights": sc["weights"],
        "avoid_penalty": sc.get("avoid_penalty"),
        "knn_k": sc.get("knn_k", 10),
        "library_n": (library_meta or {}).get("n_library", "?"),
        "background_n": (library_meta or {}).get("n_background", "?"),
        "library_auc": (library_meta or {}).get("tfidf_heldout_auc_arxiv_only"),
    }
    env = Environment(loader=PackageLoader("paperboard", "templates"), autoescape=select_autoescape())
    ctx = {"site": cfg["site"], "run": run, "profile": profile}
    SITE.mkdir(parents=True, exist_ok=True)
    listed = papers[: cfg["page"].get("max_listed", 400)]
    sections = build_sections(papers, cfg["page"].get("per_section", 5))
    (SITE / "index.html").write_text(env.get_template("index.html").render(**ctx, sections=sections, listed=listed))
    (SITE / "about.html").write_text(env.get_template("about.html").render(**ctx))
    # Email version (sent by the workflow on scheduled runs) and its subject line.
    subject = f"Papers: arXiv listing of {dt.date.fromisoformat(latest):%a %-d %b} ({run['n_new']} new)"
    top = papers[: cfg.get("email", {}).get("top_n", 30)]
    (SITE / "email.html").write_text(env.get_template("email.html").render(
        **ctx, sections=sections, top=top, subject=subject))
    (SITE / "email_subject.txt").write_text(subject)
    # Machine-readable copy of the run, for later analysis or other front ends.
    keep = ("arxiv", "title", "authors", "categories", "primary", "announced", "score", "ranks", "raw", "why", "penalty")
    (SITE / "papers.json").write_text(json.dumps({"run": run, "papers": [{k: p.get(k) for k in keep} for p in papers]}))
    (SITE / ".nojekyll").write_text("")
