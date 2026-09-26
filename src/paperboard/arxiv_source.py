"""New arXiv submissions in the watched categories (primary or cross-listed).

One search per run over the whole window, paged. Each paper gets the date of
the daily listing it appeared in, computed from arXiv's schedule: submissions
up to 14:00 US Eastern on a weekday are announced that evening at 20:00, and
Thursday-to-Friday submissions are announced on Sunday evening. arXiv labels
each listing with the following day's date (Sunday evening's is "Monday"), and
so does this. Holidays are ignored, which at worst files a paper a day off.
"""

from __future__ import annotations

import datetime as dt
import time
from zoneinfo import ZoneInfo

import feedparser

from . import resolve

ET = ZoneInfo("America/New_York")
PAGE = 500


def listing_date(published: str) -> dt.date:
    t = dt.datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(ET)
    d = t.date()
    # First weekday cutoff (14:00 ET) at or after the submission time.
    while d.weekday() >= 5 or (d == t.date() and t.time() >= dt.time(14, 0)):
        d += dt.timedelta(days=1)
    # Announced that evening (a Friday cutoff on Sunday evening); arXiv dates the
    # listing by the next day, so Mon-Thu cutoffs list Tue-Fri and Friday's lists Monday.
    return d + dt.timedelta(days=3 if d.weekday() == 4 else 1)


def fetch(categories: list[str], window_days: int, log=print) -> list[dict]:
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=window_days + 1)
    cat_q = "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"
    q = f"{cat_q} AND submittedDate:[{start:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
    papers: dict[str, dict] = {}
    offset = 0
    while True:
        r = resolve._get(resolve.ARXIV_API, params={
            "search_query": q, "start": offset, "max_results": PAGE,
            "sortBy": "submittedDate", "sortOrder": "descending"}, tries=6, pause=5)
        if r is None:
            if offset == 0:
                raise RuntimeError("arXiv API unreachable")
            log(f"arXiv: stopped paging at {offset} after repeated errors")
            break
        feed = feedparser.parse(r.text)
        entries = [e for e in feed.entries if "title" in e and e.title != "Error"]
        for e in entries:
            rec = resolve.parse_arxiv_entry(e)
            rec["announced"] = listing_date(rec["published"]).isoformat()
            papers[rec["arxiv"]] = rec
        total = int(feed.feed.get("opensearch_totalresults", 0) or 0)
        offset += PAGE
        log(f"arXiv: {len(papers)} of {total}")
        if not entries or offset >= total:
            break
    return list(papers.values())
