"""New arXiv submissions in the watched categories (primary or cross-listed).

Harvested through arXiv's OAI-PMH interface (oaipmh.arxiv.org), not the search
API: the search API refuses requests from cloud machines such as GitHub's
runners (HTTP 406), while OAI-PMH is arXiv's supported bulk-metadata route.
Each watched category is an OAI set (``stat.ML`` -> ``stat:stat:ML``) that
includes cross-lists. Records changed since the start of the window are
fetched, and only papers whose first version falls in the window are kept, so
replacements of old papers are skipped.

Each paper gets the date of the daily listing it appeared in, computed from
arXiv's schedule: submissions up to 14:00 US Eastern on a weekday are
announced that evening at 20:00, and Thursday-to-Friday submissions are
announced on Sunday evening. arXiv labels each listing with the following
day's date (Sunday evening's is "Monday"), and so does this. Holidays are
ignored, which at worst files a paper a day off.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import re
import unicodedata
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from . import resolve

EASTERN = ZoneInfo("America/New_York")
OAI = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}


def listing_date(published: str) -> dt.date:
    t = dt.datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(EASTERN)
    d = t.date()
    # First weekday cutoff (14:00 ET) at or after the submission time.
    while d.weekday() >= 5 or (d == t.date() and t.time() >= dt.time(14, 0)):
        d += dt.timedelta(days=1)
    # Announced that evening (a Friday cutoff on Sunday evening); arXiv dates the
    # listing by the next day, so Mon-Thu cutoffs list Tue-Fri and Friday's lists Monday.
    return d + dt.timedelta(days=3 if d.weekday() == 4 else 1)


# --- TeX accents in arXivRaw author lists and titles -> Unicode ---------------

_ACCENT = {"'": "́", "`": "̀", "^": "̂", '"': "̈", "~": "̃",
           "=": "̄", ".": "̇", "u": "̆", "v": "̌", "H": "̋",
           "c": "̧", "k": "̨", "r": "̊"}
_SPECIAL = {"o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "ss": "ß", "ae": "æ", "AE": "Æ",
            "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å", "i": "ı"}


def detex(s: str) -> str:
    def acc(m):
        return unicodedata.normalize("NFC", m.group(2) + _ACCENT[m.group(1)])
    s = re.sub(r"\{?\\([`'^\"~=.])\s*\{?\\?([A-Za-z])\}?\}?", acc, s)
    s = re.sub(r"\{?\\([uvHckr])\s*\{\\?([A-Za-z])\}\}?", acc, s)
    s = re.sub(r"\\([uvHckr]) ([A-Za-z])", acc, s)
    s = re.sub(r"\{\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L|i)\}", lambda m: _SPECIAL[m.group(1)], s)
    s = re.sub(r"\\(ss|ae|AE|oe|OE|aa|AA|o|O|l|L|i)(?![A-Za-z])\s?", lambda m: _SPECIAL[m.group(1)], s)
    return re.sub(r"\s+", " ", s).strip()


def _authors(raw: str) -> list[str]:
    raw = re.sub(r"\([^()]*\)", "", detex(raw))         # drop affiliations in parentheses
    raw = re.sub(r"\{([^{}\\$]*)\}", r"\1", raw)          # leftover grouping braces
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", raw)
    return [p.strip() for p in parts if p.strip()]


def _record(rec: ET.Element) -> dict | None:
    m = rec.find("oai:metadata/raw:arXivRaw", NS)
    if m is None:
        return None
    versions = m.findall("raw:version", NS)
    if not versions:
        return None
    first = email.utils.parsedate_to_datetime(versions[0].findtext("raw:date", "", NS))
    last = email.utils.parsedate_to_datetime(versions[-1].findtext("raw:date", "", NS))
    aid = m.findtext("raw:id", "", NS)
    cats = m.findtext("raw:categories", "", NS).split()
    return {
        "id": f"arxiv:{aid}",
        "arxiv": aid,
        "version": len(versions),
        "title": detex(re.sub(r"\s+", " ", m.findtext("raw:title", "", NS))),
        "abstract": re.sub(r"\s+", " ", m.findtext("raw:abstract", "", NS)).strip(),
        "authors": _authors(m.findtext("raw:authors", "", NS)),
        "categories": cats,
        "primary": cats[0] if cats else "",
        "published": first.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "updated": last.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "url": f"https://arxiv.org/abs/{aid}",
        "doi": m.findtext("raw:doi", "", NS),
        "source": "arxiv",
    }


def get_records(ids: list[str]) -> dict[str, dict]:
    """Look up individual papers by arXiv id (OAI GetRecord; works from CI)."""
    out = {}
    for aid in ids:
        r = resolve._get(OAI, params={"verb": "GetRecord", "metadataPrefix": "arXivRaw",
                                      "identifier": f"oai:arXiv.org:{aid}"}, tries=3, pause=10)
        if r is None:
            continue
        rec = ET.fromstring(r.content if hasattr(r, "content") else r.text.encode()).find(
            "oai:GetRecord/oai:record", NS)
        if rec is not None and (p := _record(rec)):
            out[aid] = p
    return out


def _oai_set(category: str) -> str:
    archive, _, sub = category.partition(".")
    return f"{archive}:{archive}:{sub}" if sub else archive


def fetch(categories: list[str], window_days: int, log=print) -> list[dict]:
    now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=window_days + 1)
    papers: dict[str, dict] = {}
    for cat in categories:
        params = {"verb": "ListRecords", "metadataPrefix": "arXivRaw",
                  "set": _oai_set(cat), "from": f"{start:%Y-%m-%d}"}
        n_seen = 0
        while True:
            r = resolve._get(OAI, params=params, tries=5, pause=10)
            if r is None:
                raise RuntimeError(f"arXiv OAI-PMH unreachable ({cat})")
            root = ET.fromstring(r.content if hasattr(r, "content") else r.text.encode())
            err = root.find("oai:error", NS)
            if err is not None:
                if err.get("code") != "noRecordsMatch":
                    log(f"arXiv OAI {cat}: {err.get('code')}: {err.text}")
                break
            lr = root.find("oai:ListRecords", NS)
            for rec in lr.findall("oai:record", NS):
                p = _record(rec)
                n_seen += 1
                if p and dt.datetime.fromisoformat(p["published"].replace("Z", "+00:00")) >= start:
                    p["announced"] = listing_date(p["published"]).isoformat()
                    papers[p["arxiv"]] = p
            token = (lr.findtext("oai:resumptionToken", "", NS) or "").strip()
            if not token:
                break
            params = {"verb": "ListRecords", "resumptionToken": token}
        log(f"arXiv {cat}: {n_seen} records changed since {start:%Y-%m-%d}; {len(papers)} new papers so far")
    return list(papers.values())
