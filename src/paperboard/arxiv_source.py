"""New arXiv submissions in the watched categories (primary or cross-listed).

Harvested through arXiv's OAI-PMH interface (oaipmh.arxiv.org), not the search
API: the search API refuses requests from cloud machines such as GitHub's
runners (HTTP 406), while OAI-PMH is arXiv's supported bulk-metadata route.
Each watched category is an OAI set (``stat.ML`` -> ``stat:stat:ML``) that
includes cross-lists.

Each paper gets the date of the daily listing it was announced in, which is
not what the submission time predicts: about a quarter of a stat listing is
papers held in moderation for days or weeks. Two facts about arXiv give the
date exactly instead:

- New ids are handed out in announcement order, so a later listing's papers
  have higher ids than an earlier one's.
- A record's OAI datestamp is the UTC date of its last change, which is never
  before its announcement; for a paper nobody has touched since, it is the
  listing date itself (arXiv dates listings by the UTC day they go out).

So a paper's listing date is the earliest datestamp among all records with the
same or a higher id. Replacements and metadata edits only move datestamps
later, and old papers' low ids tie them to dates before the window. Checked
against arXiv's own /list/stat/pastweek pages for five listings (453 papers):
all but 5 dated correctly, none misdated, nothing extra. The 5 were cross-lists
added to papers first announced elsewhere days or months earlier; those are
missed.
"""

from __future__ import annotations

import datetime as dt
import email.utils
import re
import unicodedata
import xml.etree.ElementTree as ET

from . import resolve

OAI = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "raw": "http://arxiv.org/OAI/arXivRaw/"}


def _id_key(aid: str) -> tuple[int, int] | None:
    """Sort key for new-style ids (2609.30274); None for old ones (math/0501001)."""
    m = re.fullmatch(r"(\d{4})\.(\d{4,5})", aid)
    return (int(m[1]), int(m[2])) if m else None


def listing_dates(records: list[dict]) -> dict[str, str]:
    """arXiv id -> date of the listing it was announced in (see module docstring).
    Needs records with datestamps from before the window, so that old papers
    edited recently get dated before it too."""
    dated = sorted((r for r in records if _id_key(r["arxiv"])), key=lambda r: _id_key(r["arxiv"]))
    out, earliest = {}, "9999-99-99"
    for r in reversed(dated):
        earliest = min(earliest, r["datestamp"])
        out[r["arxiv"]] = earliest
    return out


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
    """Papers announced in the listings of the last `window_days` days (UTC),
    today's included: 7 days is the last five listings, like arXiv's "pastweek"."""
    today = dt.datetime.now(dt.timezone.utc).date()
    first = today - dt.timedelta(days=window_days - 1)
    # Harvest from a week earlier so there are records dated before the window
    # (listing_dates needs them, even across arXiv's holiday breaks).
    start = first - dt.timedelta(days=7)
    records: dict[str, dict] = {}
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
                n_seen += 1
                if p := _record(rec):
                    p["datestamp"] = rec.findtext("oai:header/oai:datestamp", "", NS)
                    records[p["arxiv"]] = p
            token = (lr.findtext("oai:resumptionToken", "", NS) or "").strip()
            if not token:
                break
            params = {"verb": "ListRecords", "resumptionToken": token}
        log(f"arXiv {cat}: {n_seen} records changed since {start:%Y-%m-%d}")
    dates = listing_dates(list(records.values()))
    papers = []
    for aid, d in dates.items():
        if d >= first.isoformat():
            p = records[aid]
            p["announced"] = d
            papers.append(p)
    log(f"{len(records)} records, {len(papers)} announced since {first:%Y-%m-%d}")
    return papers
