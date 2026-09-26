"""Turn identifiers or bare titles into {title, abstract, ...} using free, keyless APIs.

Order of preference: arXiv (for arXiv ids), Semantic Scholar batch lookup
(DOIs and arXiv ids), OpenAlex (DOIs, then title search). Everything is
polite: small batches, pauses between requests, retries on 429/5xx.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import time
import urllib.parse

import feedparser
import requests

UA = {"User-Agent": "paperboard/0.1 (+https://github.com/donskerclass/papers)"}
ARXIV_API = "https://export.arxiv.org/api/query"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
OPENALEX = "https://api.openalex.org/works"

S2_MATCH = "https://api.semanticscholar.org/graph/v1/paper/search/match"
# Optional free keys. Without them both services share a small anonymous
# budget that is often exhausted (OpenAlex in particular); with them they work.
OPENALEX_KEY = os.environ.get("OPENALEX_API_KEY", "")
S2_KEY = os.environ.get("S2_API_KEY", "")



_last_call: dict[str, float] = {}
MIN_INTERVAL = {"export.arxiv.org": 4.0, "oaipmh.arxiv.org": 3.0, "api.semanticscholar.org": 1.2, "api.crossref.org": 0.5}


def _throttle(url: str) -> None:
    host = urllib.parse.urlparse(url).netloc
    gap = MIN_INTERVAL.get(host, 0.0)
    wait = _last_call.get(host, 0.0) + gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()


class _CurlResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code, self.text = status_code, text


def _curl_get(url: str, params: dict | None) -> _CurlResponse | None:
    """GET through the curl binary. arXiv's CDN sometimes answers Python's HTTP
    clients with a lasting run of 406s (after a burst of requests) while
    serving curl normally; curl ships on macOS and on GitHub's Ubuntu runners."""
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    try:
        out = subprocess.run(["curl", "-s", "-A", UA["User-Agent"], "-w", "\n%{http_code}", full],
                             capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired):
        return None
    body, _, code = out.stdout.rpartition("\n")
    return _CurlResponse(int(code or 0), body)


def _get(url: str, *, params=None, json=None, tries: int = 5, pause: float = 3.0):
    """GET/POST with per-host pacing and exponential backoff.

    arXiv answers an over-eager client with 406 rather than 429; for arXiv a
    406 is retried at once through curl, then backed off like a rate limit.
    """
    headers = dict(UA)
    if "semanticscholar" in url and S2_KEY:
        headers["x-api-key"] = S2_KEY
    if "openalex" in url and OPENALEX_KEY:
        params = {**(params or {}), "api_key": OPENALEX_KEY}
    arxiv = "arxiv.org" in url
    if arxiv:
        # arXiv's throttle lasts minutes and retrying inside it seems to extend it.
        pause = max(pause, 60.0)
    for attempt in range(tries):
        _throttle(url)
        try:
            r = (requests.post(url, params=params, json=json, headers=headers, timeout=60) if json is not None
                 else requests.get(url, params=params, headers=headers, timeout=60))
        except requests.RequestException:
            r = None
        if arxiv and json is None and (r is None or r.status_code == 406):
            _throttle(url)
            r = _curl_get(url, params) or r
        if r is not None and r.status_code == 200:
            return r
        if r is not None and r.status_code in (400, 404):
            return None
        time.sleep(min(pause * (2 ** attempt), 300))
    return None


def crossref_by_doi(doi: str) -> dict | None:
    r = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", tries=3, pause=2)
    if r is None:
        return None
    it = r.json().get("message", {})
    t = (it.get("title") or [""])[0]
    return {"title": clean(t), "doi": doi,
            "abstract": clean(re.sub(r"^\s*(<jats:title>)?Abstract(</jats:title>)?", "", it.get("abstract") or ""))}


def crossref_by_title(title: str, min_ratio: float = 0.9) -> dict | None:
    """Title -> DOI (and abstract, when the publisher deposits one) via Crossref."""
    r = _get("https://api.crossref.org/works", params={
        "query.bibliographic": title[:300], "rows": 3, "select": "DOI,title,abstract,published"}, tries=3, pause=2)
    if r is None:
        return None
    for it in r.json().get("message", {}).get("items", []):
        t = (it.get("title") or [""])[0]
        if title_match(title, t) >= min_ratio:
            return {"title": clean(t), "doi": it["DOI"].lower(),
                    "abstract": clean(re.sub(r"^\s*(<jats:title>)?Abstract(</jats:title>)?", "", it.get("abstract") or ""))}
    return None


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def title_match(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).strip()


# --- arXiv -------------------------------------------------------------------

def parse_arxiv_entry(e) -> dict:
    aid = e.id.rsplit("/abs/", 1)[-1]
    base = re.sub(r"v\d+$", "", aid)
    return {
        "id": f"arxiv:{base}",
        "arxiv": base,
        "version": int(m.group(1)) if (m := re.search(r"v(\d+)$", aid)) else 1,
        "title": clean(e.title),
        "abstract": clean(e.summary),
        "authors": [a.name for a in e.get("authors", [])],
        "categories": [t["term"] for t in e.get("tags", [])],
        "primary": e.get("arxiv_primary_category", {}).get("term", ""),
        "published": e.published,
        "updated": e.updated,
        "url": f"https://arxiv.org/abs/{base}",
        "doi": e.get("arxiv_doi", ""),
        "source": "arxiv",
    }


def arxiv_by_ids(ids: list[str]) -> dict[str, dict]:
    from .arxiv_source import get_records  # OAI-PMH: the search API is closed to cloud machines
    return get_records(ids)


# --- Semantic Scholar ----------------------------------------------------------

def s2_batch(keys: list[str]) -> dict[str, dict]:
    """keys like 'DOI:10.x/y' or 'ARXIV:2101.00001'. Returns key -> record."""
    out = {}
    for i in range(0, len(keys), 400):
        chunk = keys[i:i + 400]
        r = _get(S2_BATCH, params={"fields": "title,abstract,year,externalIds,authors"}, json={"ids": chunk}, pause=5)
        if r is not None:
            for k, p in zip(chunk, r.json()):
                if p:
                    out[k] = {
                        "title": p.get("title") or "",
                        "abstract": clean(p.get("abstract") or ""),
                        "year": p.get("year"),
                        "authors": [a["name"] for a in p.get("authors") or []],
                        "external_ids": p.get("externalIds") or {},
                    }
    return out


def s2_by_title(title: str, min_ratio: float = 0.9) -> dict | None:
    r = _get(S2_MATCH, params={"query": title[:300], "fields": "title,abstract,year,externalIds,authors"},
             tries=4, pause=5)
    if r is None:
        return None
    for p in r.json().get("data", [])[:1]:
        if title_match(title, p.get("title") or "") >= min_ratio:
            return {
                "title": p.get("title") or "",
                "abstract": clean(p.get("abstract") or ""),
                "year": p.get("year"),
                "authors": [a["name"] for a in p.get("authors") or []],
                "external_ids": p.get("externalIds") or {},
            }
    return None


# --- OpenAlex ----------------------------------------------------------------

def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return clean(" ".join(w for _, w in pos))


def _openalex_rec(w: dict) -> dict:
    return {
        "title": clean(w.get("display_name") or ""),
        "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
        "year": w.get("publication_year"),
        "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
        "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
    }


def openalex_by_dois(dois: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(dois), 50):
        chunk = dois[i:i + 50]
        r = _get(OPENALEX, params={"filter": "doi:" + "|".join(chunk), "per-page": 50}, pause=2)
        if r is not None:
            for w in r.json().get("results", []):
                rec = _openalex_rec(w)
                out[rec["doi"]] = rec
    return out


def openalex_by_title(title: str, min_ratio: float = 0.9) -> dict | None:
    q = re.sub(r"[,:;|()\[\]{}!?\"']", " ", title)[:250]
    r = _get(OPENALEX, params={"search": q, "per-page": 5}, tries=3, pause=2)
    if r is None:
        return None
    best, score = None, 0.0
    for w in r.json().get("results", []):
        s = title_match(title, w.get("display_name") or "")
        if s > score and w.get("abstract_inverted_index"):
            best, score = w, s
    return _openalex_rec(best) if best is not None and score >= min_ratio else None


def arxiv_by_title(title: str, min_ratio: float = 0.9) -> dict | None:
    if os.environ.get("GITHUB_ACTIONS"):
        return None  # arXiv's search API refuses GitHub's runners; would only time out
    words = norm_title(title).split()[:14]
    q = " AND ".join(f"ti:{w}" for w in words if len(w) > 2)
    r = _get(ARXIV_API, params={"search_query": q, "max_results": 5}, tries=3)
    if r is None:
        return None
    for e in feedparser.parse(r.text).entries:
        if "title" in e and title_match(title, e.title) >= min_ratio:
            return parse_arxiv_entry(e)
    return None


def doi_from_url(url: str) -> str:
    url = urllib.parse.unquote(url)
    m = re.search(r"10\.\d{4,9}/[^\s?#&]+", url)
    return m.group(0).rstrip(".").lower() if m else ""


def arxiv_from_url(url: str) -> str:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", url, re.I)
    return m.group(1) if m else ""
