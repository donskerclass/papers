"""Load the editable profile (YAML) and resolve paper entries to title + abstract.

Resolutions are cached in data/profile_cache.json (committed), keyed by the
entry's identifier, so a daily run only looks up entries that are new.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import yaml

from . import resolve
from .paths import DATA, PROFILE

CACHE_FILE = DATA / "profile_cache.json"


@dataclass
class Item:
    kind: str                 # own | favorite | interest | avoid
    name: str                 # title, or interest name
    text: str                 # what gets embedded
    weight: float = 1.0
    url: str = ""
    meta: dict = field(default_factory=dict)


def load_yaml(name: str) -> dict:
    return yaml.safe_load((PROFILE / name).read_text()) or {}


def config() -> dict:
    return load_yaml("config.yaml")


def _key(e: dict) -> str:
    if e.get("arxiv"):
        return f"arxiv:{e['arxiv']}"
    if e.get("doi"):
        return f"doi:{str(e['doi']).lower()}"
    return f"title:{resolve.norm_title(e['title'])}"


def _is_placeholder(title: str) -> bool:
    return title.startswith("(")


def resolve_entries(entries: list[dict], log=print) -> dict[str, dict]:
    """Look up any entry not already cached. Returns the whole cache."""
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
    todo = [e for e in entries if not e.get("abstract") and _key(e) not in cache]
    if not todo:
        return cache
    log(f"resolving {len(todo)} profile entries")

    def check(e, rec) -> bool:
        """Accept a lookup only if its title agrees with the one written in the YAML."""
        if not rec or not rec.get("title"):
            return False
        if _is_placeholder(e["title"]):
            return True
        ok = resolve.title_match(e["title"], rec["title"]) >= 0.75
        if not ok:
            log(f"  ! id {_key(e)} resolved to '{rec['title']}', not '{e['title']}' -- trying title search")
        return ok

    ax = [e for e in todo if e.get("arxiv")]
    found = resolve.arxiv_by_ids([str(e["arxiv"]) for e in ax]) if ax else {}
    s2 = resolve.s2_batch([f"DOI:{str(e['doi']).lower()}" for e in todo if e.get("doi") and not e.get("arxiv")])
    for e in todo:
        k = _key(e)
        rec = None
        if e.get("arxiv"):
            rec = found.get(str(e["arxiv"]))
        elif e.get("doi"):
            rec = s2.get(f"DOI:{str(e['doi']).lower()}")
            if not (rec and rec.get("abstract")):
                rec = resolve.crossref_by_doi(str(e["doi"]).lower()) or rec
        if not check(e, rec):
            rec = None
            if not _is_placeholder(e["title"]):
                rec = resolve.s2_by_title(e["title"], min_ratio=0.85)
                if not (rec and rec.get("abstract")):
                    rec = resolve.arxiv_by_title(e["title"], min_ratio=0.85) or rec
        if rec:
            cache[k] = {"title": rec.get("title", ""), "abstract": rec.get("abstract", "")}
        else:
            log(f"  ? could not resolve {k} ('{e['title']}'); using the title alone")
            cache[k] = {"title": e["title"], "abstract": ""}
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(dict(sorted(cache.items())), indent=1, ensure_ascii=False))
    return cache


def _paper_items(kind: str, entries: list[dict], cache: dict) -> list[Item]:
    items = []
    for e in entries:
        w = float(e.get("weight", 1.0))
        if w <= 0:
            continue
        hit = cache.get(_key(e), {})
        title = e["title"] if not _is_placeholder(e["title"]) else (hit.get("title") or e["title"])
        abstract = (e.get("abstract") or hit.get("abstract") or "").strip()
        url = (f"https://arxiv.org/abs/{e['arxiv']}" if e.get("arxiv")
               else f"https://doi.org/{e['doi']}" if e.get("doi") else e.get("url", ""))
        items.append(Item(kind, title, f"{title}. {abstract}".strip(), w, url, {"source": e.get("source", "")}))
    return items


def load(resolve_missing: bool = True, log=print) -> dict[str, list[Item]]:
    own = load_yaml("own_papers.yaml").get("papers", [])
    fav = load_yaml("favorites.yaml").get("papers", [])
    ints = load_yaml("interests.yaml")
    cache = resolve_entries(own + fav, log=log) if resolve_missing else (
        json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {})
    return {
        "own": _paper_items("own", own, cache),
        "favorites": _paper_items("favorite", fav, cache),
        "interests": [Item("interest", i["name"], i["text"].strip(), float(i.get("weight", 1.0)))
                      for i in ints.get("interests", []) if float(i.get("weight", 1.0)) > 0],
        "avoid": [Item("avoid", i["name"], i["text"].strip(), float(i.get("weight", 1.0)))
                  for i in ints.get("avoid", []) if float(i.get("weight", 1.0)) > 0],
    }
