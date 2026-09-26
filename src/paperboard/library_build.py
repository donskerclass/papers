"""Local-only: export the Zotero library, fill in missing abstracts, fetch a
background sample of arXiv, and fit the models that represent "what I read".

Only the fitted models (``data/library/``) are committed. The export and the
background corpus stay under ``local/`` (gitignored).
"""

from __future__ import annotations

import datetime as dt
import gzip
import json
import random
from pathlib import Path

import feedparser

from . import resolve, zotero
from .paths import LOCAL

ENRICH_CACHE = LOCAL / "enrich_cache.json"
LIBRARY_FILE = LOCAL / "library.jsonl.gz"
BACKGROUND_FILE = LOCAL / "background.jsonl.gz"


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(p: Path) -> list[dict]:
    with gzip.open(p, "rt") as f:
        return [json.loads(line) for line in f]


def export_library(log=print) -> list[dict]:
    recs = zotero.export()
    cache = _load_json(ENRICH_CACHE)
    todo = [r for r in recs if not r["abstract"] and r["key"] not in cache]
    log(f"{len(recs)} items; {sum(not r['abstract'] for r in recs)} lack abstracts, {len(todo)} not yet looked up")

    def save():
        ENRICH_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ENRICH_CACHE.write_text(json.dumps(cache, ensure_ascii=False))

    def accept(r, rec, source, check_title=True):
        if rec and rec.get("abstract") and len(rec["abstract"]) > 200:
            if check_title and rec.get("title") and resolve.title_match(r["title"], rec["title"]) < 0.8:
                return False
            cache[r["key"]] = {"abstract": rec["abstract"], "source": source}
            return True
        return False

    # 1. arXiv ids, through Semantic Scholar's batch endpoint (one request, and
    # it keeps this bulk step off the arXiv API, which throttles bursts hard)
    ax = [r for r in todo if r["arxiv"]]
    found = resolve.s2_batch([f"ARXIV:{r['arxiv']}" for r in ax])
    for r in ax:
        accept(r, found.get(f"ARXIV:{r['arxiv']}"), "semanticscholar")
    save()
    # 2. DOIs via Semantic Scholar, then OpenAlex
    left = [r for r in todo if r["key"] not in cache and r["doi"]]
    s2 = resolve.s2_batch([f"DOI:{r['doi']}" for r in left])
    for r in left:
        accept(r, s2.get(f"DOI:{r['doi']}"), "semanticscholar")
    left = [r for r in left if r["key"] not in cache]
    oa = resolve.openalex_by_dois([r["doi"] for r in left])
    for r in left:
        accept(r, oa.get(r["doi"]), "openalex")
    save()
    # 3. title search for items with no usable PDF text either: Crossref finds
    # the DOI (sometimes with an abstract), then one Semantic Scholar batch
    # call fetches abstracts for all the DOIs found. OpenAlex title search is
    # used only when a (free) key is configured.
    left = [r for r in todo if r["key"] not in cache
            and len(zotero.abstract_from_fulltext(r["fulltext_head"])) <= 200]
    log(f"title search for {len(left)}")
    dois = {}
    for i, r in enumerate(left):
        hit = resolve.crossref_by_title(r["title"])
        if hit and not accept(r, hit, "crossref", check_title=False):
            dois[r["key"]] = hit["doi"]
        if i % 50 == 0:
            log(f"  crossref {i}/{len(left)}, {len(dois)} DOIs")
    s2 = resolve.s2_batch([f"DOI:{d}" for d in dois.values()])
    for r in left:
        if r["key"] in dois:
            accept(r, s2.get(f"DOI:{dois[r['key']]}"), "semanticscholar", check_title=False)
    if resolve.OPENALEX_KEY:
        for r in left:
            if r["key"] not in cache:
                accept(r, resolve.openalex_by_title(r["title"]), "openalex-title", check_title=False)
    for r in left:
        cache.setdefault(r["key"], {"abstract": "", "source": "none"})
    save()

    for r in recs:
        if not r["abstract"]:
            hit = cache.get(r["key"], {})
            if hit.get("abstract"):
                r["abstract"], r["abstract_source"] = hit["abstract"], hit["source"]
            elif (ft := zotero.abstract_from_fulltext(r["fulltext_head"])) and len(ft) > 200:
                r["abstract"], r["abstract_source"] = ft, "pdf"
        r.pop("fulltext_head", None)
    write_jsonl(LIBRARY_FILE, recs)
    n = sum(bool(r["abstract"]) for r in recs)
    log(f"wrote {LIBRARY_FILE}: {len(recs)} items, {n} with abstracts")
    return recs


def _embed_cached(texts: list[str], model: str, log=print):
    """Embed library texts, reusing vectors from earlier builds (local cache)."""
    import hashlib

    import numpy as np

    from .embed import embed

    path = LOCAL / f"emb_cache_{model.replace('/', '__')}.npz"
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    cache = {}
    if path.exists():
        z = np.load(path)
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [i for i, k in enumerate(keys) if k not in cache]
    log(f"embedding {len(todo)} of {len(texts)} library texts with {model} ({len(texts) - len(todo)} cached)")
    if todo:
        for i, v in zip(todo, embed([texts[i] for i in todo], model)):
            cache[keys[i]] = v
        np.savez(path, keys=np.array(list(cache)), vecs=np.stack(list(cache.values())))
    return np.stack([cache[k] for k in keys])


def build(skip_export: bool = False, skip_background: bool = False, log=print) -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from . import library_model
    from .embed import embed, paper_text
    from .profile import config

    cfg = config()
    model = cfg["embedding"]["model"]
    lib = read_jsonl(LIBRARY_FILE) if skip_export and LIBRARY_FILE.exists() else export_library(log=log)
    if skip_background and BACKGROUND_FILE.exists():
        bg = read_jsonl(BACKGROUND_FILE)
    else:
        # Mostly the last two years, plus older windows so that "recent" is not itself the signal.
        cats = cfg["sources"]["arxiv"]["categories"]
        fetch_background(cats, log=log)
        bg = fetch_background(cats, days=3000, min_back=730, windows=40, seed=1, log=log)

    lib_titles = {resolve.norm_title(r["title"]) for r in lib}
    bg = [r for r in bg if resolve.norm_title(r["title"]) not in lib_titles]
    # The classifier is trained only on library items with a real abstract (not
    # PDF-extracted text, which drags in journal boilerplate), so that neither
    # "has an abstract" nor publisher formatting becomes the signal.
    labelled = [r for r in lib if r["abstract"] and r["abstract_source"] != "pdf"]
    txt = lambda r: f"{r['title']}. {r['abstract']}"  # noqa: E731
    pos, neg = [txt(r) for r in labelled], [txt(r) for r in bg]
    log(f"tf-idf: {len(pos)} library papers vs {len(neg)} background")

    # Held-out check before fitting on everything. The arXiv-only figure keeps
    # to library papers that are themselves arXiv preprints, so writing style is
    # the same on both sides and only topic can separate them.
    rng = np.random.default_rng(0)
    li, bi = rng.permutation(len(labelled)), rng.permutation(len(bg))
    cut_l, cut_b = int(0.8 * len(li)), int(0.8 * len(bi))
    v, c = library_model.fit_tfidf([pos[i] for i in li[:cut_l]], [neg[i] for i in bi[:cut_b]])
    s_bg = c.decision_function(v.transform([neg[i] for i in bi[cut_b:]]))
    te = [labelled[i] for i in li[cut_l:]]
    aucs = {}
    for name, rows in (("all", te), ("arxiv_only", [r for r in te if r["arxiv"]])):
        s_lib = c.decision_function(v.transform([txt(r) for r in rows]))
        aucs[name] = float(roc_auc_score(np.r_[np.ones(len(rows)), np.zeros(len(s_bg))], np.r_[s_lib, s_bg]))
    log(f"tf-idf held-out AUC: {aucs['all']:.3f} overall, {aucs['arxiv_only']:.3f} on arXiv preprints only")

    vec, clf = library_model.fit_tfidf(pos, neg)
    max_chars = cfg["embedding"].get("max_chars", 2000)
    lib_emb = _embed_cached([paper_text(r["title"], r["abstract"], max_chars) for r in lib], model, log)
    library_model.save(vec, clf, lib_emb, {
        "embedding_model": model,
        "n_library": len(lib),
        "n_library_with_abstract": len(pos),
        "n_background": len(neg),
        "tfidf_heldout_auc": round(aucs["all"], 4),
        "tfidf_heldout_auc_arxiv_only": round(aucs["arxiv_only"], 4),
        "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"),
    })
    log(f"wrote {library_model.LIB_DIR}")


def fetch_background(categories: list[str], days: int = 730, per_window: int = 1000,
                     windows: int = 48, seed: int = 0, min_back: int = 7, log=print) -> list[dict]:
    """A random-ish sample of arXiv from the watched categories over the last ``days``.

    Samples ``windows`` random 3-day windows and takes up to ``per_window``
    papers from each, so the sample spans the period rather than the last week.
    """
    rng = random.Random(seed)
    today = dt.date.today()
    cat_q = "(" + " OR ".join(f"cat:{c}" for c in categories) + ")"
    rows: dict[str, dict] = {}
    if BACKGROUND_FILE.exists():
        rows = {r["arxiv"]: r for r in read_jsonl(BACKGROUND_FILE)}
        log(f"background: {len(rows)} cached")
    starts = sorted(rng.sample(range(min_back, days), windows))
    for k, back in enumerate(starts):
        d0 = today - dt.timedelta(days=back)
        d1 = d0 + dt.timedelta(days=3)
        q = f"{cat_q} AND submittedDate:[{d0:%Y%m%d}0000 TO {d1:%Y%m%d}0000]"
        r = resolve._get(resolve.ARXIV_API, params={
            "search_query": q, "start": 0, "max_results": per_window,
            "sortBy": "submittedDate", "sortOrder": "descending"})
        entries = [e for e in feedparser.parse(r.text).entries if "title" in e] if r is not None else []
        for e in entries:
            rec = resolve.parse_arxiv_entry(e)
            rows[rec["arxiv"]] = rec
        got = len(entries)
        log(f"  window {k + 1}/{windows} {d0}: {got}")
    write_jsonl(BACKGROUND_FILE, rows.values())
    log(f"wrote {BACKGROUND_FILE}: {len(rows)} papers")
    return list(rows.values())
