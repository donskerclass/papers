"""Score candidates on four criteria and combine them.

Each criterion is computed as a raw similarity, then replaced by its rank
within the run scaled to [0, 1] so the criteria are comparable (raw cosines
against different profile sets live on different scales). The combined score
is the weighted sum of ranks minus an avoid penalty.
"""

from __future__ import annotations

import numpy as np

from .embed import embed, paper_text
from .library_model import Library
from .profile import Item

CRITERIA = ("own", "favorites", "interests", "library")


def rank01(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.ones_like(x, dtype=float)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return order / (len(x) - 1)


def _weighted_max(sims: np.ndarray, items: list[Item], hub_correction: bool = False
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per candidate: best weighted similarity, index of the item that won, its raw cosine.

    With ``hub_correction`` each profile item's mean similarity over the day's
    candidates is subtracted first, so an item that is moderately close to
    everything (a generic abstract) doesn't win every comparison; an item wins
    where it matches unusually well.
    """
    w = np.array([it.weight for it in items])
    base = sims - sims.mean(axis=0, keepdims=True) if hub_correction and len(sims) > 1 else sims
    ws = base * w
    idx = ws.argmax(axis=1)
    rows = np.arange(len(sims))
    return ws[rows, idx], idx, sims[rows, idx]


def score(papers: list[dict], profile: dict[str, list[Item]], cfg: dict, log=print) -> list[dict]:
    model = cfg["embedding"]["model"]
    max_chars = cfg["embedding"].get("max_chars", 2000)
    sc = cfg["scoring"]
    texts = [paper_text(p["title"], p["abstract"], max_chars) for p in papers]
    log(f"embedding {len(texts)} candidates with {model}")
    X = embed(texts, model)
    prof_vecs = {k: embed([it.text[:max_chars] for it in v], model) for k, v in profile.items() if v}

    raw: dict[str, np.ndarray] = {}
    why: dict[str, list[str]] = {}
    wanted = np.full(len(papers), -np.inf)   # best raw similarity to anything wanted
    for crit, key in (("own", "own"), ("favorites", "favorites"), ("interests", "interests")):
        items = profile.get(key, [])
        if not items:
            continue
        sims = X @ prof_vecs[key].T
        best, idx, cos = _weighted_max(sims, items, sc.get("hub_correction", True))
        raw[crit] = best
        why[crit] = [f"{items[j].name}|{c:.2f}" for j, c in zip(idx, cos)]
        wanted = np.maximum(wanted, sims.max(axis=1))

    # The avoid penalty uses raw cosines: a paper is pushed down only when it is
    # closer to an avoid statement than to *every* wanted item (own papers,
    # favorites and interests). Methods papers in causal inference sit near
    # clinical-trial reports in embedding space, so comparing against the
    # interests alone would penalise exactly the papers that match my own work.
    penalty = np.zeros(len(papers))
    if profile.get("avoid") and np.isfinite(wanted).all():
        a_sims = X @ prof_vecs["avoid"].T
        a_idx = a_sims.argmax(axis=1)
        excess = a_sims.max(axis=1) - wanted
        penalty = sc.get("avoid_penalty", 0.3) * np.clip(excess / sc.get("avoid_margin", 0.02), 0, 1)
        why["avoid"] = [profile["avoid"][j].name for j in a_idx]

    library_terms = [[] for _ in papers]
    if Library.available():
        lib = Library()
        if lib.meta.get("embedding_model") != model:
            log(f"warning: library vectors were built with {lib.meta.get('embedding_model')}, "
                f"not {model}; skipping the nearest-neighbour part")
            knn = None
        else:
            knn = lib.knn_scores(X, sc.get("knn_k", 10))
        tf, library_terms = lib.tfidf_scores([f"{p['title']}. {p['abstract']}" for p in papers])
        mix = sc.get("library_mix", {"tfidf": 0.5, "knn": 0.5})
        raw["library"] = (mix["tfidf"] * rank01(tf) + (mix["knn"] * rank01(knn) if knn is not None else 0)) \
            / (mix["tfidf"] + (mix["knn"] if knn is not None else 0))
        raw_tfidf, raw_knn = tf, knn
    else:
        log("no library model found (run `paperboard build-library` locally); skipping that criterion")
        raw_tfidf = raw_knn = None

    weights = {c: sc["weights"][c] for c in CRITERIA if c in raw}
    total_w = sum(weights.values())
    ranks = {c: rank01(raw[c]) for c in weights}
    combined = sum(weights[c] / total_w * ranks[c] for c in weights) - penalty

    for i, p in enumerate(papers):
        p["score"] = float(combined[i])
        p["ranks"] = {c: float(ranks[c][i]) for c in ranks}
        p["raw"] = {c: float(raw[c][i]) for c in raw}
        if raw_tfidf is not None:
            p["raw"]["tfidf"] = float(raw_tfidf[i])
        if raw_knn is not None:
            p["raw"]["knn"] = float(raw_knn[i])
        p["penalty"] = float(penalty[i])
        p["why"] = {}
        for c in ("own", "favorites", "interests"):
            if c in why:
                name, cos = why[c][i].rsplit("|", 1)
                p["why"][c] = {"item": name, "cos": float(cos)}
        if "library" in raw:
            p["why"]["library"] = {"terms": library_terms[i]}
        if penalty[i] > 0:
            p["why"]["avoid"] = why["avoid"][i]
        p["best"] = max(ranks, key=lambda c: ranks[c][i]) if ranks else None
    pct = rank01(combined)
    for i, p in enumerate(papers):
        p["percentile"] = float(pct[i])
    return papers
