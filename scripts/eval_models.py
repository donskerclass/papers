"""Compare embedding models on how well each criterion separates my library
from a background sample of arXiv (ROC AUC). Local only; needs local/ data.

    uv run python scripts/eval_models.py BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5
"""

import random
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from paperboard import profile
from paperboard.embed import embed, paper_text
from paperboard.library_build import BACKGROUND_FILE, LIBRARY_FILE, read_jsonl
from paperboard.resolve import norm_title

rng = random.Random(0)
lib = [r for r in read_jsonl(LIBRARY_FILE) if r["abstract"]]
titles = {norm_title(r["title"]) for r in lib}
bg = [r for r in read_jsonl(BACKGROUND_FILE) if norm_title(r["title"]) not in titles]
bg = rng.sample(bg, min(1500, len(bg)))
rng.shuffle(lib)
cut = int(0.8 * len(lib))
lib_train = lib[:cut]
# Test positives are library papers that are themselves arXiv preprints with a
# real abstract, so genre (journal/PDF text vs arXiv abstract) can't separate them.
lib_test = [r for r in lib[cut:] + lib[:cut] if r["arxiv"] and r["abstract_source"] in ("zotero", "semanticscholar")]
lib_test = lib_test[:400]
test_titles = {r["key"] for r in lib_test}
lib_train = [r for r in lib if r["key"] not in test_titles]
prof = profile.load(resolve_missing=False)
y = np.r_[np.ones(len(lib_test)), np.zeros(len(bg))]
print(f"{len(lib_train)} library train, {len(lib_test)} library test, {len(bg)} background")

for model in sys.argv[1:] or ["BAAI/bge-small-en-v1.5"]:
    t0 = time.time()
    E_train = embed([paper_text(r["title"], r["abstract"]) for r in lib_train], model)
    E_test = embed([paper_text(r["title"], r["abstract"]) for r in lib_test + bg], model)
    secs = time.time() - t0
    out = {}
    for k in (5, 10, 25):
        s = np.sort(E_test @ E_train.T, axis=1)[:, -k:].mean(axis=1)
        out[f"knn{k}"] = roc_auc_score(y, s)
    for key in ("own", "favorites", "interests"):
        items = prof[key]
        P = embed([it.text for it in items], model)
        w = np.array([it.weight for it in items])
        out[key] = roc_auc_score(y, ((E_test @ P.T) * w).max(axis=1))
    rate = (len(E_train) + len(E_test)) / secs
    print(f"{model:40s} {rate:6.0f} texts/s  " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)
