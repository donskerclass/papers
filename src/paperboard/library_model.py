"""The "what I read" component: a tf-idf classifier and nearest-neighbour score.

Fitted locally from the Zotero export (positives) against a background sample
of arXiv in the watched categories (negatives), in the spirit of
arxiv-sanity-lite. What gets committed (data/library/) is only derived data:
the tf-idf vocabulary and idf weights, the classifier coefficients, and the
library's embedding vectors. No titles or abstracts.
"""

from __future__ import annotations

import json
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .paths import DATA

LIB_DIR = DATA / "library"

# The library mixes journal PDFs, working papers and preprints from 2009 on; the
# background is recent arXiv. Words that mark *where or when* a text comes from
# rather than *what it is about* would otherwise dominate the classifier, so
# they are removed: publisher boilerplate, and generic academic phrasing whose
# frequency drifts across venues and years.
STYLE_WORDS = """
abstract introduction keywords keyword jel classification codes code elsevier springer wiley
rights reserved copyright journal university department working paper papers draft version
preliminary comments welcome email http https www doi org pdf vol pp press published
article section appendix thanks thank grateful acknowledge seminar participants
novel propose proposed proposes framework findings finding demonstrate demonstrates
effectiveness extensive experiments experiment superior outperforms state-of-the-art
address addresses leverage leverages leveraging remains existing recent recently study
studies paper introduce introduces establish establishes yield yields real-world
argue argues argued discuss discusses discussed suggest suggests suggested note notes
review reviews basic general simple important useful way ways new use uses used idea ideas
variety problems problem world survey lecture lectures chapter book course
""".split()
STOP = sorted(set(TfidfVectorizer(stop_words="english").get_stop_words()) | set(STYLE_WORDS))

_MATH = re.compile(r"\$[^$]{0,300}\$|\\\(.{0,300}?\\\)|\\[a-zA-Z]+(\{[^}]*\})?")
_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")


def preprocess(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _MATH.sub(" ", text)
    return text.lower()


TFIDF_PARAMS = dict(
    preprocessor=preprocess, strip_accents="unicode", stop_words=STOP, ngram_range=(1, 2),
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]+\b", sublinear_tf=True, norm="l2",
)


def fit_tfidf(pos_texts: list[str], neg_texts: list[str], C: float = 2.0,
              max_features: int = 40000, min_df: int = 3):
    vec = TfidfVectorizer(**TFIDF_PARAMS, min_df=min_df, max_df=0.3, max_features=max_features)
    X = vec.fit_transform(pos_texts + neg_texts)
    y = np.r_[np.ones(len(pos_texts)), np.zeros(len(neg_texts))]
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(X, y)
    return vec, clf


def save(vec: TfidfVectorizer, clf: LogisticRegression, lib_emb: np.ndarray, meta: dict) -> None:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    vocab = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    (LIB_DIR / "tfidf_vocab.json").write_text(json.dumps(vocab))
    np.savez_compressed(LIB_DIR / "tfidf_weights.npz", idf=vec.idf_.astype(np.float32),
                        coef=clf.coef_.ravel().astype(np.float32),
                        intercept=np.float32(clf.intercept_[0]))
    # Row order is shuffled so the file carries no information about the library's order.
    rng = np.random.default_rng(0)
    np.save(LIB_DIR / "embeddings.npy", lib_emb[rng.permutation(len(lib_emb))].astype(np.float16))
    (LIB_DIR / "meta.json").write_text(json.dumps(meta, indent=1))


class Library:
    """Loaded, fitted library model used by the daily run."""

    def __init__(self):
        vocab = json.loads((LIB_DIR / "tfidf_vocab.json").read_text())
        w = np.load(LIB_DIR / "tfidf_weights.npz")
        self.vec = TfidfVectorizer(**TFIDF_PARAMS, vocabulary={t: i for i, t in enumerate(vocab)})
        self.vec.idf_ = w["idf"].astype(np.float64)
        self.coef = w["coef"].astype(np.float64)
        self.intercept = float(w["intercept"])
        self.vocab = np.array(vocab)
        self.emb = np.load(LIB_DIR / "embeddings.npy").astype(np.float32)
        self.meta = json.loads((LIB_DIR / "meta.json").read_text())

    @staticmethod
    def available() -> bool:
        return (LIB_DIR / "meta.json").exists()

    def tfidf_scores(self, texts: list[str], top_terms: int = 4) -> tuple[np.ndarray, list[list[str]]]:
        X = self.vec.transform(texts)
        scores = X @ self.coef + self.intercept
        contrib = sparse.csr_matrix(X.multiply(self.coef))
        terms = []
        for i in range(X.shape[0]):
            row = contrib.getrow(i)
            order = np.argsort(-row.data)[:top_terms]
            terms.append([str(self.vocab[row.indices[j]]) for j in order if row.data[j] > 0])
        return np.asarray(scores).ravel(), terms

    def knn_scores(self, vecs: np.ndarray, k: int = 10) -> np.ndarray:
        sims = vecs @ self.emb.T
        k = min(k, sims.shape[1])
        top = np.partition(sims, -k, axis=1)[:, -k:]
        return top.mean(axis=1)
