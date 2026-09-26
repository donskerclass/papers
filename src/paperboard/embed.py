"""Local text embeddings (ONNX on CPU via fastembed), L2-normalised."""

from __future__ import annotations

import json
import os
from pathlib import Path
from functools import lru_cache

import numpy as np

from .paths import ROOT

CACHE_DIR = os.environ.get("FASTEMBED_CACHE", str(ROOT / ".cache" / "fastembed"))


# Models fastembed doesn't ship but can run from an ONNX file on the Hub.
CUSTOM = {
    # SPECTER (Cohan et al. 2020): scientific-paper embeddings trained on citations.
    "sentence-transformers/allenai-specter": dict(pooling="CLS", dim=768, model_file="onnx/model.onnx"),
}


@lru_cache(maxsize=2)
def _model(name: str):
    from fastembed import TextEmbedding
    if name in CUSTOM and name not in {m["model"] for m in TextEmbedding.list_supported_models()}:
        from fastembed.common.model_description import ModelSource, PoolingType
        spec = CUSTOM[name]
        TextEmbedding.add_custom_model(
            model=name, pooling=getattr(PoolingType, spec["pooling"]), normalization=True,
            sources=ModelSource(hf=name), dim=spec["dim"], model_file=spec["model_file"])
    try:
        return TextEmbedding(model_name=name, cache_dir=CACHE_DIR)
    except ValueError as e:
        # Older Hub repos (SPECTER) omit model_max_length from tokenizer_config.json,
        # which fastembed requires; add it to the downloaded copy and retry.
        if "maximum context length" not in str(e):
            raise
        for cfg in Path(CACHE_DIR).glob(f"models--{name.replace('/', '--')}/snapshots/*/tokenizer_config.json"):
            data = json.loads(cfg.read_text())
            data.setdefault("model_max_length", 512)
            cfg.unlink()  # usually a symlink into the blob store; replace it with a real file
            cfg.write_text(json.dumps(data))
        return TextEmbedding(model_name=name, cache_dir=CACHE_DIR)


def paper_text(title: str, abstract: str, max_chars: int = 2000) -> str:
    """Title first, then as much abstract as fits: the title is never cut."""
    text = f"{title.strip()}. {abstract.strip()}".strip()
    return text[:max(max_chars, len(title) + 2)]


def embed(texts: list[str], model: str, batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    vecs = np.asarray(list(_model(model).embed(texts, batch_size=batch_size)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs
