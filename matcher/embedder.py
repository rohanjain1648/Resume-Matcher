"""Thin wrapper around a sentence-embedding model (loaded once, lazily)."""
import os
import warnings

import numpy as np

MODEL_NAME = os.environ.get("MATCHER_MODEL", "BAAI/bge-small-en-v1.5")
_model = None
DIM = 384


def _load():
    global _model
    if _model is None:
        warnings.filterwarnings("ignore")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts):
    """Unit-length float32 vectors, shape (n, d)."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return np.asarray(_load().encode(list(texts), normalize_embeddings=True, batch_size=32,
                                     show_progress_bar=False), dtype=np.float32)
