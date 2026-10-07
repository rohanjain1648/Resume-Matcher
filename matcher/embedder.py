"""Thin wrapper around a sentence-embedding model (loaded once, lazily)."""
import os
import warnings

import numpy as np

MODEL_NAME = os.environ.get("MATCHER_MODEL", "BAAI/bge-base-en-v1.5")   # e.g. BAAI/bge-small-en-v1.5 on low-RAM machines
_model = None


def _load():
    global _model
    if _model is None:
        warnings.filterwarnings("ignore")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def dim() -> int:
    return _load().get_sentence_embedding_dimension()


def encode(texts):
    """Unit-length float32 vectors, shape (n, d)."""
    if not texts:
        return np.zeros((0, dim()), dtype=np.float32)
    return np.asarray(_load().encode(list(texts), normalize_embeddings=True, batch_size=32,
                                     show_progress_bar=False), dtype=np.float32)
