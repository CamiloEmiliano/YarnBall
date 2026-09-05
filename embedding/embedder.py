# embedder.py
"""Utility for generating embeddings using a free HuggingFace model."""

import os
from typing import List

_MODEL_NAME = os.getenv("EMBEDDER_MODEL")

# Load model lazily – shared singleton
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    class _DummyModel:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, normalize_embeddings=False):
            size = 768
            return [[0.0] * size for _ in texts]
    SentenceTransformer = _DummyModel

_model = SentenceTransformer(_MODEL_NAME)

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of embedding vectors for the supplied texts.
    The vectors are normalised to unit length for compatibility with pgvector.
    """
    result = _model.encode(texts, normalize_embeddings=True)
    # If the model returns a NumPy array (or similar) with a .tolist() method, convert it.
    return result.tolist() if hasattr(result, "tolist") else result
