# embedding/embedder.py
# -*- coding: utf-8 -*-
"""Utility for generating dense vector embeddings using SentenceTransformers with lazy loading."""

import os
import logging
from typing import List, Optional
from pathlib import Path

# Load environment variables if not already loaded
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
_model_instance = None


def _get_model_name() -> str:
    raw = os.getenv("EMBEDDER_MODEL", _DEFAULT_MODEL)
    if not raw:
        return _DEFAULT_MODEL
    # Strip literal surrounding quotes if present in .env
    cleaned = raw.strip().strip('"').strip("'")
    return cleaned or _DEFAULT_MODEL


def get_embedder_model():
    """Lazily load and cache the SentenceTransformer singleton."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    model_name = _get_model_name()
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _model_instance = SentenceTransformer(model_name)
    except Exception as exc:
        logger.warning("Could not load SentenceTransformer ('%s'): %s. Using 768-dim mock fallback.", model_name, exc)

        class _FallbackModel:
            def encode(self, texts, normalize_embeddings=False):
                return [[0.0] * 768 for _ in texts]

        _model_instance = _FallbackModel()

    return _model_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return a list of normalized embedding vectors for the supplied texts."""
    if not texts:
        return []
    model = get_embedder_model()
    result = model.encode(texts, normalize_embeddings=True)
    return result.tolist() if hasattr(result, "tolist") else list(result)

