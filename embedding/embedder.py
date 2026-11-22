# embedding/embedder.py
# -*- coding: utf-8 -*-
"""Utility for generating dense vector embeddings using SentenceTransformers (ADR-00020).

Enforces fail-fast initialization on missing ML dependencies/weights and prohibits
zero-norm or non-finite embedding vectors from reaching pgvector storage or queries.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Load environment variables if not already loaded
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
_DEFAULT_EMBEDDING_DIM = 768
_MIN_NORM_SQ = 1e-8

_model_instance: Any = None
_mock_model_instance: Any = None


class EmbedderUnavailableError(RuntimeError):
    """Raised when SentenceTransformer cannot be loaded and mock mode was not requested."""


class InvalidEmbeddingError(ValueError):
    """Raised when an embedding vector is empty, non-finite (NaN/Inf), or zero-norm."""


def reset_embedder_cache() -> None:
    """Reset cached embedder instances (intended for deterministic unit tests)."""
    global _model_instance, _mock_model_instance
    _model_instance = None
    _mock_model_instance = None


def _get_model_name() -> str:
    raw = os.getenv("EMBEDDER_MODEL", _DEFAULT_MODEL)
    if not raw:
        return _DEFAULT_MODEL
    cleaned = raw.strip().strip('"').strip("'")
    return cleaned or _DEFAULT_MODEL


def _get_embedding_dim() -> int:
    raw = os.getenv("EMBEDDING_DIM", str(_DEFAULT_EMBEDDING_DIM)).strip()
    try:
        dim = int(raw)
        return dim if dim > 0 else _DEFAULT_EMBEDDING_DIM
    except ValueError:
        return _DEFAULT_EMBEDDING_DIM


def _is_mock_enabled(allow_mock: Optional[bool] = None) -> bool:
    if allow_mock is not None:
        return bool(allow_mock)
    env_flag = os.getenv("YARNBALL_EMBEDDER_MOCK", "").strip().lower()
    return env_flag in ("1", "true", "yes", "on")


class _DeterministicMockEmbedder:
    """Explicit opt-in mock embedder emitting deterministic unit-norm non-zero vectors."""

    is_mock: bool = True

    def __init__(self, dim: int = _DEFAULT_EMBEDDING_DIM) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            raw_vals: List[float] = []
            for idx in range(self.dim):
                byte_a = digest[idx % len(digest)]
                byte_b = digest[(idx * 7 + 3) % len(digest)]
                val = ((byte_a ^ byte_b) / 127.5) - 1.0
                if abs(val) < 1e-3:
                    val = 0.25 if (idx % 2 == 0) else -0.25
                raw_vals.append(val)
            if normalize_embeddings:
                norm = math.sqrt(sum(v * v for v in raw_vals))
                raw_vals = [v / norm for v in raw_vals]
            vectors.append(raw_vals)
        return vectors


def validate_embedding_vectors(
    vectors: Sequence[Sequence[float]],
    min_norm_sq: float = _MIN_NORM_SQ,
) -> List[List[float]]:
    """Verify every vector is non-empty, finite, and non-zero under L2 norm."""
    validated: List[List[float]] = []
    for idx, vec in enumerate(vectors):
        if vec is None or len(vec) == 0:
            raise InvalidEmbeddingError(f"Embedding vector at index {idx} is empty.")
        row: List[float] = []
        for coord_idx, raw_val in enumerate(vec):
            val = float(raw_val)
            if not math.isfinite(val):
                raise InvalidEmbeddingError(
                    f"Embedding vector at index {idx} contains non-finite value at dim {coord_idx}: {raw_val!r}"
                )
            row.append(val)
        norm_sq = sum(v * v for v in row)
        if norm_sq <= min_norm_sq:
            raise InvalidEmbeddingError(
                f"Zero-norm or degenerate embedding vector rejected at index {idx} "
                f"(||v||^2={norm_sq:.3e} <= {min_norm_sq:.3e})."
            )
        validated.append(row)
    return validated


def format_pgvector_literal(vector: Sequence[float]) -> str:
    """Validate and serialize a dense vector into a canonical pgvector literal string (ADR-00021)."""
    validated_vec = validate_embedding_vectors([vector])[0]
    return "[" + ",".join(repr(val) for val in validated_vec) + "]"


def get_embedder_model(allow_mock: Optional[bool] = None) -> Any:
    """Lazily load and cache SentenceTransformer, failing fast unless mock mode is opted in."""
    global _model_instance, _mock_model_instance

    model_name = _get_model_name()
    if _is_mock_enabled(allow_mock):
        if _mock_model_instance is None:
            dim = _get_embedding_dim()
            logger.info("Explicit mock embedder enabled for '%s' (dim=%d).", model_name, dim)
            _mock_model_instance = _DeterministicMockEmbedder(dim=dim)
        return _mock_model_instance

    if _model_instance is not None:
        return _model_instance

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        _model_instance = SentenceTransformer(model_name)
        return _model_instance
    except Exception as exc:
        raise EmbedderUnavailableError(
            f"Failed to load SentenceTransformer model '{model_name}': {exc}. "
            "Ensure `sentence-transformers` and model weights are installed, or explicitly set "
            "`allow_mock=True` / `YARNBALL_EMBEDDER_MOCK=1` for offline mock tests."
        ) from exc


def embed_texts(texts: List[str], allow_mock: Optional[bool] = None) -> List[List[float]]:
    """Return validated, non-zero embedding vectors for the supplied texts."""
    if not texts:
        return []
    model = get_embedder_model(allow_mock=allow_mock)
    result = model.encode(texts, normalize_embeddings=True)
    raw_vectors: List[List[float]] = (
        result.tolist() if hasattr(result, "tolist") else [list(row) for row in result]
    )
    return validate_embedding_vectors(raw_vectors)


def verify_embedder_ready(
    allow_mock: Optional[bool] = None,
    probe_text: str = "YarnBall embedder readiness probe",
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Probe embedder availability and zero-vector integrity at startup or health checks."""
    model_name = _get_model_name()
    use_mock = _is_mock_enabled(allow_mock)
    try:
        probe_vecs = embed_texts([probe_text], allow_mock=allow_mock)
        validated = validate_embedding_vectors(probe_vecs)
        return {
            "healthy": True,
            "ready": True,
            "model": model_name,
            "dim": len(validated[0]),
            "mock": use_mock,
            "detail": "mock-enabled" if use_mock else "ready",
        }
    except Exception as exc:
        if raise_on_error:
            raise
        return {
            "healthy": False,
            "ready": False,
            "model": model_name,
            "dim": 0,
            "mock": use_mock,
            "detail": str(exc),
        }
