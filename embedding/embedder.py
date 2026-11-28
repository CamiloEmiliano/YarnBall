# embedding/embedder.py
# -*- coding: utf-8 -*-
"""Dense vector embedding service supporting Qwen3-Embedding-0.6B via Ollama and SentenceTransformers (ADR-00020 / ADR-00035).

Defaults (ADR-00035):
- Model: `qwen3-embedding:0.6b` (32K context, native 1024 dims, Apache 2.0)
- Serving: Ollama `/api/embed` pinned to CPU (`options: {"num_gpu": 0}`) during live serving so
  `yarnball-qwen:7b` retains 100% of 8 GB VRAM with zero model eviction, with optional
  `batch_gpu=True` for offline Phase-2 batch embedding after unloading the generative model.
- Explicit asymmetric entry points: `embed_query(text)` (instruction-prefixed) and `embed_documents(texts)`.
- Enforces fail-fast initialization on unreachable/missing backends and prohibits zero-norm or
  non-finite embedding vectors from reaching pgvector storage or queries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import urllib.request

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "qwen3-embedding:0.6b"
_DEFAULT_EMBEDDING_DIM = 1024
_MIN_NORM_SQ = 1e-8

QUERY_INSTRUCTION_PREFIX = (
    "Instruct: Given a financial supply-chain, SEC regulatory filing, or corporate disruption query, "
    "retrieve relevant entities, filings, and evidence passages.\nQuery: "
)

_model_instance: Any = None
_model_cache_key: Optional[tuple[str, bool, Optional[int]]] = None
_mock_model_instance: Any = None
_mock_model_dim: Optional[int] = None


class EmbedderUnavailableError(RuntimeError):
    """Raised when the configured embedder backend cannot be loaded/reached and mock mode is off."""


class InvalidEmbeddingError(ValueError):
    """Raised when an embedding vector is empty, non-finite (NaN/Inf), or zero-norm."""


def reset_embedder_cache() -> None:
    """Reset cached embedder instances (intended for deterministic unit tests)."""
    global _model_instance, _model_cache_key, _mock_model_instance, _mock_model_dim
    _model_instance = None
    _model_cache_key = None
    _mock_model_instance = None
    _mock_model_dim = None


def _get_model_name() -> str:
    raw = os.getenv("EMBEDDER_MODEL", _DEFAULT_MODEL)
    if not raw:
        return _DEFAULT_MODEL
    cleaned = raw.strip().strip('"').strip("'")
    return cleaned or _DEFAULT_MODEL


def get_embedding_dim() -> int:
    """Resolve and validate positive configured EMBEDDING_DIM (default 1024 for qwen3-embedding:0.6b)."""
    raw = os.getenv("EMBEDDING_DIM", str(_DEFAULT_EMBEDDING_DIM)).strip()
    if not raw:
        return _DEFAULT_EMBEDDING_DIM
    try:
        dim = int(raw)
    except ValueError as exc:
        raise InvalidEmbeddingError(f"Invalid EMBEDDING_DIM='{raw}': expected positive integer.") from exc
    if dim <= 0:
        raise InvalidEmbeddingError(f"Invalid EMBEDDING_DIM={dim}: expected positive integer.")
    return dim


def _get_embedding_dim() -> int:
    return get_embedding_dim()


def _is_mock_enabled(allow_mock: Optional[bool] = None) -> bool:
    if allow_mock is not None:
        return bool(allow_mock)
    env_flag = os.getenv("YARNBALL_EMBEDDER_MOCK", "").strip().lower()
    return env_flag in ("1", "true", "yes", "on")


def _resolve_mrl_dim(mrl_dim: Optional[int] = None) -> Optional[int]:
    """Resolve optional Matryoshka (MRL) target dimension from explicit arg or EMBEDDER_MRL_TRUNCATE (ADR-00047)."""
    if mrl_dim is not None:
        dim = int(mrl_dim)
        if dim <= 0:
            raise InvalidEmbeddingError(f"Invalid mrl_dim={mrl_dim}: expected positive integer.")
        return dim
    env_flag = os.getenv("EMBEDDER_MRL_TRUNCATE", "").strip().lower()
    if env_flag in ("1", "true", "yes", "on"):
        return get_embedding_dim()
    return None


def _uses_sentence_transformers(model_name: str) -> bool:
    """Return True when `model_name` specifies a HuggingFace / SentenceTransformers repo path."""
    return "/" in model_name or model_name.startswith("sentence-transformers")


def _resolve_ollama_embed_url() -> str:
    from sft_pipeline.serving.model_client import resolve_ollama_urls

    base_url, _ = resolve_ollama_urls(os.getenv("OLLAMA_EMBED_API_BASE") or os.getenv("OLLAMA_API_BASE"))
    return f"{base_url}/api/embed"


def unload_ollama_model(model_tag: Optional[str] = None, timeout_sec: float = 10.0) -> bool:
    """Request Ollama to unload a generative model (`keep_alive: 0`) prior to GPU batch embedding."""
    from sft_pipeline.serving.model_client import DEFAULT_MODEL_TAG, resolve_ollama_urls

    tag = model_tag or os.getenv("YARNBALL_MODEL_TAG") or DEFAULT_MODEL_TAG
    _, generate_url = resolve_ollama_urls()
    payload = json.dumps({"model": tag, "prompt": "", "keep_alive": 0, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        generate_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec):
            return True
    except Exception as exc:
        logger.warning("Could not unload Ollama model '%s' before batch embedding: %s", tag, exc)
        return False


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


class _OllamaEmbedderClient:
    """Ollama `/api/embed` client pinned to CPU (`num_gpu: 0`) for live serving (ADR-00035)."""

    is_mock: bool = False

    def __init__(
        self,
        model_name: str,
        *,
        embed_url: Optional[str] = None,
        num_gpu: Optional[int] = 0,
        timeout_sec: float = 60.0,
        dimensions: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.embed_url = embed_url or _resolve_ollama_embed_url()
        self.num_gpu = num_gpu
        self.timeout_sec = timeout_sec
        self.dimensions = int(dimensions) if dimensions is not None else None

    def build_request_payload(self, texts: Sequence[str]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": list(texts),
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if self.num_gpu is not None:
            payload["options"] = {"num_gpu": int(self.num_gpu)}
        return payload

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True) -> List[List[float]]:
        if not texts:
            return []
        payload = self.build_request_payload(texts)
        req = urllib.request.Request(
            self.embed_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise EmbedderUnavailableError(
                f"Failed to query Ollama embedder '{self.model_name}' at '{self.embed_url}': {exc}. "
                "Ensure Ollama is running with `ollama pull qwen3-embedding:0.6b`, or set "
                "`YARNBALL_EMBEDDER_MOCK=1` for offline mock tests."
            ) from exc

        embeddings = body.get("embeddings")
        if embeddings is None and "embedding" in body:
            embeddings = [body["embedding"]]
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbedderUnavailableError(
                f"Malformed Ollama /api/embed response for '{self.model_name}': "
                f"expected {len(texts)} vectors, got {type(embeddings).__name__}."
            )

        out: List[List[float]] = []
        for vec in embeddings:
            floats = [float(v) for v in vec]
            if normalize_embeddings:
                norm_sq = sum(v * v for v in floats)
                if norm_sq > _MIN_NORM_SQ:
                    norm = math.sqrt(norm_sq)
                    floats = [v / norm for v in floats]
            out.append(floats)
        return out


def truncate_and_renormalize_mrl(
    vectors: Sequence[Sequence[float]],
    target_dim: int,
    *,
    min_norm_sq: float = _MIN_NORM_SQ,
) -> List[List[float]]:
    """Slice raw vectors to `target_dim` via Matryoshka Representation Learning (MRL) and L2 re-normalize (ADR-00047)."""
    dim = int(target_dim)
    if dim <= 0:
        raise InvalidEmbeddingError(f"Invalid MRL target_dim={target_dim}: expected positive integer.")

    projected: List[List[float]] = []
    for idx, vec in enumerate(vectors):
        if vec is None or len(vec) < dim:
            actual = 0 if vec is None else len(vec)
            raise InvalidEmbeddingError(
                f"MRL truncation impossible at index {idx}: vector dim={actual} < target_dim={dim}."
            )
        prefix: List[float] = []
        for coord_idx in range(dim):
            val = float(vec[coord_idx])
            if not math.isfinite(val):
                raise InvalidEmbeddingError(
                    f"Non-finite value at index {idx} dim {coord_idx} during MRL truncation: {vec[coord_idx]!r}"
                )
            prefix.append(val)
        norm_sq = sum(v * v for v in prefix)
        if norm_sq <= min_norm_sq:
            raise InvalidEmbeddingError(
                f"Zero-norm MRL prefix rejected at index {idx} (dim={dim}, ||v||^2={norm_sq:.3e} <= {min_norm_sq:.3e})."
            )
        norm = math.sqrt(norm_sq)
        projected.append([v / norm for v in prefix])
    return projected


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


def get_embedder_model(
    allow_mock: Optional[bool] = None,
    *,
    batch_gpu: bool = False,
    model_name: Optional[str] = None,
    mrl_dim: Optional[int] = None,
) -> Any:
    """Lazily resolve and cache the configured embedder (Ollama Qwen3-Embedding or SentenceTransformer)."""
    global _model_instance, _model_cache_key, _mock_model_instance, _mock_model_dim

    resolved_model = (model_name or "").strip() or _get_model_name()
    resolved_mrl = _resolve_mrl_dim(mrl_dim)
    if _is_mock_enabled(allow_mock):
        dim = resolved_mrl if resolved_mrl is not None else _get_embedding_dim()
        if _mock_model_instance is None or _mock_model_dim != dim:
            logger.info("Explicit mock embedder enabled for '%s' (dim=%d).", resolved_model, dim)
            _mock_model_instance = _DeterministicMockEmbedder(dim=dim)
            _mock_model_dim = dim
        return _mock_model_instance

    cache_key = (resolved_model, bool(batch_gpu), resolved_mrl)
    if _model_instance is not None and _model_cache_key == cache_key:
        return _model_instance

    if _uses_sentence_transformers(resolved_model):
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading SentenceTransformer embedding model: %s", resolved_model)
            _model_instance = SentenceTransformer(resolved_model)
            _model_cache_key = cache_key
            return _model_instance
        except Exception as exc:
            raise EmbedderUnavailableError(
                f"Failed to load SentenceTransformer model '{resolved_model}': {exc}. "
                "Ensure `sentence-transformers` and model weights are installed, or explicitly set "
                "`allow_mock=True` / `YARNBALL_EMBEDDER_MOCK=1` for offline mock tests."
            ) from exc

    logger.info(
        "Configuring Ollama embedder '%s' (batch_gpu=%s, num_gpu=%s, mrl_dim=%s)",
        resolved_model,
        batch_gpu,
        "auto" if batch_gpu else "0",
        resolved_mrl,
    )
    _model_instance = _OllamaEmbedderClient(
        model_name=resolved_model,
        num_gpu=None if batch_gpu else 0,
        dimensions=resolved_mrl,
    )
    _model_cache_key = cache_key
    return _model_instance


def embed_texts(
    texts: List[str],
    allow_mock: Optional[bool] = None,
    *,
    batch_gpu: bool = False,
    model_name: Optional[str] = None,
    mrl_dim: Optional[int] = None,
) -> List[List[float]]:
    """Return validated, non-zero embedding vectors for the supplied texts (with optional MRL projection)."""
    if not texts:
        return []
    resolved_mrl = _resolve_mrl_dim(mrl_dim)
    model = get_embedder_model(
        allow_mock=allow_mock,
        batch_gpu=batch_gpu,
        model_name=model_name,
        mrl_dim=resolved_mrl,
    )
    result = model.encode(texts, normalize_embeddings=True)
    raw_vectors: List[List[float]] = (
        result.tolist() if hasattr(result, "tolist") else [list(row) for row in result]
    )
    if resolved_mrl is not None:
        raw_vectors = truncate_and_renormalize_mrl(raw_vectors, resolved_mrl)
    return validate_embedding_vectors(raw_vectors)


def embed_documents(
    texts: Sequence[str],
    *,
    allow_mock: Optional[bool] = None,
    batch_gpu: bool = False,
    model_name: Optional[str] = None,
    mrl_dim: Optional[int] = None,
) -> List[List[float]]:
    """Embed document chunks, entity descriptors, or news passages without query instructions (ADR-00035)."""
    text_list = [str(t) for t in texts]
    if not text_list:
        return []
    if allow_mock is None and not batch_gpu and model_name is None and mrl_dim is None:
        return embed_texts(text_list)
    return embed_texts(
        text_list,
        allow_mock=allow_mock,
        batch_gpu=batch_gpu,
        model_name=model_name,
        mrl_dim=mrl_dim,
    )


def embed_query(
    text: str,
    *,
    instruction: str = QUERY_INSTRUCTION_PREFIX,
    allow_mock: Optional[bool] = None,
    batch_gpu: bool = False,
    model_name: Optional[str] = None,
    mrl_dim: Optional[int] = None,
) -> List[float]:
    """Embed an asymmetric retrieval query with financial retrieval instruction prefix (ADR-00035)."""
    prefixed = f"{instruction}{str(text).strip()}"
    if allow_mock is None and not batch_gpu and model_name is None and mrl_dim is None:
        vectors = embed_texts([prefixed])
    else:
        vectors = embed_texts(
            [prefixed],
            allow_mock=allow_mock,
            batch_gpu=batch_gpu,
            model_name=model_name,
            mrl_dim=mrl_dim,
        )
    if not vectors:
        raise InvalidEmbeddingError("Query embedding returned empty list.")
    return vectors[0]


def verify_embedder_ready(
    allow_mock: Optional[bool] = None,
    probe_text: str = "YarnBall embedder readiness probe",
    raise_on_error: bool = False,
    expected_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Probe embedder availability, zero-vector integrity, and EMBEDDING_DIM alignment."""
    model_name = _get_model_name()
    use_mock = _is_mock_enabled(allow_mock)
    target_dim: Optional[int] = expected_dim
    try:
        if target_dim is None:
            target_dim = get_embedding_dim()
        probe_vecs = embed_texts([probe_text], allow_mock=allow_mock)
        validated = validate_embedding_vectors(probe_vecs)
        actual_dim = len(validated[0])
        if actual_dim != target_dim:
            raise InvalidEmbeddingError(
                f"Embedder dimension mismatch for '{model_name}': "
                f"model emitted dim={actual_dim}, expected EMBEDDING_DIM={target_dim}."
            )
        return {
            "healthy": True,
            "ready": True,
            "model": model_name,
            "dim": actual_dim,
            "expected_dim": target_dim,
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
            "expected_dim": target_dim,
            "mock": use_mock,
            "detail": str(exc),
        }
