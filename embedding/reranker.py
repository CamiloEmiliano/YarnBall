"""Pluggable Second-Stage Cross-Encoder Reranker Protocol (ADR-00047).

Provides a zero-internal-dependency reranking interface (`RerankerPort`) for evaluating
Arm `E3` (`qwen3-embedding:0.6b` Hybrid RRF top-M -> compact cross-encoder) against larger
first-stage bi-encoders (`E2` 4B MRL, `E2b` 8B MRL):
- Default local backend: `Qwen/Qwen3-Reranker-0.6B` (configurable via `RERANKER_MODEL`
  to alternatives such as `BAAI/bge-reranker-v2-m3`), CPU-pinned by default for live serving.
- Explicit deterministic mock backend (`DeterministicMockReranker`, opt-in via
  `allow_mock=True` or `YARNBALL_RERANKER_MOCK=1`) for offline CI and unit tests.
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL: str = "Qwen/Qwen3-Reranker-0.6B"
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_reranker_instance: Optional[RerankerPort] = None
_reranker_cache_key: Optional[Tuple[str, str]] = None


class RerankerUnavailableError(RuntimeError):
    """Raised when the configured cross-encoder reranker cannot be loaded and mock mode is off."""


@runtime_checkable
class RerankerPort(Protocol):
    """Protocol implemented by all second-stage query-passage cross-encoder rerankers."""

    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]:
        """Return one relevance score per passage in the same order as `passages`."""
        ...


def reset_reranker_cache() -> None:
    """Reset cached reranker instances (intended for deterministic unit tests)."""
    global _reranker_instance, _reranker_cache_key
    _reranker_instance = None
    _reranker_cache_key = None


def _get_reranker_model_name(model_name: Optional[str] = None) -> str:
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    raw = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    cleaned = str(raw or "").strip().strip('"').strip("'")
    return cleaned or DEFAULT_RERANKER_MODEL


def _is_reranker_mock_enabled(allow_mock: Optional[bool] = None) -> bool:
    if allow_mock is not None:
        return bool(allow_mock)
    env_flag = os.getenv("YARNBALL_RERANKER_MOCK", "").strip().lower()
    return env_flag in ("1", "true", "yes", "on")


class DeterministicMockReranker:
    """Offline lexical + bigram overlap reranker emitting deterministic finite scores in [0, 1]."""

    is_mock: bool = True

    def __init__(self, model_name: str = "mock-cross-encoder") -> None:
        self.model_name = model_name

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(str(text or "").lower())

    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        q_tokens = self._tokenize(query)
        q_set = set(q_tokens)
        q_bigrams = set(zip(q_tokens, q_tokens[1:])) if len(q_tokens) >= 2 else set()
        q_phrase = " ".join(q_tokens)

        scores: List[float] = []
        for passage in passages:
            p_tokens = self._tokenize(passage)
            if not q_set or not p_tokens:
                scores.append(0.0)
                continue
            p_set = set(p_tokens)
            unigram_recall = len(q_set & p_set) / float(len(q_set))
            p_bigrams = set(zip(p_tokens, p_tokens[1:])) if len(p_tokens) >= 2 else set()
            bigram_recall = (len(q_bigrams & p_bigrams) / float(len(q_bigrams))) if q_bigrams else 0.0
            exact_bonus = 0.15 if (q_phrase and q_phrase in " ".join(p_tokens)) else 0.0
            length_damp = 1.0 / (1.0 + 0.02 * math.log1p(len(p_tokens)))
            raw = (0.65 * unigram_recall + 0.35 * bigram_recall + exact_bonus) * length_damp
            scores.append(round(min(1.0, max(0.0, raw)), 6))
        return scores


class LocalCrossEncoderReranker:
    """Lazy `sentence_transformers.CrossEncoder` reranker pinned to CPU by default for live serving."""

    is_mock: bool = False

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading local CrossEncoder reranker '%s' on device='%s'", self.model_name, self.device)
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            return self._model
        except Exception as exc:
            raise RerankerUnavailableError(
                f"Failed to load local CrossEncoder reranker '{self.model_name}' (device={self.device}): {exc}. "
                "Install `sentence-transformers` and download the model weights, or set "
                "`YARNBALL_RERANKER_MOCK=1` / `allow_mock=True` for offline mock evaluation."
            ) from exc

    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]:
        if not passages:
            return []
        model = self._load()
        pairs = [(str(query), str(p)) for p in passages]
        raw_scores = model.predict(pairs)
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)
        out: List[float] = []
        for idx, raw_val in enumerate(values):
            val = float(raw_val)
            if not math.isfinite(val):
                raise ValueError(f"CrossEncoder emitted non-finite rerank score at index {idx}: {raw_val!r}")
            out.append(val)
        return out


def get_reranker(
    model_name: Optional[str] = None,
    *,
    allow_mock: Optional[bool] = None,
    device: str = "cpu",
) -> RerankerPort:
    """Resolve and cache the configured second-stage reranker (fail-fast unless mock mode is opted in)."""
    global _reranker_instance, _reranker_cache_key

    resolved_model = _get_reranker_model_name(model_name)
    if _is_reranker_mock_enabled(allow_mock):
        return DeterministicMockReranker(model_name=f"mock:{resolved_model}")

    cache_key = (resolved_model, str(device).strip().lower() or "cpu")
    if _reranker_instance is not None and _reranker_cache_key == cache_key:
        return _reranker_instance

    reranker = LocalCrossEncoderReranker(model_name=resolved_model, device=cache_key[1])
    _reranker_instance = reranker
    _reranker_cache_key = cache_key
    return reranker


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "DeterministicMockReranker",
    "LocalCrossEncoderReranker",
    "RerankerPort",
    "RerankerUnavailableError",
    "get_reranker",
    "reset_reranker_cache",
]
