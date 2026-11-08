"""Unified Model Client with Prefix Forcing and Mode-Bleed Guardrails.

Wraps local Ollama / vLLM serving runtimes for `yarnball-qwen:7b`:
1. Prefix Forcing: Injects `<think>\\n` into prompt conditioning for reasoning tasks
   (Tasks D and E) to eliminate first-token Cypher mode bleed.
2. Output Demuxer: Splits raw generation on `</think>` into `think_trace` and `final_output`.
3. Task-Specific Token Budgets: Enforces strict caps (256–768 tokens) per operational task.
4. Runaway Generation Watchdog: Detects repeating n-gram loops and truncates runaway output.
5. Hardware & Latency Telemetry: Tracks prefill ms, decode tok/s, and token counts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("model_client")

DEFAULT_SYSTEM_PROMPT = (
    "You are YarnBall, an expert financial intelligence engine combining SEC regulatory disclosures, "
    "knowledge-graph supply-chain topology, and real-time market microstructure."
)

TASK_MAX_NEW_TOKENS: Dict[str, int] = {
    # Full task names
    "EXTRACT_SEC_GRAPH": 512,
    "EXTRACT_NEWS_EVENT": 512,
    "TEXT_TO_CYPHER": 256,
    "CONTAGION_REASONING": 768,
    "PORTFOLIO_RECOMMENDATION": 512,
    # Shorthand aliases
    "A": 512,
    "B": 512,
    "C": 256,
    "D": 768,
    "E": 512,
}

TASK_NORMALIZATION: Dict[str, str] = {
    "A": "EXTRACT_SEC_GRAPH",
    "B": "EXTRACT_NEWS_EVENT",
    "C": "TEXT_TO_CYPHER",
    "D": "CONTAGION_REASONING",
    "E": "PORTFOLIO_RECOMMENDATION",
}


@dataclass
class ModelResponse:
    """Structured response from the YarnBall model engine."""

    raw_text: str
    think_trace: Optional[str]
    final_output: str
    task: str
    latency_sec: float
    prefill_ms: float
    tokens_per_sec: float
    prompt_tokens: int
    generated_tokens: int
    runaway_detected: bool = False
    error: Optional[str] = None


class YarnBallModelClient:
    """Production inference client for local yarnball-qwen:7b with guardrails."""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_tag: Optional[str] = None,
        timeout_sec: float = 120.0,
        transport_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        base_url = ollama_url or os.getenv("OLLAMA_API_BASE") or "http://127.0.0.1:11434"
        if not base_url.endswith("/api/generate"):
            self.ollama_url = f"{base_url.rstrip('/')}/api/generate"
        else:
            self.ollama_url = base_url

        self.model_tag = model_tag or os.getenv("YARNBALL_MODEL_TAG", "yarnball-qwen:7b")
        self.timeout_sec = timeout_sec
        self._transport_fn = transport_fn  # Optional mock transport for testing

    def generate(
        self,
        prompt: str,
        task: str = "CONTAGION_REASONING",
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: Optional[int] = None,
        enforce_think_prefix: bool = True,
        num_ctx: int = 8192,
    ) -> ModelResponse:
        """Execute inference with prefix forcing, watchdog, and output demuxing.

        Args:
            prompt: User-turn text or instruction.
            task: Task type (A-E or full name).
            system_prompt: Optional custom system prompt.
            temperature: Sampling temperature.
            max_new_tokens: Optional explicit token cap.
            enforce_think_prefix: Whether to inject `<think>\\n` into the assistant turn.
            num_ctx: Ollama context window size.

        Returns:
            Structured ModelResponse with separated CoT trace and telemetry.
        """
        norm_task = TASK_NORMALIZATION.get(task.upper(), task.upper())
        token_cap = max_new_tokens or TASK_MAX_NEW_TOKENS.get(norm_task, 512)
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # 1. Prefix Forcing: Inject <think>\n on reasoning tasks to eliminate mode-bleed
        is_reasoning_task = norm_task in ("CONTAGION_REASONING", "PORTFOLIO_RECOMMENDATION")
        assistant_prefix = "<think>\n" if (enforce_think_prefix and is_reasoning_task) else ""

        raw_chatml = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_prefix}"
        )

        payload = {
            "model": self.model_tag,
            "prompt": raw_chatml,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_ctx": num_ctx,
                "num_predict": token_cap,
                "stop": ["<|im_end|>", "<|endoftext|>"],
            },
        }

        t0 = time.time()
        try:
            if self._transport_fn:
                body = self._transport_fn(payload)
            else:
                body = self._post_http(payload)
            wall_elapsed = time.time() - t0
        except Exception as exc:
            return ModelResponse(
                raw_text="",
                think_trace=None,
                final_output="(none)",
                task=norm_task,
                latency_sec=round(time.time() - t0, 3),
                prefill_ms=0.0,
                tokens_per_sec=0.0,
                prompt_tokens=0,
                generated_tokens=0,
                error=str(exc),
            )

        raw_response = body.get("response", "")
        full_pred = f"{assistant_prefix}{raw_response}".strip()

        # 2. Runaway Loop Watchdog
        sanitized_pred, runaway_detected = self.detect_and_truncate_runaway(full_pred)

        # 3. Output Demuxer: Separate <think> from final synthesis
        think_trace, final_synthesis = self.demux_cot(sanitized_pred, is_reasoning_task)

        # 4. Telemetry Calculation
        prompt_tokens = int(body.get("prompt_eval_count") or max(1, len(raw_chatml.split())))
        prefill_ns = int(body.get("prompt_eval_duration") or 0)
        prefill_ms = round(prefill_ns / 1e6, 2) if prefill_ns > 0 else 0.0

        eval_count = int(body.get("eval_count") or max(1, len(sanitized_pred.split())))
        eval_duration_ns = int(body.get("eval_duration") or 0)
        if eval_duration_ns > 0:
            tok_per_sec = eval_count / (eval_duration_ns / 1e9)
        else:
            tok_per_sec = eval_count / wall_elapsed if wall_elapsed > 0 else 0.0

        return ModelResponse(
            raw_text=sanitized_pred,
            think_trace=think_trace,
            final_output=final_synthesis,
            task=norm_task,
            latency_sec=round(wall_elapsed, 3),
            prefill_ms=prefill_ms,
            tokens_per_sec=round(tok_per_sec, 2),
            prompt_tokens=prompt_tokens,
            generated_tokens=eval_count,
            runaway_detected=runaway_detected,
        )

    def _post_http(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP POST to Ollama API."""
        req = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            data: Dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return data

    @staticmethod
    def demux_cot(
        text: str, is_reasoning_task: bool = True
    ) -> Tuple[Optional[str], str]:
        """Split text on `</think>` into reasoning block and final synthesis."""
        clean = text.strip()
        if not is_reasoning_task:
            return None, clean

        if "<think>" in clean and "</think>" in clean:
            parts = clean.split("</think>", 1)
            think_content = parts[0].replace("<think>", "").strip()
            final_content = parts[1].strip()
            return think_content, final_content

        if "<think>" in clean and "</think>" not in clean:
            # Model truncated mid-thought without finishing synthesis
            think_content = clean.replace("<think>", "").strip()
            return think_content, "(unclosed reasoning)"

        return None, clean

    @staticmethod
    def detect_and_truncate_runaway(
        text: str, min_n: int = 4, max_repeats: int = 4
    ) -> Tuple[str, bool]:
        """Detect repeating word n-gram loops and truncate generation."""
        words = text.split()
        if len(words) < min_n * max_repeats:
            return text, False

        # Check for 3-gram, 4-gram, and 5-gram loops
        for n in range(min_n, 8):
            for i in range(len(words) - n * max_repeats + 1):
                pattern = words[i : i + n]
                repeats = 1
                curr_idx = i + n
                while curr_idx + n <= len(words) and words[curr_idx : curr_idx + n] == pattern:
                    repeats += 1
                    curr_idx += n
                    if repeats >= max_repeats:
                        truncated_words = words[:curr_idx]
                        logger.warning(
                            "Runaway n-gram loop detected (pattern: '%s', repeats: %d). Truncating.",
                            " ".join(pattern),
                            repeats,
                        )
                        return " ".join(truncated_words), True

        return text, False

