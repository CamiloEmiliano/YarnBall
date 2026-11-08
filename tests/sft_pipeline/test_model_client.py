"""Unit tests for Stage 6.4: YarnBallModelClient."""

import pytest
from sft_pipeline.serving.model_client import (
    YarnBallModelClient,
    ModelResponse,
    TASK_MAX_NEW_TOKENS,
)


def test_prefix_forcing_reasoning_tasks():
    captured_payloads = []

    def mock_transport(payload):
        captured_payloads.append(payload)
        return {
            "response": "Step 1: Upstream analysis.</think>\nClassification: DISRUPTIVE_SHOCK.",
            "prompt_eval_count": 120,
            "prompt_eval_duration": 50000000,
            "eval_count": 45,
            "eval_duration_ns": 600000000,
        }

    client = YarnBallModelClient(transport_fn=mock_transport)

    # 1. Reasoning task (Task D) -> should have <think>\n prefix in assistant turn
    res_d = client.generate("Trace contagion path", task="CONTAGION_REASONING")
    assert "<|im_start|>assistant\n<think>\n" in captured_payloads[0]["prompt"]
    assert res_d.think_trace == "Step 1: Upstream analysis."
    assert res_d.final_output == "Classification: DISRUPTIVE_SHOCK."
    assert res_d.runaway_detected is False
    assert res_d.tokens_per_sec > 0

    # 2. Non-reasoning task (Task C) -> should NOT have <think>\n prefix
    captured_payloads.clear()
    res_c = client.generate("MATCH (c:Company) RETURN c", task="TEXT_TO_CYPHER")
    assert "<|im_start|>assistant\n<think>\n" not in captured_payloads[0]["prompt"]
    assert captured_payloads[0]["options"]["num_predict"] == 256


def test_output_demuxing_clean_split():
    raw_text = (
        "<think>\n"
        "1. Direct supplier halts 40%.\n"
        "2. Downstream customer lacks alternate capacity.\n"
        "</think>\n"
        "**Conclusion**: `DISRUPTIVE_SHOCK`."
    )
    think, final = YarnBallModelClient.demux_cot(raw_text, is_reasoning_task=True)
    assert think == "1. Direct supplier halts 40%.\n2. Downstream customer lacks alternate capacity."
    assert final == "**Conclusion**: `DISRUPTIVE_SHOCK`."


def test_output_demuxing_unclosed_think():
    raw_text = "<think>\n1. Analyzing supply chain... (interrupted)"
    think, final = YarnBallModelClient.demux_cot(raw_text, is_reasoning_task=True)
    assert "Analyzing supply chain" in think
    assert final == "(unclosed reasoning)"


def test_output_demuxing_non_reasoning_task():
    raw_cypher = "MATCH (c:Company {ticker: 'AAPL'}) RETURN c.name"
    think, final = YarnBallModelClient.demux_cot(raw_cypher, is_reasoning_task=False)
    assert think is None
    assert final == raw_cypher


def test_runaway_loop_detection_watchdog():
    # Construct a repeating 4-gram loop
    base_text = "The shock propagates from vendor to vendor"
    repeating_pattern = "curtailing production and margins"
    runaway_text = f"{base_text} {repeating_pattern} {repeating_pattern} {repeating_pattern} {repeating_pattern} {repeating_pattern}"

    truncated, detected = YarnBallModelClient.detect_and_truncate_runaway(runaway_text, min_n=4, max_repeats=4)
    assert detected is True
    assert len(truncated.split()) < len(runaway_text.split())


def test_normal_text_does_not_trigger_runaway_watchdog():
    normal_text = (
        "Alpha Corp supplies 40% of substrates to Beta Ltd. "
        "Beta Ltd holds 60 days of inventory and can absorb short outages without production halts. "
        "Downstream deliveries to Gamma Inc remain fully buffered."
    )
    truncated, detected = YarnBallModelClient.detect_and_truncate_runaway(normal_text)
    assert detected is False
    assert truncated == normal_text


def test_task_token_caps_assigned():
    assert TASK_MAX_NEW_TOKENS["TEXT_TO_CYPHER"] == 256
    assert TASK_MAX_NEW_TOKENS["C"] == 256
    assert TASK_MAX_NEW_TOKENS["CONTAGION_REASONING"] == 768
    assert TASK_MAX_NEW_TOKENS["D"] == 768
    assert TASK_MAX_NEW_TOKENS["EXTRACT_SEC_GRAPH"] == 512


def test_http_error_handling_graceful():
    def failing_transport(payload):
        raise ConnectionRefusedError("Ollama endpoint unreachable at 127.0.0.1:11434")

    client = YarnBallModelClient(transport_fn=failing_transport)
    response = client.generate("Test prompt", task="CONTAGION_REASONING")

    assert response.error is not None
    assert "unreachable" in response.error.lower()
    assert response.final_output == "(none)"
    assert response.latency_sec >= 0.0

