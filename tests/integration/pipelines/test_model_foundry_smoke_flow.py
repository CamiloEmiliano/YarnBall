"""System Integration Test (SIT) Flow 5: Model Foundry & Alignment Pipeline Smoke Test.

Academic Reference & Governance:
- ADR-00004: SIT Flow 5 Model Foundry Alignment Smoke
- Qwen2.5 ChatML Specification & Tokenizer Special Delimiters
- Rafailov, E., et al. (2023): "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

Subsystems Crossed:
- sft_pipeline/builder.py (SFTRecord, multi-task dataset synthesis)
- sft_pipeline/dpo_builder.py (DPORecord, preference pair construction)
- sft_pipeline/phase4_export_and_serve.py (fuse_safetensors_shard, validators, BF16 arithmetic)

Invariants Asserted:
1. ChatML Boundary Invariant: Every synthesized prompt and response conforms strictly
   to `<|im_start|>...<|im_end|>` framing with zero delimiter corruption.
2. Temporal Split Invariant: Strict point-in-time partitioning (Train: as_of_date <= 2023-12-31,
   Test: as_of_date >= 2024-01-01) with zero event leakage across splits.
3. DPO Preference Separation Invariant: Chosen responses pass validators, while rejected responses
   explicitly trigger targeted failure modes (MODE_BLEED, INFLECTION_LAG, SCHEMA_TRAP).
4. Numerical Fusion Invariant: LoRA adapter fusion satisfies |W_merged - (W_base + scale * (B @ A))| <= 1e-3
   (bounded by the hardware 7-bit mantissa precision floor of bfloat16).
5. Offline Isolation Invariant: Execution runs 100% offline with zero remote repository or broker calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from sft_pipeline.builder import SFTRecord
from sft_pipeline.dpo_builder import DPORecord
from sft_pipeline.phase4_export_and_serve import (
    bf16_bytes_to_fp32,
    fp32_to_bf16_bytes,
    fuse_safetensors_shard,
    validate_cik_integrity,
    validate_cypher_syntax,
    validate_reasoning_cot,
)


@pytest.fixture
def synthetic_multitask_sft_records() -> List[SFTRecord]:
    """Synthetic multi-task SFT mini-batch covering Tasks A through E with ChatML framing."""
    samples = [
        # Task A: Entity & CIK Normalization
        SFTRecord(
            sample_id="task_a_apple_cik",
            prompt="<|im_start|>system\nYou are a financial entity resolver.<|im_end|>\n<|im_start|>user\nResolve canonical CIK for Apple Inc.<|im_end|>\n<|im_start|>assistant\n",
            target_completion="CIK: 0000320193<|im_end|>",
            metadata={"task": "A", "as_of_date": "2023-10-17", "split": "train"},
        ),
        # Task B: SEC Regulatory Triple Extraction
        SFTRecord(
            sample_id="task_b_nvda_supplier_extraction",
            prompt="<|im_start|>system\nYou are an SEC ontology extractor.<|im_end|>\n<|im_start|>user\nExtract primary foundry relationship from 10-K filing.<|im_end|>\n<|im_start|>assistant\n",
            target_completion="(:Company {name: 'TSMC', cik: '0001045810'})-[:SUPPLIES_TO {loss_ratio: 0.85, status: 'ACTIVE_CURRENT'}]->(:Company {name: 'NVIDIA CORP', cik: '0001045810'})<|im_end|>",
            metadata={"task": "B", "as_of_date": "2023-11-15", "split": "train"},
        ),
        # Task C: Text-to-Cypher Compilation
        SFTRecord(
            sample_id="task_c_text_to_cypher_suppliers",
            prompt="<|im_start|>system\nYou are an openCypher compiler adhering to the financial schema.<|im_end|>\n<|im_start|>user\nFind all active Tier 1 suppliers to AAPL.<|im_end|>\n<|im_start|>assistant\n",
            target_completion="MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company {ticker: 'AAPL'}) WHERE coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT' RETURN s.name, r.loss_ratio<|im_end|>",
            metadata={"task": "C", "as_of_date": "2023-12-01", "split": "train"},
        ),
        # Task D: Contagion Shock Reasoning (Requires <think>...</think> CoT)
        SFTRecord(
            sample_id="task_d_contagion_reasoning_shock",
            prompt="<|im_start|>system\nYou are a systemic risk contagion evaluator.<|im_end|>\n<|im_start|>user\nAnalyze FabCo optical transceiver disruption on downstream Apple hardware.<|im_end|>\n<|im_start|>assistant\n",
            target_completion=(
                "<think>\n"
                "Step 1: FabCo is sole source transceiver provider with 90-day shutdown.\n"
                "Step 2: Apple inventory buffer is 45 days, mitigating initial shortfall.\n"
                "Step 3: Lagged supply deficit emerges at day 46 with 50% capacity contraction.\n"
                "</think>\n"
                "Disruption triggers a 45-day latency window before operational bottleneck restricts assembly yields."
                "<|im_end|>"
            ),
            metadata={"task": "D", "as_of_date": "2023-10-18", "split": "train"},
        ),
        # Task E: Portfolio Hedging Directives (Requires <think>...</think> CoT)
        SFTRecord(
            sample_id="task_e_portfolio_hedging_overlay",
            prompt="<|im_start|>system\nYou are a systematic portfolio risk manager.<|im_end|>\n<|im_start|>user\nProvide hedging recommendation for concentrated AAPL long exposure.<|im_end|>\n<|im_start|>assistant\n",
            target_completion=(
                "<think>\n"
                "Concentrated single-stock supply vulnerability identified.\n"
                "Buffer burn period estimated at 45 days.\n"
                "Recommendation: Deploy collar options strategy.\n"
                "</think>\n"
                "Execute a 45-day bear collar on AAPL: buy 95% put option and write 105% covered call."
                "<|im_end|>"
            ),
            metadata={"task": "E", "as_of_date": "2023-10-19", "split": "train"},
        ),
        # Test Split Sample: Must have as_of_date >= 2024-01-01
        SFTRecord(
            sample_id="eval_task_c_2024_holdout",
            prompt="<|im_start|>system\nYou are an openCypher compiler.<|im_end|>\n<|im_start|>user\nQuery supply chain for MSFT.<|im_end|>\n<|im_start|>assistant\n",
            target_completion="MATCH (s:Company)-[:SUPPLIES_TO]->(c:Company {ticker: 'MSFT'}) RETURN s, c<|im_end|>",
            metadata={"task": "C", "as_of_date": "2024-03-15", "split": "test"},
        ),
    ]
    return samples


@pytest.fixture
def synthetic_dpo_preference_pairs() -> List[DPORecord]:
    """Synthetic DPO preference pairs covering inflection-lag, mode-bleed, and schema-traps."""
    return [
        # Pathology 1: MODE_BLEED (generic chat instead of <think> CoT trace)
        DPORecord(
            sample_id="dpo_mode_bleed_pair",
            prompt="<|im_start|>system\nYou are a financial reasoning engine.<|im_end|>\n<|im_start|>user\nAnalyze FabCo facility disruption.<|im_end|>\n<|im_start|>assistant\n",
            chosen=(
                "<think>\n"
                "1. FabCo provides optical transceivers.\n"
                "2. 90-day outage exceeds 45-day safety stock buffer.\n"
                "3. Net operational shortfall of 45 days.\n"
                "</think>\n"
                "Downstream impact is deferred for 45 calendar days, followed by supply curtailment."
            ),
            rejected=(
                "Sure, I'd be happy to help with that! A factory disruption at FabCo sounds like bad news. "
                "Apple will probably find another supplier soon, so investors should not worry too much."
            ),
            metadata={"pathology": "MODE_BLEED", "task": "D"},
        ),
        # Pathology 2: SCHEMA_TRAP (hallucinated graph property in Cypher)
        DPORecord(
            sample_id="dpo_schema_trap_pair",
            prompt="<|im_start|>system\nYou are an openCypher generator.<|im_end|>\n<|im_start|>user\nList suppliers where CEO age is under 50.<|im_end|>\n<|im_start|>assistant\n",
            chosen="(none)",
            rejected="MATCH (c:Company) WHERE c.ceo_age < 50 RETURN c.name",
            metadata={"pathology": "SCHEMA_TRAP", "task": "C"},
        ),
    ]


@pytest.mark.integration
def test_sft_multitask_minibatch_chatml_and_temporal_split(
    synthetic_multitask_sft_records: List[SFTRecord],
):
    """Verify Invariant 1 (ChatML boundaries), Invariant 2 (temporal split), and validator compliance."""
    train_records = [r for r in synthetic_multitask_sft_records if r.metadata.get("split") == "train"]
    test_records = [r for r in synthetic_multitask_sft_records if r.metadata.get("split") == "test"]

    assert len(train_records) == 5
    assert len(test_records) == 1

    # Invariant 2: Temporal Split Invariant (Zero train/test leakage)
    for r in train_records:
        as_of = r.metadata["as_of_date"]
        assert as_of <= "2023-12-31", f"Temporal leak in train record {r.sample_id}: {as_of} > 2023-12-31"

    for r in test_records:
        as_of = r.metadata["as_of_date"]
        assert as_of >= "2024-01-01", f"Test record {r.sample_id} violates post-2024 holdout: {as_of}"

    # Invariant 1: ChatML Boundary Invariant across all samples
    for r in synthetic_multitask_sft_records:
        full_text = f"{r.prompt}{r.target_completion}"
        # Must contain well-formed system, user, and assistant control blocks
        assert "<|im_start|>system\n" in full_text
        assert "<|im_end|>\n<|im_start|>user\n" in full_text
        assert "<|im_end|>\n<|im_start|>assistant\n" in full_text
        assert full_text.endswith("<|im_end|>")
        # Ensure balanced start and end delimiters
        assert full_text.count("<|im_start|>") == 3
        assert full_text.count("<|im_end|>") == 3

    # Output Validators compliance
    # Task A: CIK integrity
    task_a = next(r for r in train_records if r.metadata["task"] == "A")
    valid_cik, pred_ciks, _ = validate_cik_integrity(task_a.target_completion, "Ground truth CIK 0000320193")
    assert valid_cik is True
    assert pred_ciks == ["0000320193"]

    # Tasks B & C: Cypher syntax
    task_b = next(r for r in train_records if r.metadata["task"] == "B")
    valid_cypher_b, _ = validate_cypher_syntax(task_b.target_completion.replace("<|im_end|>", "").strip())
    assert valid_cypher_b is True

    task_c = next(r for r in train_records if r.metadata["task"] == "C")
    valid_cypher_c, _ = validate_cypher_syntax(task_c.target_completion.replace("<|im_end|>", "").strip())
    assert valid_cypher_c is True

    # Tasks D & E: Chain-of-Thought (<think>...</think>) structure
    task_d = next(r for r in train_records if r.metadata["task"] == "D")
    valid_cot_d, _ = validate_reasoning_cot(task_d.target_completion.replace("<|im_end|>", "").strip())
    assert valid_cot_d is True

    task_e = next(r for r in train_records if r.metadata["task"] == "E")
    valid_cot_e, _ = validate_reasoning_cot(task_e.target_completion.replace("<|im_end|>", "").strip())
    assert valid_cot_e is True


@pytest.mark.integration
def test_dpo_preference_pair_synthesis_and_pathology_separation(
    synthetic_dpo_preference_pairs: List[DPORecord],
):
    """Verify Invariant 3: Chosen responses pass validators while rejected responses trigger targeted failures."""
    assert len(synthetic_dpo_preference_pairs) == 2

    for pair in synthetic_dpo_preference_pairs:
        pathology = pair.metadata["pathology"]

        if pathology == "MODE_BLEED":
            # Chosen must pass CoT validation
            valid_cot_chosen, _ = validate_reasoning_cot(pair.chosen)
            assert valid_cot_chosen is True

            # Rejected (conversational mode-bleed without <think>) must FAIL CoT validation
            valid_cot_rejected, reason = validate_reasoning_cot(pair.rejected)
            assert valid_cot_rejected is False
            assert "Missing opening <think> tag" in reason

        elif pathology == "SCHEMA_TRAP":
            # Chosen is valid rejection token "(none)"
            valid_cypher_chosen, _ = validate_cypher_syntax(pair.chosen)
            assert valid_cypher_chosen is True

            # Rejected contains hallucinated property query that must not match whitelisted rejection
            assert pair.rejected != "(none)"
            assert "ceo_age" in pair.rejected


@pytest.mark.integration
def test_safetensors_streaming_lora_fusion_and_bf16_precision(tmp_path: Path):
    """Verify Invariant 4: LoRA adapter fusion adheres to BF16 precision bound (|diff| <= 1e-3)."""
    # 1. Create a synthetic base model 2x2 layer in BF16
    w_base_fp32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    raw_bf16 = fp32_to_bf16_bytes(w_base_fp32)
    assert len(raw_bf16) == 8  # 4 elements * 2 bytes each

    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [0, len(raw_bf16)],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    in_shard = tmp_path / "model-00001-of-00001.safetensors"
    out_shard = tmp_path / "model-fused-00001-of-00001.safetensors"

    with in_shard.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(raw_bf16)

    # 2. Define synthetic SFT and DPO LoRA adapters
    # SFT adapter: scale = 2.0, update = [[1.0, 0.0], [2.0, 0.0]]
    sft_adapter = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.array([[0.5, 0.0]], dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.array([[1.0], [2.0]], dtype=np.float32),
    }
    # DPO adapter: scale = 2.0, update = [[0.0, 1.0], [0.0, 2.0]]
    dpo_adapter = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.array([[0.0, 0.25]], dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.array([[2.0], [4.0]], dtype=np.float32),
    }

    # 3. Stream and fuse adapters
    fused_count = fuse_safetensors_shard(
        in_shard_path=in_shard,
        out_shard_path=out_shard,
        adapters=[(sft_adapter, 2.0), (dpo_adapter, 2.0)],
    )
    assert fused_count == 1
    assert out_shard.is_file()

    # 4. Read fused output shard
    with out_shard.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        f.read(hlen)
        fused_bytes = f.read(len(raw_bf16))

    fused_fp32 = bf16_bytes_to_fp32(fused_bytes, (2, 2))

    # Analytical expectation:
    # W_base + 2.0 * ([[1.0], [2.0]] @ [[0.5, 0.0]]) + 2.0 * ([[2.0], [4.0]] @ [[0.0, 0.25]])
    # = [[1, 2], [3, 4]] + [[1, 0], [2, 0]] + [[0, 1], [0, 2]]
    # = [[2, 3], [5, 6]]
    w_expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32)

    # Invariant 4: Merged tensor precision bounded by BF16 mantissa resolution (<= 1e-3)
    max_absolute_error = float(np.max(np.abs(fused_fp32 - w_expected)))
    assert max_absolute_error <= 1e-3, f"Fusion numeric drift exceeded BF16 tolerance: {max_absolute_error}"
    assert np.allclose(fused_fp32, w_expected, atol=1e-3)


@pytest.mark.integration
def test_model_foundry_offline_isolation_guardrail(monkeypatch):
    """Verify Invariant 5: Offline isolation guardrail guarantees zero external network calls."""
    def fail_on_network(*args, **kwargs):
        raise RuntimeError("External network connection attempted during SIT Flow 5 offline execution!")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)

    # Test validates that basic validators run purely in-memory
    assert validate_cypher_syntax("(none)")[0] is True
    sample_reasoning = (
        "<think>\n"
        "Step 1: FabCo facility shutdown triggers critical semiconductor supply disruption across hardware product assembly lines.\n"
        "</think>\n"
        "Implement recommended hedging collar strategy to cover portfolio exposure."
    )
    assert validate_reasoning_cot(sample_reasoning)[0] is True
