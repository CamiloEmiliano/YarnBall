"""Unit tests for Phase 4 Model Merging, GGUF Modelfile Generation, Guardrails & Validators."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")

from sft_pipeline.phase4_export_and_serve import (
    apply_runtime_guardrails,
    bf16_bytes_to_fp32,
    convert_safetensors_to_gguf,
    fp32_to_bf16_bytes,
    fuse_safetensors_shard,
    generate_ollama_modelfile,
    generate_vllm_awq_manifest,
    hf_tensor_name_to_gguf,
    validate_cik_integrity,
    validate_cypher_syntax,
    validate_reasoning_cot,
)


def test_gguf_v3_conversion(tmp_path: Path) -> None:
    import json
    import struct
    import numpy as np

    assert hf_tensor_name_to_gguf("model.embed_tokens.weight") == "token_embd.weight"
    assert hf_tensor_name_to_gguf("model.norm.weight") == "output_norm.weight"
    assert hf_tensor_name_to_gguf("lm_head.weight") == "output.weight"
    assert hf_tensor_name_to_gguf("model.layers.3.self_attn.q_proj.weight") == "blk.3.attn_q.weight"
    assert hf_tensor_name_to_gguf("model.layers.3.self_attn.q_proj.bias") == "blk.3.attn_q.bias"
    assert hf_tensor_name_to_gguf("model.layers.3.mlp.down_proj.weight") == "blk.3.ffn_down.weight"

    model_dir = tmp_path / "mini_qwen"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": 8,
                "num_hidden_layers": 1,
                "max_position_embeddings": 128,
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "eos_token_id": 2,
                "bos_token_id": 1,
            }
        ),
        encoding="utf-8",
    )

    w_2d = fp32_to_bf16_bytes(np.ones((4, 4), dtype=np.float32))
    w_1d = fp32_to_bf16_bytes(np.ones((4,), dtype=np.float32))
    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "BF16",
            "shape": [4, 4],
            "data_offsets": [0, len(w_2d)],
        },
        "model.norm.weight": {
            "dtype": "BF16",
            "shape": [4],
            "data_offsets": [len(w_2d), len(w_2d) + len(w_1d)],
        },
    }
    h_bytes = json.dumps(header).encode("utf-8")
    with (model_dir / "model-00001-of-00001.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(h_bytes)))
        f.write(h_bytes)
        f.write(w_2d)
        f.write(w_1d)

    out_gguf = tmp_path / "mini.gguf"
    convert_safetensors_to_gguf(model_dir, out_gguf)
    assert out_gguf.is_file()
    with out_gguf.open("rb") as f:
        assert f.read(4) == b"GGUF"
        ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        assert ver == 3
        assert n_tensors == 2
        assert n_kv >= 15


def test_bf16_safetensors_streaming_fusion(tmp_path: Path) -> None:
    import json
    import struct
    import numpy as np

    w0_f32 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    raw_bf16 = fp32_to_bf16_bytes(w0_f32)
    decoded = bf16_bytes_to_fp32(raw_bf16, (2, 2))
    assert np.allclose(w0_f32, decoded)

    # Build a synthetic BF16 .safetensors shard
    header = {
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "BF16",
            "shape": [2, 2],
            "data_offsets": [0, len(raw_bf16)],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    in_shard = tmp_path / "in.safetensors"
    out_shard = tmp_path / "out.safetensors"
    with in_shard.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(raw_bf16)

    # SFT + DPO LoRA adapters (scale = 2.0 each)
    sft_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.array([[0.5, 0.0]], dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.array([[1.0], [2.0]], dtype=np.float32),
    }
    dpo_dict = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": np.array([[0.0, 0.25]], dtype=np.float32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": np.array([[2.0], [4.0]], dtype=np.float32),
    }
    fused_layers = fuse_safetensors_shard(
        in_shard_path=in_shard,
        out_shard_path=out_shard,
        adapters=[(sft_dict, 2.0), (dpo_dict, 2.0)],
    )
    assert fused_layers == 1

    with out_shard.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        f.read(hlen)
        fused_f32 = bf16_bytes_to_fp32(f.read(len(raw_bf16)), (2, 2))

    # Expected: w0 + 2.0 * [[0.5, 0], [1.0, 0]] + 2.0 * [[0, 0.5], [0, 1.0]] = [[2.0, 3.0], [5.0, 6.0]]
    expected = np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float32)
    assert np.allclose(fused_f32, expected)



def test_phase4_modelfile_and_validators(tmp_path: Path) -> None:
    merged_dir = tmp_path / "yarnball-qwen-7b-merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    modelfile = generate_ollama_modelfile(merged_dir)
    assert modelfile.is_file()
    content = modelfile.read_text(encoding="utf-8")
    assert f"FROM {merged_dir}" in content
    assert "<|im_end|>" in content

    vllm_script = generate_vllm_awq_manifest(merged_dir)
    assert vllm_script.is_file()
    assert "--enable-prefix-caching" in vllm_script.read_text(encoding="utf-8")

    # Test subsidiary CIK rollup guardrail (DISH DBS 0001001811 -> DISH Network 0001001082)
    raw_pred = "(:Company {cik: '0001001811', ticker: 'DISH'})-[:LICENSES_TO]->(:Company {cik: '0000320193', ticker: 'AAPL'})"
    guarded = apply_runtime_guardrails("DISH filing", raw_pred, "EXTRACT_NEWS_EVENT")
    assert "0001001082" in guarded
    assert "0001001811" not in guarded

    # Test Tier 4 Axis-2 Ontology Gate pruning out-of-ontology commentary predicates to (none)
    non_schema_pred = "(n:Company {ticker: 'AMP'})-[:COMPARES_TO {polarity: 'NEUTRAL_STABLE'}]->(t:Company {ticker: 'ABT'})"
    assert apply_runtime_guardrails("Market commentary", non_schema_pred, "EXTRACT_NEWS_EVENT") == "(none)"

    # Test Cypher validator
    ok, _ = validate_cypher_syntax(guarded)
    assert ok
    ok_none, _ = validate_cypher_syntax("(none)")
    assert ok_none

    # Test CIK integrity validator
    cik_ok, pred_ciks, gt_ciks = validate_cik_integrity(
        guarded,
        "(:Company {cik: '0001001082'})-[:LICENSES_TO]->(:Company {cik: '0000320193'})",
    )
    assert cik_ok
    assert pred_ciks == ["0000320193", "0001001082"]
    assert gt_ciks == ["0000320193", "0001001082"]

    # Test CoT validator
    cot_sample = (
        "<think>\nEvaluating Tier-1 wafer supply shock from TSMC to NVDA across buffered inventory.\n</think>\n"
        "Recommendation: Initiate 30-day zero-cost equity collar on NVDA."
    )
    cot_ok, _ = validate_reasoning_cot(cot_sample)
    assert cot_ok
