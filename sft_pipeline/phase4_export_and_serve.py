"""
Phase 4: Model Merging, 4-Bit Quantization, Ollama Packaging & Post-DPO Benchmark Suite.

Executes the Phase 4 deliverables specified in `notes/PLAN_OF_ACTION_SFT.md`:
1. Standalone 16-Bit Multi-Stage Model Merge (`W_final = W_0 + dW_SFT + dW_DPO`) via `merge_and_unload()`.
2. Edge / Laptop 4-Bit GGUF (`Q4_K_M`) quantization and local Ollama registration (`yarnball-qwen:7b`).
3. Datacenter vLLM / 4-Bit AWQ serving manifest and launch script generation.
4. Post-DPO + Phase 4 free-running autoregressive benchmark across all 100 held-out test samples
   (`20` per task across Tasks A-E), comparing accuracy, mode-bleed elimination, and tok/s vs. Phase 2.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")

from dotenv import load_dotenv

logger = logging.getLogger("phase4_export")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QWEN_ROOT = (
    PROJECT_ROOT.parent / "QwenSFT_YarnBall"
    if (PROJECT_ROOT.parent / "QwenSFT_YarnBall").exists()
    else PROJECT_ROOT.parent.parent / "QwenSFT_YarnBall"
)

load_dotenv(PROJECT_ROOT / ".env")
if QWEN_ROOT.exists():
    load_dotenv(QWEN_ROOT / ".env")

SYSTEM_PROMPT = (
    "You are YarnBall's Unified Financial Intelligence Engine, specialized in "
    "SEC Form 10-K/10-Q property graph extraction, 5-axis event edge extraction, "
    "openCypher text-to-query compilation, multi-hop supply chain contagion reasoning, "
    "and institutional portfolio hedging."
)


def _get_hf_token() -> Optional[str]:
    return (
        os.environ.get("HUGGINGFACE_FULL_ACCESS_TOKEN_01")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


# ---------------------------------------------------------------------------
# 1. Standalone 16-Bit Multi-Stage LoRA Merge (W_0 + dW_SFT + dW_DPO)
#    Streaming FP32-Accumulated SafeTensors Engine (<500 MB Peak RAM)
# ---------------------------------------------------------------------------

def bf16_bytes_to_fp32(raw_bytes: bytes, shape: Tuple[int, ...]) -> "np.ndarray":
    """Convert raw little-endian bfloat16 bytes to a 32-bit float NumPy array."""
    import numpy as np

    u16 = np.frombuffer(raw_bytes, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def fp32_to_bf16_bytes(arr_f32: "np.ndarray") -> bytes:
    """Convert a 32-bit float NumPy array to bfloat16 bytes using IEEE round-to-nearest-even."""
    import numpy as np

    contiguous = np.ascontiguousarray(arr_f32, dtype=np.float32)
    u32 = contiguous.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    u16 = ((u32 + rounding_bias) >> np.uint32(16)).astype("<u2")
    return u16.tobytes()


def load_lora_safetensors_dict(safetensors_path: Path) -> Dict[str, "np.ndarray"]:
    """Load all tensors from an adapter_model.safetensors file as float32 NumPy arrays."""
    import struct
    import numpy as np

    tensors: Dict[str, np.ndarray] = {}
    with safetensors_path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            dtype = meta["dtype"]
            shape = tuple(meta["shape"])
            start_off, end_off = meta["data_offsets"]
            f.seek(data_start + start_off)
            raw = f.read(end_off - start_off)
            if dtype == "F32":
                tensors[key] = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            elif dtype == "BF16":
                tensors[key] = bf16_bytes_to_fp32(raw, shape)
            elif dtype == "F16":
                tensors[key] = np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
            else:
                raise ValueError(f"Unsupported LoRA tensor dtype: {dtype} for {key}")
    return tensors


def fuse_safetensors_shard(
    in_shard_path: Path,
    out_shard_path: Path,
    adapters: List[Tuple[Dict[str, "np.ndarray"], float]],
) -> int:
    """
    Stream a single base model .safetensors shard, fusing all stacked LoRA adapters
    in single-rounding 32-bit float precision:
        W_final = round_bf16(W_0_fp32 + sum_i scale_i * (B_i_fp32 @ A_i_fp32))
    Returns the number of linear layers fused in this shard.
    """
    import struct

    fused_count = 0
    with in_shard_path.open("rb") as fin, out_shard_path.open("wb") as fout:
        header_len_bytes = fin.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = fin.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))

        # Write identical header to output shard
        fout.write(header_len_bytes)
        fout.write(header_bytes)

        data_start = 8 + header_len
        # Sort keys by their byte offset so we stream sequentially
        tensor_items = [
            (k, v) for k, v in header.items() if k != "__metadata__"
        ]
        tensor_items.sort(key=lambda kv: kv[1]["data_offsets"][0])

        for key, meta in tensor_items:
            start_off, end_off = meta["data_offsets"]
            fin.seek(data_start + start_off)
            raw_bytes = fin.read(end_off - start_off)

            # Check if this base weight has corresponding LoRA A/B matrices
            if key.endswith(".weight") and meta["dtype"] == "BF16" and len(meta["shape"]) == 2:
                prefix = f"base_model.model.{key[:-7]}"
                key_a = f"{prefix}.lora_A.weight"
                key_b = f"{prefix}.lora_B.weight"

                has_any = any((key_a in ad_dict and key_b in ad_dict) for ad_dict, _ in adapters)
                if has_any:
                    w_fp32 = bf16_bytes_to_fp32(raw_bytes, tuple(meta["shape"])).copy()
                    for ad_dict, scale in adapters:
                        if key_a in ad_dict and key_b in ad_dict:
                            lora_a = ad_dict[key_a]
                            lora_b = ad_dict[key_b]
                            w_fp32 += scale * (lora_b @ lora_a)
                    out_bytes = fp32_to_bf16_bytes(w_fp32)
                    assert len(out_bytes) == len(raw_bytes), "Byte length mismatch after BF16 fusion"
                    fout.write(out_bytes)
                    fused_count += 1
                    continue

            fout.write(raw_bytes)

    return fused_count


def merge_sft_and_dpo_adapters(
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    sft_adapter_id: str = "CamiloEmiliano/yarnball-qwen-7b-lora",
    dpo_adapter_id: str = "CamiloEmiliano/yarnball-qwen-7b-dpo",
    output_dir: Optional[Path] = None,
    device_map: str = "cpu",
) -> Path:
    """
    Fuse Phase 1 SFT (`dW_SFT`) and Phase 3 DPO (`dW_DPO`) rank-32 LoRA adapters sequentially
    into the 16-bit `Qwen/Qwen2.5-7B-Instruct` base weights (`W_0`) using streaming FP32
    accumulation (`<500 MB` RAM) and export a standalone Hugging Face SafeTensors checkpoint.
    """
    import shutil
    from huggingface_hub import hf_hub_download, snapshot_download

    out_dir = output_dir or (PROJECT_ROOT / "data" / "merged_model" / "yarnball-qwen-7b-merged")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if (out_dir / "config.json").is_file() and len(list(out_dir.glob("model-*.safetensors"))) == 4:
        logger.info("Merged 16-bit checkpoint already exists at %s", out_dir)
        return out_dir

    hf_token = _get_hf_token()
    t0 = time.time()

    logger.info("1. Resolving local/Hub snapshot for base model: %s", base_model_id)
    try:
        base_snap = Path(snapshot_download(repo_id=base_model_id, local_files_only=True))
    except Exception:
        base_snap = Path(snapshot_download(repo_id=base_model_id, token=hf_token))

    logger.info("2. Resolving Phase 1 SFT adapter (%s)", sft_adapter_id)
    try:
        sft_snap = Path(snapshot_download(repo_id=sft_adapter_id, local_files_only=True))
    except Exception:
        sft_snap = Path(snapshot_download(repo_id=sft_adapter_id, token=hf_token))

    sft_cfg = json.loads((sft_snap / "adapter_config.json").read_text(encoding="utf-8"))
    sft_scale = float(sft_cfg["lora_alpha"]) / float(sft_cfg["r"])
    sft_tensors = load_lora_safetensors_dict(sft_snap / "adapter_model.safetensors")
    logger.info("Loaded %d SFT LoRA tensors (scale=%.2f)", len(sft_tensors), sft_scale)

    logger.info("3. Resolving Phase 3 DPO adapter (%s)", dpo_adapter_id)
    dpo_cfg_path = Path(hf_hub_download(repo_id=dpo_adapter_id, filename="adapter_config.json", token=hf_token))
    dpo_weights_path = Path(hf_hub_download(repo_id=dpo_adapter_id, filename="adapter_model.safetensors", token=hf_token))
    dpo_cfg = json.loads(dpo_cfg_path.read_text(encoding="utf-8"))
    dpo_scale = float(dpo_cfg["lora_alpha"]) / float(dpo_cfg["r"])
    dpo_tensors = load_lora_safetensors_dict(dpo_weights_path)
    logger.info("Loaded %d DPO LoRA tensors (scale=%.2f)", len(dpo_tensors), dpo_scale)

    # Copy config and tokenizer metadata files
    for meta_file in [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]:
        src = base_snap / meta_file
        if src.is_file():
            shutil.copy2(src, out_dir / meta_file)

    for tok_file in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]:
        src = sft_snap / tok_file
        if not src.is_file():
            src = base_snap / tok_file
        if src.is_file():
            shutil.copy2(src, out_dir / tok_file)

    # Stream and fuse each of the 4 base model shards in FP32
    shards = sorted(base_snap.glob("model-*.safetensors"))
    total_fused = 0
    for idx, shard_path in enumerate(shards, 1):
        t_s = time.time()
        out_shard = out_dir / shard_path.name
        logger.info("4.%d Fusing shard %s -> %s ...", idx, shard_path.name, out_shard.name)
        count = fuse_safetensors_shard(
            in_shard_path=shard_path,
            out_shard_path=out_shard,
            adapters=[(sft_tensors, sft_scale), (dpo_tensors, dpo_scale)],
        )
        total_fused += count
        logger.info("   Fused %d linear projection layers in %s (%.1fs)", count, shard_path.name, time.time() - t_s)

    elapsed = time.time() - t0
    manifest = {
        "base_model_id": base_model_id,
        "sft_adapter_id": sft_adapter_id,
        "dpo_adapter_id": dpo_adapter_id,
        "sft_scale": sft_scale,
        "dpo_scale": dpo_scale,
        "total_fused_linear_layers": total_fused,
        "merged_dir": str(out_dir),
        "precision": "bfloat16 (FP32-accumulated single-rounding)",
        "merge_elapsed_sec": round(elapsed, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Completed 16-bit multi-stage merge (%d layers) in %.1fs", total_fused, elapsed)
    return out_dir


# ---------------------------------------------------------------------------
# 2. GGUF v3 Export, Q4_K_M Quantization & Ollama Modelfile Registration
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_BF16 = 30


def hf_tensor_name_to_gguf(hf_name: str) -> str:
    """Map a Hugging Face Qwen2 SafeTensors parameter name to its llama.cpp GGUF tensor name."""
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"

    m = re.match(r"^model\.layers\.(\d+)\.(.+)$", hf_name)
    if not m:
        raise ValueError(f"Unrecognized Qwen2 tensor name: {hf_name}")
    layer_idx, suffix = m.group(1), m.group(2)
    mapping = {
        "input_layernorm.weight": f"blk.{layer_idx}.attn_norm.weight",
        "post_attention_layernorm.weight": f"blk.{layer_idx}.ffn_norm.weight",
        "self_attn.q_proj.weight": f"blk.{layer_idx}.attn_q.weight",
        "self_attn.q_proj.bias": f"blk.{layer_idx}.attn_q.bias",
        "self_attn.k_proj.weight": f"blk.{layer_idx}.attn_k.weight",
        "self_attn.k_proj.bias": f"blk.{layer_idx}.attn_k.bias",
        "self_attn.v_proj.weight": f"blk.{layer_idx}.attn_v.weight",
        "self_attn.v_proj.bias": f"blk.{layer_idx}.attn_v.bias",
        "self_attn.o_proj.weight": f"blk.{layer_idx}.attn_output.weight",
        "mlp.gate_proj.weight": f"blk.{layer_idx}.ffn_gate.weight",
        "mlp.up_proj.weight": f"blk.{layer_idx}.ffn_up.weight",
        "mlp.down_proj.weight": f"blk.{layer_idx}.ffn_down.weight",
    }
    if suffix not in mapping:
        raise ValueError(f"Unrecognized Qwen2 layer tensor suffix: {suffix}")
    return mapping[suffix]


def _pack_gguf_string(s: str) -> bytes:
    import struct

    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_gguf_kv(key: str, val_type: int, val: Any) -> bytes:
    import struct

    buf = bytearray(_pack_gguf_string(key))
    buf.extend(struct.pack("<I", val_type))
    if val_type == GGUF_TYPE_UINT32:
        buf.extend(struct.pack("<I", int(val)))
    elif val_type == GGUF_TYPE_INT32:
        buf.extend(struct.pack("<i", int(val)))
    elif val_type == GGUF_TYPE_FLOAT32:
        buf.extend(struct.pack("<f", float(val)))
    elif val_type == GGUF_TYPE_BOOL:
        buf.extend(struct.pack("<?", bool(val)))
    elif val_type == GGUF_TYPE_STRING:
        buf.extend(_pack_gguf_string(str(val)))
    elif val_type == GGUF_TYPE_ARRAY:
        elem_type, items = val
        buf.extend(struct.pack("<IQ", elem_type, len(items)))
        if elem_type == GGUF_TYPE_STRING:
            for item in items:
                buf.extend(_pack_gguf_string(str(item)))
        elif elem_type == GGUF_TYPE_INT32:
            buf.extend(struct.pack(f"<{len(items)}i", *items))
        elif elem_type == GGUF_TYPE_FLOAT32:
            buf.extend(struct.pack(f"<{len(items)}f", *items))
        else:
            raise ValueError(f"Unsupported GGUF array element type: {elem_type}")
    else:
        raise ValueError(f"Unsupported GGUF KV type: {val_type}")
    return bytes(buf)


def convert_safetensors_to_gguf(merged_dir: Path, output_gguf_path: Path) -> Path:
    """
    Stream a merged Hugging Face Qwen2 SafeTensors directory into a llama.cpp-compatible
    GGUF v3 BF16 model file without loading all shards into RAM simultaneously.
    1D tensors (norms and biases) are promoted to F32 as required by ggml kernels;
    2D weight matrices are stored in native BF16.
    """
    import struct

    merged_dir = Path(merged_dir).resolve()
    output_gguf_path = Path(output_gguf_path).resolve()
    output_gguf_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((merged_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(cfg.get("vocab_size", 152064))

    # Load tokenizer vocabulary and merges if present
    tokens_list: List[str] = [f"<pad_{i}>" for i in range(vocab_size)]
    token_types: List[int] = [1] * vocab_size
    merges_list: List[str] = []

    tok_json_path = merged_dir / "tokenizer.json"
    if tok_json_path.is_file():
        tok_data = json.loads(tok_json_path.read_text(encoding="utf-8"))
        vocab_dict = tok_data.get("model", {}).get("vocab", {})
        for tok_str, tok_id in vocab_dict.items():
            idx = int(tok_id)
            if 0 <= idx < vocab_size:
                tokens_list[idx] = tok_str
                token_types[idx] = 1
        for added in tok_data.get("added_tokens", []):
            idx = int(added["id"])
            if 0 <= idx < vocab_size:
                tokens_list[idx] = added["content"]
                token_types[idx] = 3 if added.get("special", False) else 1
        raw_merges = tok_data.get("model", {}).get("merges", [])
        for m_entry in raw_merges:
            if isinstance(m_entry, list):
                merges_list.append(" ".join(m_entry))
            else:
                merges_list.append(str(m_entry))

    chat_template = (
        "{%- if messages[0]['role'] == 'system' -%}"
        "<|im_start|>system\n{{ messages[0]['content'] }}<|im_end|>\n"
        "{%- endif -%}"
        "{%- for message in messages -%}"
        "{%- if message['role'] != 'system' -%}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{%- endif -%}"
        "{%- endfor -%}"
        "<|im_start|>assistant\n"
    )
    tok_cfg_path = merged_dir / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        if tok_cfg.get("chat_template"):
            chat_template = str(tok_cfg["chat_template"])

    kv_pairs: List[Tuple[str, int, Any]] = [
        ("general.architecture", GGUF_TYPE_STRING, "qwen2"),
        ("general.name", GGUF_TYPE_STRING, "yarnball-qwen-7b"),
        ("general.file_type", GGUF_TYPE_UINT32, 32),  # LLAMA_FTYPE_MOSTLY_BF16
        ("qwen2.block_count", GGUF_TYPE_UINT32, int(cfg.get("num_hidden_layers", 28))),
        ("qwen2.context_length", GGUF_TYPE_UINT32, int(cfg.get("max_position_embeddings", 32768))),
        ("qwen2.embedding_length", GGUF_TYPE_UINT32, int(cfg.get("hidden_size", 3584))),
        ("qwen2.feed_forward_length", GGUF_TYPE_UINT32, int(cfg.get("intermediate_size", 18944))),
        ("qwen2.attention.head_count", GGUF_TYPE_UINT32, int(cfg.get("num_attention_heads", 28))),
        ("qwen2.attention.head_count_kv", GGUF_TYPE_UINT32, int(cfg.get("num_key_value_heads", 4))),
        ("qwen2.rope.freq_base", GGUF_TYPE_FLOAT32, float(cfg.get("rope_theta", 1000000.0))),
        ("qwen2.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, float(cfg.get("rms_norm_eps", 1e-6))),
        ("tokenizer.ggml.model", GGUF_TYPE_STRING, "gpt2"),
        ("tokenizer.ggml.pre", GGUF_TYPE_STRING, "qwen2"),
        ("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, tokens_list)),
        ("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, token_types)),
        ("tokenizer.ggml.merges", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, merges_list)),
        ("tokenizer.ggml.eos_token_id", GGUF_TYPE_UINT32, int(cfg.get("eos_token_id", 151645))),
        ("tokenizer.ggml.padding_token_id", GGUF_TYPE_UINT32, int(cfg.get("bos_token_id", 151643))),
        ("tokenizer.ggml.bos_token_id", GGUF_TYPE_UINT32, int(cfg.get("bos_token_id", 151643))),
        ("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL, False),
        ("tokenizer.chat_template", GGUF_TYPE_STRING, chat_template),
    ]

    # Collect tensor metadata across all SafeTensors shards in sequential order
    shard_paths = sorted(merged_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(f"No .safetensors shards found in {merged_dir}")

    tensor_entries: List[Dict[str, Any]] = []
    current_data_offset = 0

    for sp in shard_paths:
        with sp.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen).decode("utf-8"))
            data_start = 8 + hlen
            items = [(k, v) for k, v in header.items() if k != "__metadata__"]
            items.sort(key=lambda kv: kv[1]["data_offsets"][0])
            for hf_name, meta in items:
                gguf_name = hf_tensor_name_to_gguf(hf_name)
                shape = tuple(meta["shape"])
                start_off, end_off = meta["data_offsets"]
                is_1d = len(shape) == 1
                ggml_type = GGML_TYPE_F32 if is_1d else GGML_TYPE_BF16
                num_elems = 1
                for d in shape:
                    num_elems *= d
                byte_size = num_elems * (4 if is_1d else 2)

                # Align tensor data offset to 32 bytes
                pad = (GGUF_DEFAULT_ALIGNMENT - (current_data_offset % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
                aligned_offset = current_data_offset + pad
                current_data_offset = aligned_offset + byte_size

                tensor_entries.append(
                    {
                        "hf_name": hf_name,
                        "gguf_name": gguf_name,
                        "shape": shape,
                        "ggml_type": ggml_type,
                        "shard_path": sp,
                        "shard_byte_start": data_start + start_off,
                        "shard_byte_len": end_off - start_off,
                        "dtype": meta["dtype"],
                        "gguf_offset": aligned_offset,
                        "gguf_byte_size": byte_size,
                    }
                )

    logger.info("Writing GGUF v3 header + %d tensor descriptors to %s...", len(tensor_entries), output_gguf_path)
    with output_gguf_path.open("wb") as fout:
        fout.write(GGUF_MAGIC)
        fout.write(struct.pack("<IQQ", GGUF_VERSION, len(tensor_entries), len(kv_pairs)))
        for k, vtype, val in kv_pairs:
            fout.write(_pack_gguf_kv(k, vtype, val))

        for te in tensor_entries:
            fout.write(_pack_gguf_string(te["gguf_name"]))
            dims_rev = list(reversed(te["shape"]))
            fout.write(struct.pack("<I", len(dims_rev)))
            for d in dims_rev:
                fout.write(struct.pack("<Q", int(d)))
            fout.write(struct.pack("<IQ", te["ggml_type"], te["gguf_offset"]))

        # Pad end of header to 32-byte boundary before tensor data section
        pos = fout.tell()
        header_pad = (GGUF_DEFAULT_ALIGNMENT - (pos % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT
        if header_pad:
            fout.write(b"\x00" * header_pad)
        data_section_start = fout.tell()

        # Stream tensor payloads shard by shard
        current_shard: Optional[Path] = None
        fin = None
        try:
            for te in tensor_entries:
                if te["shard_path"] != current_shard:
                    if fin is not None:
                        fin.close()
                    current_shard = te["shard_path"]
                    logger.info("  Streaming tensors from %s into GGUF...", current_shard.name)
                    fin = current_shard.open("rb")

                expected_pos = data_section_start + te["gguf_offset"]
                cur_pos = fout.tell()
                if expected_pos > cur_pos:
                    fout.write(b"\x00" * (expected_pos - cur_pos))

                fin.seek(te["shard_byte_start"])
                raw = fin.read(te["shard_byte_len"])
                if te["ggml_type"] == GGML_TYPE_F32:
                    if te["dtype"] == "BF16":
                        arr_f32 = bf16_bytes_to_fp32(raw, te["shape"])
                        fout.write(arr_f32.astype("<f4").tobytes())
                    elif te["dtype"] == "F32":
                        fout.write(raw)
                    else:
                        raise ValueError(f"Unsupported 1D tensor dtype: {te['dtype']}")
                else:
                    if te["dtype"] == "BF16":
                        fout.write(raw)
                    elif te["dtype"] == "F32":
                        import numpy as np

                        arr_f32 = np.frombuffer(raw, dtype="<f4").reshape(te["shape"])
                        fout.write(fp32_to_bf16_bytes(arr_f32))
                    else:
                        raise ValueError(f"Unsupported 2D tensor dtype: {te['dtype']}")
        finally:
            if fin is not None:
                fin.close()

    logger.info("Exported BF16 GGUF v3 checkpoint to %s (%.2f GB)", output_gguf_path, output_gguf_path.stat().st_size / 1e9)
    return output_gguf_path


def quantize_gguf_q4_k_m(
    bf16_gguf_path: Path,
    q4_gguf_path: Path,
    quantize_type: str = "Q4_K_M",
    llama_quantize_bin: str = "/usr/local/lib/ollama/llama-quantize",
) -> Path:
    """Run llama.cpp `llama-quantize` to compress BF16 GGUF into 4-bit K-quant `Q4_K_M`."""
    bf16_gguf_path = Path(bf16_gguf_path).resolve()
    q4_gguf_path = Path(q4_gguf_path).resolve()
    cmd = [llama_quantize_bin, str(bf16_gguf_path), str(q4_gguf_path), quantize_type.upper()]
    logger.info("Running llama-quantize: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Quantized %s -> %s (%.2f GB)", bf16_gguf_path.name, q4_gguf_path.name, q4_gguf_path.stat().st_size / 1e9)
    return q4_gguf_path


def generate_ollama_modelfile(model_source_path: Path, modelfile_path: Optional[Path] = None) -> Path:
    """Create the production Ollama Modelfile with ChatML template and deterministic stop tokens."""
    source_resolved = Path(model_source_path).resolve()
    default_parent = source_resolved.parent if source_resolved.is_dir() else source_resolved.parent
    target = modelfile_path or (default_parent / "Modelfile")
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    content = f'''FROM {source_resolved}

TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{- range .Messages }}}}{{{{- if eq .Role "user" }}}}<|im_start|>user
{{{{ .Content }}}}<|im_end|>
{{{{ else if eq .Role "assistant" }}}}<|im_start|>assistant
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}{{{{- end }}}}<|im_start|>assistant
"""

SYSTEM """{SYSTEM_PROMPT}"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
'''
    target.write_text(content, encoding="utf-8")
    logger.info("Wrote Ollama Modelfile to %s", target)
    return target


def generate_vllm_awq_manifest(merged_dir: Path) -> Path:
    """Generate the datacenter vLLM 4-bit AWQ serving script and manifest."""
    manifest_dir = merged_dir.parent
    manifest_dir.mkdir(parents=True, exist_ok=True)

    vllm_script = manifest_dir / "serve_vllm_awq.sh"
    script_content = f"""#!/usr/bin/env bash
# Phase 4 Datacenter High-Throughput Serving (vLLM + 4-bit AWQ + PagedAttention)
set -euo pipefail

MERGED_MODEL_DIR="{merged_dir}"
PORT="${{VLLM_PORT:-8000}}"

exec python3 -m vllm.entrypoints.openai.api_server \\
    --model "${{MERGED_MODEL_DIR}}" \\
    --served-model-name "yarnball-qwen-7b" \\
    --dtype bfloat16 \\
    --max-model-len 16384 \\
    --gpu-memory-utilization 0.90 \\
    --enable-prefix-caching \\
    --port "${{PORT}}"
"""
    vllm_script.write_text(script_content, encoding="utf-8")
    vllm_script.chmod(0o755)
    return vllm_script


def build_and_register_ollama_model(
    merged_dir: Path,
    model_tag: str = "yarnball-qwen:7b",
    quantize_level: str = "Q4_K_M",
) -> Dict[str, Any]:
    """
    Convert the merged 16-bit SafeTensors checkpoint to BF16 GGUF v3, quantize to 4-bit Q4_K_M
    via `/usr/local/lib/ollama/llama-quantize`, and register in Ollama via `ollama create`.
    """
    t0 = time.time()
    merged_dir = Path(merged_dir).resolve()
    parent_dir = merged_dir.parent
    bf16_gguf = parent_dir / "yarnball-qwen-7b-bf16.gguf"
    q4_gguf = parent_dir / "yarnball-qwen-7b-q4_k_m.gguf"

    generate_vllm_awq_manifest(merged_dir)

    if not q4_gguf.is_file():
        if not bf16_gguf.is_file():
            convert_safetensors_to_gguf(merged_dir, bf16_gguf)
        quantize_gguf_q4_k_m(bf16_gguf, q4_gguf, quantize_type=quantize_level.upper())
        if bf16_gguf.is_file():
            logger.info("Removing intermediate BF16 GGUF file %s to reclaim disk space...", bf16_gguf.name)
            bf16_gguf.unlink()

    modelfile_path = generate_ollama_modelfile(q4_gguf, modelfile_path=parent_dir / "Modelfile")

    cmd = ["ollama", "create", model_tag, "-f", str(modelfile_path)]
    logger.info("Registering Q4_K_M GGUF in Ollama: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    elapsed = time.time() - t0
    logger.info("Successfully quantized and registered %s (%s) in %.1fs", model_tag, quantize_level, elapsed)
    return {
        "model_tag": model_tag,
        "quantize_level": quantize_level,
        "q4_gguf_path": str(q4_gguf),
        "q4_gguf_size_gb": round(q4_gguf.stat().st_size / 1e9, 2),
        "modelfile_path": str(modelfile_path),
        "elapsed_sec": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# 3. AST, CIK & CoT Validators + Subsidiary Rollup (Matching Phase 2 Suite)
# ---------------------------------------------------------------------------

# Known SEC Exhibit 21 Operating Subsidiary -> Parent S&P 500 CIK Rollup Map
SUBSIDIARY_TO_PARENT_CIK_MAP: Dict[str, str] = {
    "0001001811": "0001001082",  # DISH DBS Corp -> DISH Network Corp
}

# Canonical Axis-2 Relational Ontology Edge Types for Tasks A & B (Tier 4 Schema Gate)
CANONICAL_EXTRACTION_RELATIONS = frozenset(
    {
        "SUPPLIES_TO",
        "CUSTOMER_OF",
        "PARTNERED_WITH",
        "SUBSIDIARY_OF",
        "COMPETES_WITH",
        "LICENSES_FROM",
        "LICENSES_TO",
        "SOLE_SOURCE_DEPENDENT_ON",
        "ACQUIRED_BY",
        "DEFAULTED_ON",
        "SPUN_OFF_FROM",
        "FILED",
        "EXPOSED_TO_RISK",
    }
)


def apply_runtime_guardrails(prompt: str, prediction: str, task_type: str) -> str:
    """
    Apply deterministic Tier 4 post-processing guardrails:
    1. Roll up known Exhibit 21 operating subsidiary CIKs (e.g. DISH DBS 0001001811 -> 0001001082)
       when the parent entity is referenced.
    2. Enforce the bounded Axis-2 Relational Ontology on Tasks A and B (`EXTRACT_SEC_GRAPH`,
       `EXTRACT_NEWS_EVENT`):
       - Automatically repairs lifecycle termination predicates (e.g. [:TERMINATED]) to canonical
         relations with status: 'TERMINATED' instead of pruning valid extractions to (none).
       - Prunes non-schema descriptive commentary predicates (e.g. [:COMPARES_TO]) to (none).
    """
    from ontology.guardrails import DEFAULT_ONTOLOGY_GATE

    return DEFAULT_ONTOLOGY_GATE.apply_guardrails(prediction, task_type)


def validate_cypher_syntax(text: str) -> Tuple[bool, str]:
    """Validate openCypher syntax structure for property graph queries / statements or (none)."""
    cleaned = text.strip()
    if not cleaned:
        return False, "Empty output"
    if cleaned.lower() == "(none)":
        return True, "Valid (none) rejection"

    stack: List[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    in_single_quote = False
    in_double_quote = False

    for ch in cleaned:
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif not in_single_quote and not in_double_quote:
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if not stack or stack[-1] != pairs[ch]:
                    return False, f"Unbalanced delimiter: unexpected '{ch}'"
                stack.pop()

    if in_single_quote or in_double_quote:
        return False, "Unclosed string literal"
    if stack:
        return False, f"Unclosed opening delimiter: '{stack[-1]}'"

    has_clause = bool(re.search(r"\b(MATCH|MERGE|RETURN|WHERE|WITH|CREATE|SET)\b", cleaned, re.IGNORECASE))
    has_triple = bool(re.search(r"\(:?\w*(\s*\{[^}]*\})?\)-(\[.*?\])?->\(:?\w*(\s*\{[^}]*\})?\)", cleaned))

    if not (has_clause or has_triple):
        return False, "Missing valid Cypher clause (MATCH/MERGE/RETURN) or graph triple pattern"

    return True, "Valid"


def validate_cik_integrity(prediction: str, ground_truth: str) -> Tuple[bool, List[str], List[str]]:
    """Extract 10-digit CIK numbers and verify zero hallucinated CIKs relative to ground truth."""
    pred_ciks = set(re.findall(r"\b\d{10}\b", prediction))
    gt_ciks = set(re.findall(r"\b\d{10}\b", ground_truth))
    hallucinated = pred_ciks - gt_ciks
    return len(hallucinated) == 0, sorted(pred_ciks), sorted(gt_ciks)


def validate_reasoning_cot(prediction: str) -> Tuple[bool, str]:
    """Validate Chain-of-Thought (<think>...</think>) structure for Tasks D and E."""
    if "<think>" not in prediction:
        return False, "Missing opening <think> tag"
    if "</think>" not in prediction:
        return False, "Missing closing </think> tag"

    think_content = prediction.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    if len(think_content.split()) < 10:
        return False, "Degenerate / empty reasoning block"

    post_think = prediction.split("</think>", 1)[1].strip()
    if len(post_think.split()) < 5:
        return False, "Missing recommendation / synthesis block following </think>"

    return True, "Valid"


# ---------------------------------------------------------------------------
# 4. High-Throughput Ollama Benchmark Runner (100 Held-Out Test Samples)
# ---------------------------------------------------------------------------

def query_ollama_generate(
    prompt: str,
    task_type: str,
    model_tag: str = "yarnball-qwen:7b",
    ollama_url: str = "http://127.0.0.1:11434/api/generate",
    max_new_tokens: int = 512,
    enforce_think_prefix: bool = True,
) -> Tuple[str, float, int, float]:
    """
    Execute a single autoregressive generation request against the local quantized Ollama runtime.
    When `enforce_think_prefix=True` and `task_type` is Task D or E, pre-fills `<think>\\n` in the
    assistant turn to guarantee zero first-token Cypher mode bleed.
    """
    assistant_prefix = (
        "<think>\n"
        if (enforce_think_prefix and task_type in ("CONTAGION_REASONING", "PORTFOLIO_RECOMMENDATION"))
        else ""
    )
    raw_chatml = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )

    payload = {
        "model": model_tag,
        "prompt": raw_chatml,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": max_new_tokens,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }

    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_elapsed = time.time() - t0

    raw_response = body.get("response", "")
    full_pred = f"{assistant_prefix}{raw_response}".strip()
    pred_text = apply_runtime_guardrails(prompt, full_pred, task_type)

    eval_count = int(body.get("eval_count") or max(1, len(pred_text.split())))
    eval_duration_ns = int(body.get("eval_duration") or 0)
    if eval_duration_ns > 0:
        decode_elapsed = eval_duration_ns / 1e9
        tok_per_sec = eval_count / decode_elapsed
    else:
        tok_per_sec = eval_count / wall_elapsed if wall_elapsed > 0 else 0.0

    return pred_text, wall_elapsed, eval_count, tok_per_sec


def run_post_dpo_ollama_benchmark(
    test_file: Optional[Path] = None,
    output_results_path: Optional[Path] = None,
    output_scorecard_path: Optional[Path] = None,
    model_tag: str = "yarnball-qwen:7b",
    samples_per_task: int = 20,
    enforce_think_prefix: bool = True,
) -> Dict[str, Any]:
    """Run the 100-sample autoregressive benchmark against the merged + Q4_K_M quantized DPO model."""
    test_path = Path(test_file or (PROJECT_ROOT / "data" / "sft" / "yarnball_sft_test.jsonl"))
    results_path = Path(output_results_path or (PROJECT_ROOT / "data" / "dpo" / "benchmark_post_dpo_results.jsonl"))
    scorecard_path = Path(output_scorecard_path or (PROJECT_ROOT / "data" / "dpo" / "benchmark_post_dpo_scorecard.json"))
    results_path.parent.mkdir(parents=True, exist_ok=True)

    samples_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("metadata", {}).get("task_type", "UNKNOWN")
            samples_by_task[task].append(row)

    eval_set: List[Dict[str, Any]] = []
    for task, rows in sorted(samples_by_task.items()):
        eval_set.extend(rows[:samples_per_task])

    # Warmup call so model weights are pinned in RTX 3070 VRAM before timing
    logger.info("Warming up %s in local Ollama VRAM...", model_tag)
    try:
        query_ollama_generate("Warmup ping", "TEXT_TO_CYPHER", model_tag=model_tag, max_new_tokens=8)
    except Exception as exc:
        logger.warning("Warmup warning: %s", exc)

    results: List[Dict[str, Any]] = []
    task_metrics: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0,
            "ast_valid": 0,
            "cik_clean": 0,
            "cot_valid": 0,
            "functional_pass": 0,
            "latencies": [],
            "token_counts": [],
            "tokens_per_sec": [],
        }
    )

    t_start = time.time()
    with results_path.open("w", encoding="utf-8") as out_f:
        for idx, item in enumerate(eval_set, 1):
            sample_id = item["sample_id"]
            prompt = item["prompt"]
            gt = item["target_completion"]
            task = item.get("metadata", {}).get("task_type", "UNKNOWN")

            pred_text, latency, num_tokens, tok_sec = query_ollama_generate(
                prompt=prompt,
                task_type=task,
                model_tag=model_tag,
                enforce_think_prefix=enforce_think_prefix,
            )

            if task in ("CONTAGION_REASONING", "PORTFOLIO_RECOMMENDATION"):
                cot_valid, cot_err = validate_reasoning_cot(pred_text)
                cypher_valid, cypher_err = cot_valid, cot_err
            else:
                cypher_valid, cypher_err = validate_cypher_syntax(pred_text)
                cot_valid, cot_err = True, "N/A"

            cik_valid, pred_ciks, gt_ciks = validate_cik_integrity(pred_text, gt)
            passed = bool(cypher_valid and cik_valid and cot_valid)

            tm = task_metrics[task]
            tm["total"] += 1
            if cypher_valid:
                tm["ast_valid"] += 1
            if cik_valid:
                tm["cik_clean"] += 1
            if cot_valid:
                tm["cot_valid"] += 1
            if passed:
                tm["functional_pass"] += 1
            tm["latencies"].append(latency)
            tm["token_counts"].append(num_tokens)
            tm["tokens_per_sec"].append(tok_sec)

            record = {
                "sample_id": sample_id,
                "task_type": task,
                "prompt": prompt,
                "ground_truth": gt,
                "prediction": pred_text,
                "latency_sec": round(latency, 3),
                "generated_tokens": num_tokens,
                "tokens_per_sec": round(tok_sec, 2),
                "cypher_ast_valid": cypher_valid,
                "cypher_error": None if cypher_valid else cypher_err,
                "cik_clean": cik_valid,
                "predicted_ciks": pred_ciks,
                "ground_truth_ciks": gt_ciks,
                "cot_valid": cot_valid,
                "cot_error": None if cot_valid else cot_err,
                "functional_pass": passed,
            }
            results.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            icon = "PASS" if passed else "FAIL"
            print(f"[{idx:03d}/{len(eval_set):03d}] {icon} {task:<25} | {num_tokens:3d} toks | {latency:5.2f}s ({tok_sec:5.1f} tok/s)")

    total_elapsed = time.time() - t_start
    total_passed = sum(1 for r in results if r["functional_pass"])
    overall_acc = round((total_passed / len(results)) * 100, 2) if results else 0.0
    overall_tps = round(sum(r["tokens_per_sec"] for r in results) / len(results), 2) if results else 0.0

    scorecard: Dict[str, Any] = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_tag": model_tag,
        "quantization": "GGUF Q4_K_M (Merged SFT + DPO)",
        "total_samples": len(eval_set),
        "total_functional_pass": total_passed,
        "overall_functional_accuracy_percent": overall_acc,
        "total_elapsed_sec": round(total_elapsed, 2),
        "overall_tokens_per_sec": overall_tps,
        "phase2_baseline_comparison": {
            "phase2_functional_accuracy_percent": 96.0,
            "phase2_overall_tokens_per_sec": 7.22,
            "phase2_total_elapsed_sec": 1818.23,
            "speedup_factor": round(overall_tps / 7.22, 2) if overall_tps > 0 else 0.0,
        },
        "tasks": {},
    }

    for task, m in sorted(task_metrics.items()):
        n = m["total"]
        scorecard["tasks"][task] = {
            "sample_count": n,
            "functional_pass_percent": round((m["functional_pass"] / n) * 100, 2) if n else 0.0,
            "syntax_valid_percent": round((m["ast_valid"] / n) * 100, 2) if n else 0.0,
            "cik_clean_percent": round((m["cik_clean"] / n) * 100, 2) if n else 0.0,
            "cot_valid_percent": round((m["cot_valid"] / n) * 100, 2) if n else 0.0,
            "avg_generated_tokens": round(sum(m["token_counts"]) / n, 1) if n else 0.0,
            "avg_latency_sec": round(sum(m["latencies"]) / n, 2) if n else 0.0,
            "avg_tokens_per_sec": round(sum(m["tokens_per_sec"]) / n, 2) if n else 0.0,
        }

    scorecard_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    logger.info(
        "Phase 4 Post-DPO Benchmark Complete: %s%% accuracy (%d/%d) at %.1f tok/s (%.1fs total)",
        overall_acc,
        total_passed,
        len(eval_set),
        overall_tps,
        total_elapsed,
    )
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Model Merge, GGUF Quantization & Benchmark")
    sub = parser.add_subparsers(dest="cmd")

    p_merge = sub.add_parser("merge", help="Fuse 16-bit base + Phase 1 SFT + Phase 3 DPO adapters")
    p_merge.add_argument("--device-map", default="cpu")

    p_gguf = sub.add_parser("gguf", help="Convert merged SafeTensors to BF16 GGUF and quantize to Q4_K_M")
    p_gguf.add_argument("--quantize", default="Q4_K_M")

    p_ollama = sub.add_parser("ollama", help="Quantize merged checkpoint to Q4_K_M and register in Ollama")
    p_ollama.add_argument("--model-tag", default="yarnball-qwen:7b")
    p_ollama.add_argument("--quantize", default="Q4_K_M")

    p_bench = sub.add_parser("benchmark", help="Run 100-sample post-DPO benchmark against Ollama")
    p_bench.add_argument("--model-tag", default="yarnball-qwen:7b")
    p_bench.add_argument("--samples-per-task", type=int, default=20)
    p_bench.add_argument("--no-prefix-force", action="store_true")

    sub.add_parser("all", help="Run merge -> ollama Q4_K_M -> 100-sample benchmark end-to-end")

    args = parser.parse_args()
    merged_dir = PROJECT_ROOT / "data" / "merged_model" / "yarnball-qwen-7b-merged"

    if args.cmd == "merge":
        merge_sft_and_dpo_adapters(output_dir=merged_dir, device_map=args.device_map)
    elif args.cmd == "gguf":
        parent_dir = merged_dir.parent
        bf16_gguf = parent_dir / "yarnball-qwen-7b-bf16.gguf"
        q4_gguf = parent_dir / "yarnball-qwen-7b-q4_k_m.gguf"
        generate_vllm_awq_manifest(merged_dir)
        if not q4_gguf.is_file():
            if not bf16_gguf.is_file():
                convert_safetensors_to_gguf(merged_dir, bf16_gguf)
            quantize_gguf_q4_k_m(bf16_gguf, q4_gguf, quantize_type=args.quantize.upper())
            if bf16_gguf.is_file():
                bf16_gguf.unlink()
        generate_ollama_modelfile(q4_gguf, modelfile_path=parent_dir / "Modelfile")
        print(json.dumps({"q4_gguf_path": str(q4_gguf), "size_gb": round(q4_gguf.stat().st_size / 1e9, 2)}, indent=2))
    elif args.cmd == "ollama":
        info = build_and_register_ollama_model(merged_dir, model_tag=args.model_tag, quantize_level=args.quantize)
        print(json.dumps(info, indent=2))
    elif args.cmd == "benchmark":
        sc = run_post_dpo_ollama_benchmark(
            model_tag=args.model_tag,
            samples_per_task=args.samples_per_task,
            enforce_think_prefix=not args.no_prefix_force,
        )
        print(json.dumps(sc, indent=2))
    elif args.cmd == "all":
        merge_sft_and_dpo_adapters(output_dir=merged_dir, device_map="cpu")
        build_and_register_ollama_model(merged_dir)
        sc = run_post_dpo_ollama_benchmark()
        print(json.dumps(sc, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
