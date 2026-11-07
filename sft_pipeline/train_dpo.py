"""
Phase 3 Direct Preference Optimization (DPO) Training Script for YarnBall Qwen2.5-7B.

Pulls the empirical Cartography Inflection-Slice DPO dataset (`CamiloEmiliano/yarnball-dpo`)
and the Phase 1 SFT LoRA adapter (`CamiloEmiliano/yarnball-qwen-7b-lora`) from Hugging Face Hub,
executes 2-epoch DPO alignment on a Datacenter NVIDIA A100 (80GB SXM4) in native `bfloat16`,
and automatically publishes the aligned adapter to `CamiloEmiliano/yarnball-qwen-7b-dpo`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("train_dpo")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

SYSTEM_PROMPT = (
    "You are YarnBall's institutional financial graph and risk intelligence engine. "
    "Execute the requested task tag strictly: emit valid openCypher for graph extraction "
    "and query tasks, or emit structured <think>...</think> causal reasoning followed by "
    "an institutional synthesis for contagion and portfolio hedging tasks."
)


def format_chatml_dpo_row(row: Dict[str, Any]) -> Dict[str, str]:
    """
    Wrap raw (`prompt`, `chosen`, `rejected`) records into Qwen2.5 ChatML prompt/completion pairs
    matching the exact SFT inference contract (`benchmark_autoregressive.py`).
    """
    raw_prompt = row["prompt"].strip()
    chatml_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    chosen = f"{row['chosen'].strip()}<|im_end|>"
    rejected = f"{row['rejected'].strip()}<|im_end|>"
    return {
        "prompt": chatml_prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def run_dpo_training(
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    sft_adapter_id: str = "CamiloEmiliano/yarnball-qwen-7b-lora",
    dpo_dataset_id: str = "CamiloEmiliano/yarnball-dpo",
    output_hub_id: str = "CamiloEmiliano/yarnball-qwen-7b-dpo",
    output_dir: str = "./outputs/yarnball-qwen-7b-dpo",
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_train_epochs: float = 2.0,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 1536,
    max_prompt_length: int = 768,
    push_to_hub: bool = True,
) -> Dict[str, Any]:
    """Execute TRL DPOTrainer alignment on merged SFT base + trainable LoRA policy."""
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hf_token = (
        os.getenv("HUGGINGFACE_FULL_ACCESS_TOKEN_01")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    logger.info("1. Loading tokenizer and base model: %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    logger.info("2. Loading and merging Phase 1 SFT adapter (%s) as starting reference policy", sft_adapter_id)
    import inspect
    from huggingface_hub import hf_hub_download

    adapter_cfg_path = hf_hub_download(repo_id=sft_adapter_id, filename="adapter_config.json", token=hf_token)
    raw_adapter_cfg = json.loads(Path(adapter_cfg_path).read_text(encoding="utf-8"))
    valid_lora_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys()) - {"self", "args", "kwargs"}
    filtered_adapter_cfg = {k: v for k, v in raw_adapter_cfg.items() if k in valid_lora_keys}
    sft_lora_config = LoraConfig(**filtered_adapter_cfg)

    sft_peft = PeftModel.from_pretrained(base_model, sft_adapter_id, config=sft_lora_config, token=hf_token)
    merged_sft_model = sft_peft.merge_and_unload()
    if hasattr(merged_sft_model, "enable_input_require_grads"):
        merged_sft_model.enable_input_require_grads()

    logger.info("3. Loading DPO dataset: %s", dpo_dataset_id)
    local_train = Path("/workspace/data/dpo/yarnball_dpo_train.jsonl")
    local_val = Path("/workspace/data/dpo/yarnball_dpo_val.jsonl")
    if local_train.is_file() and local_val.is_file():
        logger.info("Found local DPO JSONL files in /workspace/data/dpo/; loading directly")
        raw_ds = load_dataset(
            "json",
            data_files={"train": str(local_train), "validation": str(local_val)},
        )
    else:
        raw_ds = load_dataset(dpo_dataset_id, token=hf_token)
    train_ds = raw_ds["train"].map(format_chatml_dpo_row, remove_columns=raw_ds["train"].column_names)
    val_ds = raw_ds["validation"].map(format_chatml_dpo_row, remove_columns=raw_ds["validation"].column_names)
    logger.info("Prepared %d train pairs and %d validation pairs", len(train_ds), len(val_ds))

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(out_path),
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        precompute_ref_log_probs=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )

    import inspect
    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    tokenizer_kw = {"processing_class": tokenizer} if "processing_class" in trainer_sig else {"tokenizer": tokenizer}

    trainer = DPOTrainer(
        model=merged_sft_model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        **tokenizer_kw,
    )

    logger.info("4. Starting Phase 3 DPO optimization (beta=%.2f, lr=%.2e, epochs=%.1f)...", beta, learning_rate, num_train_epochs)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    logger.info("5. Saving aligned DPO adapter to %s", out_path)
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    summary = {
        "base_model_id": base_model_id,
        "sft_adapter_id": sft_adapter_id,
        "dpo_dataset_id": dpo_dataset_id,
        "output_hub_id": output_hub_id,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    (out_path / "dpo_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if push_to_hub and hf_token:
        logger.info("6. Pushing aligned DPO adapter to Hugging Face Hub: %s", output_hub_id)
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=output_hub_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(out_path),
            repo_id=output_hub_id,
            repo_type="model",
            commit_message="feat(dpo): publish Phase 3 Cartography Inflection-Slice DPO adapter",
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 DPO alignment on Vast.ai A100")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-adapter", default="CamiloEmiliano/yarnball-qwen-7b-lora")
    parser.add_argument("--dpo-dataset", default="CamiloEmiliano/yarnball-dpo")
    parser.add_argument("--output-hub-id", default="CamiloEmiliano/yarnball-qwen-7b-dpo")
    parser.add_argument("--output-dir", default="./outputs/yarnball-qwen-7b-dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    summary = run_dpo_training(
        base_model_id=args.base_model,
        sft_adapter_id=args.sft_adapter,
        dpo_dataset_id=args.dpo_dataset,
        output_hub_id=args.output_hub_id,
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        push_to_hub=not args.no_push,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

