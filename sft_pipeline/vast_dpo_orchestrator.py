"""
Vast.ai A100 Orchestrator for Phase 3 DPO Alignment (`sft_pipeline/vast_dpo_orchestrator.py`).

Provides subcommands:
- `search`: Find verified 1x NVIDIA A100 80GB offers sorted by price (`dph_total`).
- `provision`: Provision the best verified A100 80GB instance.
- `deploy`: Wait for SSH, install dependencies, upload `train_dpo.py` & `data/dpo/`, and launch DPO in background.
- `status`: Check live GPU & DPO step/reward telemetry from `/workspace/dpo_training.log` and `trainer_state.json`.
- `fetch`: Download `outputs/yarnball-qwen-7b-dpo/` artifacts locally to `data/dpo/remote_artifacts/`.
- `destroy`: Destroy active Vast.ai instance to halt billing immediately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, cast

from dotenv import load_dotenv

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")
PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
QWEN_ROOT = (
    PROJECT_ROOT.parent / "QwenSFT_YarnBall"
    if (PROJECT_ROOT.parent / "QwenSFT_YarnBall").exists()
    else PROJECT_ROOT.parent.parent / "QwenSFT_YarnBall"
)

load_dotenv(PROJECT_ROOT / ".env")
if QWEN_ROOT.exists():
    load_dotenv(QWEN_ROOT / ".env")

try:
    from monitor import _find_vastai_bin, _get_vast_api_key, get_active_instance, run_ssh_command
except ImportError:
    from sft_pipeline.monitor import _find_vastai_bin, _get_vast_api_key, get_active_instance, run_ssh_command


def _get_hf_token() -> str:
    token = (
        os.environ.get("HUGGINGFACE_FULL_ACCESS_TOKEN_01")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass
    if not token:
        raise ValueError("Hugging Face token not found in environment.")
    return token.strip()


def search_a100_offers(max_dph: float = 1.65) -> List[Dict[str, Any]]:
    """Search Vast.ai for verified 1x A100 80GB instances."""
    vast_bin = _find_vastai_bin()
    key = _get_vast_api_key()
    query = (
        f"gpu_ram >= 75 num_gpus = 1 verified = true rentable = true "
        f"reliability > 0.98 inet_down > 300 inet_up > 200 dph <= {max_dph}"
    )
    cmd = [vast_bin, "search", "offers", query, "-o", "dph", "--raw", "--api-key", key]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Vast.ai search failed: {res.stderr or res.stdout}")
    offers = json.loads(res.stdout)
    # Filter to A100 / H100 class GPUs
    a100_offers = [
        o for o in offers
        if any(tag in o.get("gpu_name", "").upper() for tag in ("A100", "H100"))
    ]
    return a100_offers or offers


def provision_instance(offer_id: Optional[int] = None, disk_gb: int = 60) -> Dict[str, Any]:
    """Create a Vast.ai instance on the specified or cheapest verified A100 80GB offer."""
    existing = get_active_instance()
    if existing:
        print(f"Active instance already exists: ID {existing.get('id')} ({existing.get('gpu_name')})")
        return cast(Dict[str, Any], existing)

    if offer_id is None:
        offers = search_a100_offers()
        if not offers:
            raise RuntimeError("No matching A100 80GB offers found on Vast.ai.")
        chosen = offers[0]
        offer_id = int(chosen["id"])
        print(
            f"Selected Offer ID {offer_id}: {chosen.get('gpu_name')} "
            f"({chosen.get('gpu_ram')} GB) at {chosen.get('dph_total', 0.0):.3f} USD/hr"
        )

    vast_bin = _find_vastai_bin()
    key = _get_vast_api_key()
    cmd = [
        vast_bin,
        "create",
        "instance",
        str(offer_id),
        "--image",
        "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel",
        "--disk",
        str(disk_gb),
        "--ssh",
        "--direct",
        "--raw",
        "--api-key",
        key,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to create instance: {res.stderr or res.stdout}")
    out = json.loads(res.stdout)
    print(f"Created Vast.ai instance: {json.dumps(out)}")
    return cast(Dict[str, Any], out)


def wait_for_ssh(timeout_sec: int = 360) -> Dict[str, Any]:
    """Poll Vast.ai until the instance is running and accepts SSH connections."""
    vast_bin = _find_vastai_bin()
    key = _get_vast_api_key()
    start = time.time()
    while time.time() - start < timeout_sec:
        cmd = [vast_bin, "show", "instances", "--raw", "--api-key", key]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and res.stdout.strip():
            instances = json.loads(res.stdout)
            if instances:
                inst = instances[0]
                status = inst.get("actual_status")
                ssh_host = inst.get("ssh_host")
                ssh_port = inst.get("ssh_port")
                print(f"[{int(time.time() - start)}s] Instance {inst.get('id')} status={status} ssh={ssh_host}:{ssh_port}")
                if status == "running" and ssh_host and ssh_port:
                    try:
                        ping = run_ssh_command("echo READY", ssh_host=ssh_host, ssh_port=ssh_port, timeout=10)
                        if ping.returncode == 0 and "READY" in ping.stdout:
                            print("SSH connection established!")
                            return cast(Dict[str, Any], inst)
                    except Exception:
                        pass
        time.sleep(10)
    raise TimeoutError("Timed out waiting for Vast.ai instance SSH readiness.")


def deploy_and_start_dpo() -> None:
    """Upload DPO dataset & training script and launch Phase 3 DPO training in background."""
    inst = wait_for_ssh()
    ssh_host = inst["ssh_host"]
    ssh_port = inst["ssh_port"]
    hf_token = _get_hf_token()

    # 1. Create remote workspace directories
    run_ssh_command("mkdir -p /workspace/data/dpo /workspace/outputs", ssh_host=ssh_host, ssh_port=ssh_port, timeout=20)

    # 2. Upload train_dpo.py and local DPO JSONL splits via SCP
    files_to_scp = [
        (PROJECT_ROOT / "sft_pipeline" / "train_dpo.py", "/workspace/train_dpo.py"),
        (PROJECT_ROOT / "data" / "dpo" / "yarnball_dpo_train.jsonl", "/workspace/data/dpo/yarnball_dpo_train.jsonl"),
        (PROJECT_ROOT / "data" / "dpo" / "yarnball_dpo_val.jsonl", "/workspace/data/dpo/yarnball_dpo_val.jsonl"),
    ]
    for local_path, remote_path in files_to_scp:
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(ssh_port),
            str(local_path),
            f"root@{ssh_host}:{remote_path}",
        ]
        res = subprocess.run(scp_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"SCP failed for {local_path}: {res.stderr}")
        print(f"Uploaded {local_path.name} -> {remote_path}")

    # 3. Clean any prior stalled process, then launch dependency installation + DPO training
    cleanup_cmd = (
        "ps -eo pid,comm,args | awk '$2==\"python3\" || $2==\"bash\" {if ($0 ~ /train_dpo\\.py|run_dpo\\.sh/ && $0 !~ /awk/) print $1}' "
        "| xargs -r kill -9 2>/dev/null || true; "
        "rm -f /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/*.incomplete 2>/dev/null || true"
    )
    run_ssh_command(cleanup_cmd, ssh_host=ssh_host, ssh_port=ssh_port, timeout=15)

    launch_script = f"""cat << 'EOF' > /workspace/run_dpo.sh
#!/bin/bash
set -e
export HF_TOKEN="{hf_token}"
export HUGGINGFACE_FULL_ACCESS_TOKEN_01="{hf_token}"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[$(date -u)] Cleaning mismatched packages & installing hf_transfer / TRL / PEFT / Transformers..."
pip uninstall -y torchvision torchaudio hf-xet 2>/dev/null || true
pip install -q "hf_transfer" "transformers==4.46.3" "peft==0.13.2" "trl==0.12.2" "datasets>=3.0.0" "accelerate>=1.0.0" "huggingface_hub>=0.26.0,<1.0.0"

echo "[$(date -u)] Starting Phase 3 DPO Training..."
python3 /workspace/train_dpo.py \\
    --base-model "Qwen/Qwen2.5-7B-Instruct" \\
    --sft-adapter "CamiloEmiliano/yarnball-qwen-7b-lora" \\
    --dpo-dataset "CamiloEmiliano/yarnball-dpo" \\
    --output-hub-id "CamiloEmiliano/yarnball-qwen-7b-dpo" \\
    --output-dir "/workspace/outputs/yarnball-qwen-7b-dpo" \\
    --beta 0.1 \\
    --lr 5e-6 \\
    --epochs 2.0

echo "[$(date -u)] DPO_TRAINING_COMPLETE"
EOF
chmod +x /workspace/run_dpo.sh
nohup /workspace/run_dpo.sh > /workspace/dpo_training.log 2>&1 &
sleep 2
ps aux | grep run_dpo.sh
"""
    res = run_ssh_command(launch_script, ssh_host=ssh_host, ssh_port=ssh_port, timeout=30)
    print("Launched DPO background job on remote A100:")
    print(res.stdout)


def check_dpo_status() -> Dict[str, Any]:
    """Retrieve live GPU and DPO training status from active Vast.ai instance."""
    inst = get_active_instance()
    if not inst:
        print("No active Vast.ai instance running.")
        return {"status": "NO_ACTIVE_INSTANCE"}

    ssh_host = inst.get("ssh_host")
    ssh_port = inst.get("ssh_port")
    remote_cmd = """
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "N/A"
echo "=== PROC & CACHE ==="
ps aux | grep -E "train_dpo|run_dpo" | grep -v grep || echo "No train_dpo process"
df -h / /workspace 2>/dev/null || true
ls -lh /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/ /workspace/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/ 2>/dev/null || true
echo "=== SUMMARY ==="
if [ -f /workspace/outputs/yarnball-qwen-7b-dpo/dpo_run_summary.json ]; then
    cat /workspace/outputs/yarnball-qwen-7b-dpo/dpo_run_summary.json
else
    echo "IN_PROGRESS"
fi
echo "=== METRICS ==="
tr '\\r' '\\n' < /workspace/dpo_training.log 2>/dev/null | grep -E "\\{'loss|\\{'eval_loss|train_runtime" | tail -n 15 || true
echo "=== LOG_TAIL ==="
tr '\\r' '\\n' < /workspace/dpo_training.log 2>/dev/null | grep -v '^$' | tail -n 15 || echo "No log yet"
"""
    res = run_ssh_command(remote_cmd, ssh_host=ssh_host, ssh_port=ssh_port, timeout=20)
    print(f"Instance {inst.get('id')} ({inst.get('gpu_name')} @ {inst.get('dph_total', 0.0):.3f} USD/hr):")
    print(res.stdout)
    local_summary = PROJECT_ROOT / "data" / "dpo" / "remote_artifacts" / "dpo_run_summary.json"
    if "DPO_TRAINING_COMPLETE" in res.stdout and local_summary.is_file():
        print("Training complete and artifacts verified locally. Auto-destroying Vast.ai instance to stop billing...")
        destroy_active_instance()
    return {"instance": inst, "output": res.stdout}


def fetch_dpo_artifacts() -> Path:
    """Download DPO summary, trainer_state.json, and adapter config from the remote instance."""
    inst = get_active_instance()
    if not inst:
        raise RuntimeError("No active Vast.ai instance found.")
    ssh_host = inst["ssh_host"]
    ssh_port = inst["ssh_port"]

    dest = PROJECT_ROOT / "data" / "dpo" / "remote_artifacts"
    dest.mkdir(parents=True, exist_ok=True)

    for fname in ("dpo_run_summary.json", "adapter_config.json", "trainer_state.json"):
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(ssh_port),
            f"root@{ssh_host}:/workspace/outputs/yarnball-qwen-7b-dpo/{fname}",
            str(dest / fname),
        ]
        subprocess.run(scp_cmd, capture_output=True, text=True)

    # Also check latest checkpoint for trainer_state.json if not at root
    state_cmd = "ls -t /workspace/outputs/yarnball-qwen-7b-dpo/checkpoint-*/trainer_state.json 2>/dev/null | head -n 1"
    res = run_ssh_command(state_cmd, ssh_host=ssh_host, ssh_port=ssh_port, timeout=15)
    ckpt_state = res.stdout.strip()
    if ckpt_state:
        subprocess.run(
            [
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-P", str(ssh_port),
                f"root@{ssh_host}:{ckpt_state}",
                str(dest / "trainer_state.json"),
            ],
            capture_output=True,
            text=True,
        )

    # Fetch full training log
    subprocess.run(
        [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(ssh_port),
            f"root@{ssh_host}:/workspace/dpo_training.log",
            str(dest / "dpo_training.log"),
        ],
        capture_output=True,
        text=True,
    )
    print(f"Fetched DPO remote artifacts to {dest}")
    return dest


def destroy_active_instance() -> None:
    """Destroy all active Vast.ai instances to stop billing."""
    vast_bin = _find_vastai_bin()
    key = _get_vast_api_key()
    cmd = [vast_bin, "show", "instances", "--raw", "--api-key", key]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        print("No instances found.")
        return
    instances = json.loads(res.stdout)
    for inst in instances:
        iid = inst["id"]
        d_cmd = [vast_bin, "destroy", "instance", str(iid), "--api-key", key]
        d_res = subprocess.run(d_cmd, input="y\n", capture_output=True, text=True, timeout=15)
        print(f"Destroyed instance {iid}: {d_res.stdout.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vast.ai A100 Orchestrator for Phase 3 DPO")
    sub = parser.add_subparsers(dest="cmd")

    p_search = sub.add_parser("search")
    p_search.add_argument("--max-dph", type=float, default=1.65)

    p_prov = sub.add_parser("provision")
    p_prov.add_argument("--offer-id", type=int, default=None)

    sub.add_parser("deploy")
    sub.add_parser("status")
    sub.add_parser("fetch")
    sub.add_parser("destroy")

    args = parser.parse_args()
    if args.cmd == "search":
        offers = search_a100_offers(max_dph=args.max_dph)
        for o in offers[:8]:
            print(
                f"ID: {o['id']} | {o.get('gpu_name')} ({o.get('gpu_ram')} GB) | "
                f"DPH: {o.get('dph_total', 0.0):.3f} USD/hr | Rel: {o.get('reliability2', 0.0):.3f}"
            )
    elif args.cmd == "provision":
        provision_instance(offer_id=args.offer_id)
    elif args.cmd == "deploy":
        deploy_and_start_dpo()
    elif args.cmd == "status":
        check_dpo_status()
    elif args.cmd == "fetch":
        fetch_dpo_artifacts()
    elif args.cmd == "destroy":
        destroy_active_instance()
    else:
        check_dpo_status()


if __name__ == "__main__":
    main()
