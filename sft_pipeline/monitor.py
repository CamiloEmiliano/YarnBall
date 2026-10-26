"""
YarnBall SFT Pipeline - Training Telemetry & Remote Monitoring Module
====================================================================
Consolidated operational telemetry, remote training monitoring, and training
dynamics synchronization for Vast.ai GPU training runs.

Replaces fragmented ad-hoc scratch scripts with a permanent, structured interface.

Usage:
------
    # Display formatted live training telemetry (GPU, step, loss, epoch, ETA)
    python -m sft_pipeline.monitor status

    # Tail recent remote training logs
    python -m sft_pipeline.monitor tail -n 30

    # Inspect logged training dynamics checkpoints
    python -m sft_pipeline.monitor dynamics

    # Download training dynamics and report artifacts to local project
    python -m sft_pipeline.monitor fetch
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Resolve project paths
PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent
QWEN_ROOT = PROJECT_ROOT.parent.parent / "QwenSFT_YarnBall"

# Load environment configuration
load_dotenv(PROJECT_ROOT / ".env")
if QWEN_ROOT.exists():
    load_dotenv(QWEN_ROOT / ".env")


def _find_vastai_bin() -> str:
    """Locate vastai executable in venvs or system PATH."""
    candidates = [
        PROJECT_ROOT / ".venv" / "bin" / "vastai",
        QWEN_ROOT / ".venv" / "bin" / "vastai",
        Path.home() / ".local" / "bin" / "vastai",
        Path("/usr/local/bin/vastai"),
        Path("/usr/bin/vastai"),
    ]
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    import shutil
    sys_vast = shutil.which("vastai")
    if sys_vast:
        return sys_vast
    return "vastai"


def _get_vast_api_key() -> str:
    """Resolve Vast.ai API key from environment."""
    key = (
        os.environ.get("VASTAI_FULL_ACCESS_TOKEN_01")
        or os.environ.get("VAST_API_KEY")
    )
    if not key:
        raise ValueError("Vast.ai API key not found in environment (expected VASTAI_FULL_ACCESS_TOKEN_01).")
    return key.strip()


def get_active_instance() -> Optional[Dict[str, Any]]:
    """Retrieve details for the primary running Vast.ai instance."""
    vast_bin = _find_vastai_bin()
    key = _get_vast_api_key()
    cmd = [vast_bin, "show", "instances", "--raw", "--api-key", key]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        return None
    try:
        instances = json.loads(res.stdout)
        running = [i for i in instances if i.get("actual_status") == "running"]
        return running[0] if running else (instances[0] if instances else None)
    except Exception:
        return None


def run_ssh_command(
    cmd_str: str,
    ssh_host: Optional[str] = None,
    ssh_port: Optional[int] = None,
    timeout: int = 15,
) -> subprocess.CompletedProcess:
    """Execute a bash command on the remote training instance via SSH."""
    if not ssh_host or not ssh_port:
        inst = get_active_instance()
        if not inst:
            raise RuntimeError("No active Vast.ai instance found to execute SSH command.")
        ssh_host = inst.get("ssh_host")
        ssh_port = inst.get("ssh_port")

    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-p", str(ssh_port),
        f"root@{ssh_host}",
        cmd_str,
    ]
    return subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)


def get_live_telemetry() -> Dict[str, Any]:
    """
    Collects unified telemetry from the active GPU instance:
    - GPU utilization, memory usage, temperature, power draw
    - Training step, total steps, epoch, loss, eval_loss, learning rate
    - Checkpoint numbers and training dynamics status
    """
    inst = get_active_instance()
    if not inst:
        return {"status": "NO_ACTIVE_INSTANCE", "message": "No active Vast.ai instance running."}

    instance_id = inst.get("id")
    ssh_host = inst.get("ssh_host")
    ssh_port = inst.get("ssh_port")
    gpu_name = inst.get("gpu_name", "Unknown GPU")
    dph = inst.get("dph_total", 0.0)

    remote_script = """
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null || echo "N/A"

echo "=== CHECKPOINTS ==="
ls -d /workspace/QwenSFT_YarnBall/outputs/yarnball-qwen-7b-lora/checkpoint-* 2>/dev/null | xargs -n1 basename || echo "none"

echo "=== DYNAMICS_STEPS ==="
if [ -f /workspace/QwenSFT_YarnBall/outputs/yarnball-qwen-7b-lora/training_dynamics.jsonl ]; then
    python3 -c '
import json
steps = set()
count = 0
with open("/workspace/QwenSFT_YarnBall/outputs/yarnball-qwen-7b-lora/training_dynamics.jsonl") as f:
    for line in f:
        try:
            steps.add(json.loads(line).get("step"))
            count += 1
        except:
            pass
print(f"{count}|{sorted(list(steps))}")
' 2>/dev/null || echo "0|[]"
else
    echo "0|[]"
fi

echo "=== TRAINER_METRICS ==="
# Find newest trainer_state.json across checkpoints or root
state_file=$(ls -t /workspace/QwenSFT_YarnBall/outputs/yarnball-qwen-7b-lora/checkpoint-*/trainer_state.json 2>/dev/null | head -n 1)
if [ -z "$state_file" ]; then
    state_file="/workspace/QwenSFT_YarnBall/outputs/yarnball-qwen-7b-lora/trainer_state.json"
fi
if [ -f "$state_file" ]; then
    python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    st = json.load(f)
step = st.get("global_step", 0)
epoch = st.get("epoch", 0.0)
hist = st.get("log_history", [])
last_train = next((h for h in reversed(hist) if "loss" in h), {})
last_eval = next((h for h in reversed(hist) if "eval_loss" in h), {})
print(json.dumps({
    "step": step,
    "epoch": round(epoch, 3),
    "loss": last_train.get("loss"),
    "lr": last_train.get("learning_rate"),
    "eval_loss": last_eval.get("eval_loss"),
}))
' "$state_file" 2>/dev/null || echo "{}"
else
    echo "{}"
fi

echo "=== TAIL ==="
tail -n 8 /workspace/QwenSFT_YarnBall/training.log 2>/dev/null || echo ""
"""
    res = run_ssh_command(remote_script, ssh_host=ssh_host, ssh_port=ssh_port, timeout=20)
    stdout = res.stdout

    telemetry: Dict[str, Any] = {
        "status": "RUNNING",
        "instance_id": instance_id,
        "gpu_name": gpu_name,
        "dph": dph,
        "ssh_host": ssh_host,
        "ssh_port": ssh_port,
    }

    sections = stdout.split("=== ")
    for sec in sections:
        if sec.startswith("GPU ==="):
            line = sec.replace("GPU ===", "").strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                telemetry["gpu_util"] = parts[0]
                telemetry["gpu_mem_used"] = parts[1]
                telemetry["gpu_mem_total"] = parts[2]
                telemetry["gpu_temp"] = parts[3]
                telemetry["gpu_power"] = parts[4]
        elif sec.startswith("CHECKPOINTS ==="):
            ckpts = [c.strip() for c in sec.replace("CHECKPOINTS ===", "").strip().splitlines() if c.strip() and c.strip() != "none"]
            telemetry["checkpoints"] = ckpts
        elif sec.startswith("DYNAMICS_STEPS ==="):
            dyn_line = sec.replace("DYNAMICS_STEPS ===", "").strip()
            if "|" in dyn_line:
                cnt, steps_str = dyn_line.split("|", 1)
                telemetry["dynamics_record_count"] = int(cnt) if cnt.isdigit() else 0
                try:
                    telemetry["dynamics_steps"] = json.loads(steps_str.replace("'", '"'))
                except Exception:
                    telemetry["dynamics_steps"] = steps_str
        elif sec.startswith("TRAINER_METRICS ==="):
            metric_str = sec.replace("TRAINER_METRICS ===", "").strip()
            try:
                telemetry["metrics"] = json.loads(metric_str)
            except Exception:
                telemetry["metrics"] = {}
        elif sec.startswith("TAIL ==="):
            telemetry["recent_log_tail"] = sec.replace("TAIL ===", "").strip()

    return telemetry


def print_status_dashboard(telemetry: Dict[str, Any]) -> None:
    """Renders a clean human-readable dashboard in the console."""
    if telemetry.get("status") != "RUNNING":
        print(f"\n[Status]: {telemetry.get('message', 'No active training instance detected.')}\n")
        return

    m = telemetry.get("metrics", {})
    inst_id = telemetry.get("instance_id")
    gpu = telemetry.get("gpu_name")
    rate = telemetry.get("dph")
    ssh_host = telemetry.get("ssh_host")
    ssh_port = telemetry.get("ssh_port")

    print("\n" + "=" * 76)
    print(f" YARNBALL SFT RUN TELEMETRY | Instance ID: {inst_id} ({gpu})")
    print("=" * 76)
    print(f" Host / Port:      {ssh_host}:{ssh_port}  (${rate:.2f}/hr)")
    print(f" GPU Memory:       {telemetry.get('gpu_mem_used', 'N/A')} / {telemetry.get('gpu_mem_total', 'N/A')}")
    print(f" GPU Temp / Power: {telemetry.get('gpu_temp', 'N/A')} C  |  {telemetry.get('gpu_power', 'N/A')}")
    print("-" * 76)
    print(f" Current Step:     {m.get('step', 'N/A')}  (Epoch: {m.get('epoch', 'N/A')})")
    print(f" Training Loss:    {m.get('loss', 'N/A')}")
    print(f" Validation Loss:  {m.get('eval_loss', 'N/A')}")
    print(f" Learning Rate:    {m.get('lr', 'N/A')}")
    print("-" * 76)
    print(f" Dynamics Steps:   {telemetry.get('dynamics_steps', [])}")
    print(f" Dynamics Records: {telemetry.get('dynamics_record_count', 0):,} records logged")
    print(f" Checkpoints:      {', '.join(telemetry.get('checkpoints', [])) or 'None yet'}")
    print("-" * 76)
    tail = telemetry.get("recent_log_tail", "").strip()
    if tail:
        print(" Recent Log:\n" + "\n".join(f"   {l}" for l in tail.splitlines()[-4:]))
    print("=" * 76 + "\n")


def tail_remote_log(lines: int = 30) -> None:
    """Tails the remote training.log directly."""
    inst = get_active_instance()
    if not inst:
        print("No active instance running.")
        return
    res = run_ssh_command(
        f"tail -n {lines} /workspace/QwenSFT_YarnBall/training.log",
        ssh_host=inst.get("ssh_host"),
        ssh_port=inst.get("ssh_port"),
    )
    print(res.stdout)


def fetch_remote_artifacts(dest_dir: Optional[Path] = None) -> Path:
    """Downloads training dynamics and report files from remote instance via SCP."""
    inst = get_active_instance()
    if not inst:
        raise RuntimeError("No active instance found to fetch artifacts from.")

    target = dest_dir or (PROJECT_ROOT / "data" / "sft" / "remote_artifacts")
    target.mkdir(parents=True, exist_ok=True)

    ssh_host = inst.get("ssh_host")
    ssh_port = inst.get("ssh_port")

    print(f"Syncing artifacts from instance {inst.get('id')} to {target}...")
    scp_cmd = [
        "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-P", str(ssh_port),
        f"root@{ssh_host}:/workspace/QwenSFT_YarnBall/outputs/yarnball-qwen-7b-lora/training_dynamics.jsonl",
        str(target / "training_dynamics.jsonl"),
    ]
    res = subprocess.run(scp_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"Successfully fetched training_dynamics.jsonl -> {target / 'training_dynamics.jsonl'}")
    else:
        print(f"Warning: Could not fetch training_dynamics.jsonl: {res.stderr}")

    return target


def main():
    parser = argparse.ArgumentParser(description="YarnBall SFT Pipeline Telemetry & Remote Monitoring")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # status
    p_status = subparsers.add_parser("status", help="Show live GPU & training progress dashboard")
    p_status.set_defaults(func=lambda a: print_status_dashboard(get_live_telemetry()))

    # tail
    p_tail = subparsers.add_parser("tail", help="Tail remote training.log")
    p_tail.add_argument("-n", "--lines", type=int, default=30, help="Number of lines to tail")
    p_tail.set_defaults(func=lambda a: tail_remote_log(lines=a.lines))

    # dynamics
    p_dyn = subparsers.add_parser("dynamics", help="Display logged training dynamics steps and metrics")
    p_dyn.set_defaults(func=lambda a: print(f"Dynamics Steps: {get_live_telemetry().get('dynamics_steps')}"))

    # fetch
    p_fetch = subparsers.add_parser("fetch", help="Download remote dynamics and report artifacts locally")
    p_fetch.add_argument("--dest", default=None, help="Destination directory")
    p_fetch.set_defaults(func=lambda a: fetch_remote_artifacts(Path(a.dest) if a.dest else None))

    args = parser.parse_args()
    if not args.command:
        # Default action: status dashboard
        print_status_dashboard(get_live_telemetry())
    else:
        args.func(args)


if __name__ == "__main__":
    main()
