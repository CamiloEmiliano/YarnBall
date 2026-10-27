import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROJECT_ROOT = Path("/home/caspe/practice/rag_samples/graphrag_finance")
STATE_PATH = PROJECT_ROOT / "data" / "sft" / "remote_artifacts" / "trainer_state.json"
DYNAMICS_PATH = PROJECT_ROOT / "data" / "sft" / "remote_artifacts" / "training_dynamics.jsonl"
OUT_DIR = Path("/home/caspe/.gemini/antigravity-ide/brain/e461f87a-63c3-4226-8b4c-61de3eb9e162/plots")
QWEN_OUT_DIR = Path("/home/caspe/practice/QwenSFT_YarnBall/outputs/yarnball-qwen-7b-lora/report/plots")

OUT_DIR.mkdir(parents=True, exist_ok=True)
QWEN_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Trainer State
state_data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
log_history = state_data.get("log_history", [])

train_steps, train_losses = [], []
eval_steps, eval_losses, eval_ppls = [], [], []
lr_steps, learning_rates = [], []

for entry in log_history:
    step = entry.get("step")
    if step is None:
        continue
    if "loss" in entry:
        train_steps.append(int(step))
        train_losses.append(float(entry["loss"]))
    if "learning_rate" in entry:
        lr_steps.append(int(step))
        learning_rates.append(float(entry["learning_rate"]))
    if "eval_loss" in entry:
        eval_steps.append(int(step))
        el = float(entry["eval_loss"])
        eval_losses.append(el)
        eval_ppls.append(math.exp(min(el, 20.0)))

# 2. Load Metadata for all 2,894 validation samples
all_val_meta = []
for fn in ['yarnball_sft_val.jsonl', 'extraction_tasks_val.jsonl', 'reasoning_tasks_val.jsonl']:
    fp = PROJECT_ROOT / "data" / "sft" / fn
    with open(fp, encoding="utf-8") as f:
        for line in f:
            all_val_meta.append(json.loads(line).get("metadata", {}))

# 3. Load Dynamics Records
records = []
with open(DYNAMICS_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

steps = sorted(list(set(r["step"] for r in records)))

# Task mapping
task_info = {
    "EXTRACT_SEC_GRAPH": {
        "name": "Task A: SEC Entity & CIK Graph",
        "color": "#38bdf8",  # Sky Blue
        "marker": "o",
    },
    "EXTRACT_NEWS_EVENT": {
        "name": "Task B: 8-K Material Event Extr.",
        "color": "#f59e0b",  # Amber
        "marker": "s",
    },
    "TEXT_TO_CYPHER": {
        "name": "Task C: Text-to-Cypher Synthesis",
        "color": "#10b981",  # Emerald
        "marker": "^",
    },
    "CONTAGION_REASONING": {
        "name": "Task D: Multi-Hop Supply Chain",
        "color": "#f43f5e",  # Rose
        "marker": "D",
    },
    "PORTFOLIO_RECOMMENDATION": {
        "name": "Task E: Financial Impact & CoT",
        "color": "#c084fc",  # Purple
        "marker": "P",
    },
}

by_task_step = {k: {s: [] for s in steps} for k in task_info.keys()}
by_guid = {}

for r in records:
    st = r["step"]
    idx = int(r["guid"].replace("sample_", ""))
    meta = all_val_meta[idx] if idx < len(all_val_meta) else {}
    raw_task = meta.get("task_type", "EXTRACT_SEC_GRAPH")
    prob = float(r.get("seq_prob", 0.0))
    if raw_task in by_task_step:
        by_task_step[raw_task][st].append(prob)
    
    guid = str(r.get("guid", ""))
    by_guid.setdefault(guid, []).append(prob)

# Cartography coordinates
confs, vars_ = [], []
for guid, probs in by_guid.items():
    confs.append(np.mean(probs))
    vars_.append(np.std(probs))

# 4. Apply Dark Theme
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0b0f19",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#1f2937",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#1e293b",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "legend.facecolor": "#111827",
    "legend.edgecolor": "#334155",
    "font.family": "sans-serif",
})

fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)
fig.suptitle("YarnBall Qwen2.5-7B SFT Run 2: Training Dynamics & Convergence", fontsize=16, fontweight="bold", y=0.98, color="#f8fafc")

# -------------------------------------------------------------
# Subplot 1: Train & Validation Loss (Logarithmic Scale) + Dual Perplexity
# -------------------------------------------------------------
ax1 = axes[0, 0]
lines1, labels1 = [], []
ax1.set_yscale("log")

if train_steps and train_losses:
    l1 = ax1.plot(train_steps, train_losses, color="#38bdf8", alpha=0.35, linewidth=1, label="Raw Train Loss")
    lines1.extend(l1)
    labels1.append("Raw Train Loss")
    
    w = 5
    smoothed = [np.mean(train_losses[max(0, i - w + 1):i + 1]) for i in range(len(train_losses))]
    l2 = ax1.plot(train_steps, smoothed, color="#0284c7", linewidth=2.4, label="Smoothed Train Loss")
    lines1.extend(l2)
    labels1.append("Smoothed Train Loss")

if eval_steps and eval_losses:
    l3 = ax1.plot(eval_steps, eval_losses, color="#f43f5e", marker="o", markersize=5, linewidth=2, linestyle="--", label="Validation Loss")
    lines1.extend(l3)
    labels1.append("Validation Loss")

ax1.set_title("Loss & Validation Perplexity Trajectory (Log Scale)", fontsize=13, fontweight="bold", color="#e2e8f0")
ax1.set_xlabel("Optimization Step", fontsize=11)
ax1.set_ylabel("Cross-Entropy Loss (Log Scale)", color="#94a3b8", fontsize=11)
ax1.set_ylim(0.0004, 5.0)
ax1.grid(True, which="both", alpha=0.2)

if eval_steps and eval_ppls:
    ax1_ppl = ax1.twinx()
    l4 = ax1_ppl.plot(eval_steps, eval_ppls, color="#c084fc", marker="s", markersize=4, linewidth=1.8, linestyle=":", label="Validation Perplexity (PPL)")
    ax1_ppl.set_ylabel("Perplexity PPL = exp(loss)", color="#c084fc", fontsize=11)
    ax1_ppl.tick_params(axis="y", labelcolor="#c084fc")
    ax1_ppl.set_ylim(0.9995, 1.014)
    lines1.extend(l4)
    labels1.append("Validation Perplexity (PPL)")

ax1.legend(lines1, labels1, loc="upper right", framealpha=0.4, fontsize=9.5)

# -------------------------------------------------------------
# Subplot 2: Learning Rate Schedule
# -------------------------------------------------------------
ax2 = axes[0, 1]
if lr_steps and learning_rates:
    ax2.plot(lr_steps, learning_rates, color="#f59e0b", linewidth=2.4, label="Learning Rate (Warmup + Cosine)")
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
ax2.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold", color="#e2e8f0")
ax2.set_xlabel("Optimization Step", fontsize=11)
ax2.set_ylabel("Learning Rate", fontsize=11)
ax2.grid(True)
ax2.legend(loc="upper right", framealpha=0.4, fontsize=9.5)

# -------------------------------------------------------------
# Subplot 3: Functional Per-Task Convergence Curves (Tasks A - E)
# -------------------------------------------------------------
ax3 = axes[1, 0]
x_indices = np.arange(len(steps))

for raw_task, info in task_info.items():
    means = [np.mean(by_task_step[raw_task][s]) for s in steps]
    count = len(by_task_step[raw_task][steps[0]])
    label = f"{info['name']} (N={count:,})"
    ax3.plot(
        x_indices,
        means,
        color=info["color"],
        marker=info["marker"],
        markersize=6,
        linewidth=2.2,
        alpha=0.9,
        label=label,
    )

# Reference Threshold
ax3.axhline(0.70, color="#ef4444", linestyle=":", linewidth=1.5, label="Correctness Threshold (0.70)", alpha=0.8)

# Checkpoint Tick Labels with both step and epoch
step_labels = [f"Step {s}" if s in [0, 25, 50, 100, 200, 400] else f"S{s}\n(Ep {round(s/724, 1):.1f})" for s in steps]
ax3.set_xticks(x_indices)
ax3.set_xticklabels(step_labels, rotation=25, ha="right", fontsize=9, color="#94a3b8")
ax3.set_ylim(-0.02, 1.05)
ax3.set_title("Per-Task Target Sequence Likelihood Convergence (Tasks A–E)", fontsize=13, fontweight="bold", color="#e2e8f0")
ax3.set_xlabel("Probed Checkpoint (Warmup -> Epoch 3)", fontsize=11)
ax3.set_ylabel("Mean Target Seq Probability P(y | x)", fontsize=11)
ax3.grid(True, alpha=0.25)
ax3.legend(loc="lower right", framealpha=0.5, fontsize=8.8)

# -------------------------------------------------------------
# Subplot 4: Dataset Cartography Scatter
# -------------------------------------------------------------
ax4 = axes[1, 1]
scatter = ax4.scatter(vars_, confs, color="#eab308", alpha=0.45, s=16, edgecolor="none", label="Probed Samples (N=2,894)")

ax4.axvline(0.15, color="#eab308", linestyle="--", linewidth=1.5, label="Ambiguity Boundary (std = 0.15)")
ax4.axhline(0.70, color="#10b981", linestyle="--", linewidth=1.5, label="High Confidence (mean = 0.70)")

ax4.text(0.04, 0.88, "EASY-TO-LEARN\n(Low Var, High Conf)", color="#10b981", fontsize=9, fontweight="bold", alpha=0.8)
ax4.text(0.04, 0.30, "HARD-TO-LEARN\n(Low Var, Low Conf)", color="#ef4444", fontsize=9, fontweight="bold", alpha=0.8)
ax4.text(0.24, 0.50, "AMBIGUOUS REGION\n(Empirical Cluster: std ~ 0.33)", color="#facc15", fontsize=9, fontweight="bold", alpha=0.8)

ax4.set_xlim(-0.02, 0.50)
ax4.set_ylim(-0.02, 1.02)
ax4.set_title("Dataset Cartography Coordinate Map", fontsize=13, fontweight="bold", color="#e2e8f0")
ax4.set_xlabel("Variability (Standard Deviation of Seq Prob)", fontsize=11)
ax4.set_ylabel("Confidence (Mean Seq Prob)", fontsize=11)
ax4.grid(True)
ax4.legend(loc="lower left", framealpha=0.4, fontsize=9.5)

fig.tight_layout(rect=[0, 0, 1, 0.95])

out_dash = OUT_DIR / "training_dashboard.png"
qwen_dash = QWEN_OUT_DIR / "training_dashboard.png"

fig.savefig(out_dash, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
fig.savefig(qwen_dash, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"Successfully generated updated training dashboard with Per-Task Convergence:")
print(f"  Artifact path: {out_dash}")
print(f"  QwenSFT path:  {qwen_dash}")
