import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("results.csv")

def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------- 1. Mode comparison: bar chart per metric ----------
mode_df = df[df["mask_prob"] == 0]  # baseline mask_prob for mode comparison
metrics = ["crash_rate", "rejected_rate", "waypoints_reached", "goal_abandoned", "avg_speed", "known_frac"]

for metric in metrics:
    if metric not in mode_df.columns:
        continue
    summary = mode_df.groupby("mode")[metric].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=4)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_',' ').title()} by Mode")
    ax.set_xticklabels(summary.index, rotation=20)
    savefig(fig, f"mode_{metric}.png")

# ---------- 2. Combined dashboard: all metrics in one grid ----------
n = len(metrics)
cols = 3
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
axes = axes.flatten()
for i, metric in enumerate(metrics):
    if metric not in mode_df.columns:
        continue
    summary = mode_df.groupby("mode")[metric].agg(["mean", "std"])
    axes[i].bar(summary.index, summary["mean"], yerr=summary["std"], capsize=4)
    axes[i].set_title(metric.replace("_", " ").title())
    axes[i].tick_params(axis='x', rotation=20)
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("Mode Comparison Dashboard", fontsize=16)
savefig(fig, "dashboard_mode_comparison.png")

# ---------- 3. mask_prob degradation curves (one line per mode, per metric) ----------
degrade_metrics = ["crash_rate", "rejected_rate", "goal_abandoned", "waypoints_reached"]
for metric in degrade_metrics:
    if metric not in df.columns:
        continue
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode].groupby("mask_prob")[metric].mean()
        ax.plot(sub.index, sub.values, marker="o", label=mode)
    ax.set_xlabel("mask_prob")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{metric.replace('_',' ').title()} vs. Sensor Dropout")
    ax.legend()
    savefig(fig, f"degradation_{metric}.png")

# ---------- 4. Situation ablation (if you log a situation_mode column: real/none/random) ----------
if "situation_mode" in df.columns:
    sit_df = df[df["mode"] == "intent"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["crash_rate", "rejected_rate", "waypoints_reached"]):
        summary = sit_df.groupby("situation_mode")[metric].agg(["mean", "std"])
        ax.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=4)
        ax.set_title(metric.replace("_", " ").title())
    fig.suptitle("Effect of Situation Label (real vs. none vs. random)")
    savefig(fig, "situation_ablation.png")

# ---------- 5. Tier win distribution (stacked, needs a separate tiers.csv keyed by mode/seed) ----------
if os.path.isfile("tier_wins.csv"):
    tiers = pd.read_csv("tier_wins.csv")
    tier_cols = [c for c in tiers.columns if c not in ("mode", "seed", "mask_prob")]
    summary = tiers.groupby("mode")[tier_cols].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Override count")
    ax.set_title("Which Tier Decided Each Override")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig, "tier_win_distribution.png")

# ---------- 6. Action distribution per mode (stacked bar, needs action_dist_0..4 columns) ----------
action_cols = [c for c in df.columns if c.startswith("action_dist_")]
if action_cols:
    summary = mode_df.groupby("mode")[action_cols].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    summary.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Fraction of steps")
    ax.set_title("Action Distribution by Mode")
    ax.legend(["left", "keep", "right", "accel", "brake"], bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(fig, "action_distribution.png")

print(f"Saved figures to {OUT_DIR}/")