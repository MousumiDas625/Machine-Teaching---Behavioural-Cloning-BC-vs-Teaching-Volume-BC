# experiments/run_experiment.py
# Full experiment: trains BC and TV-BC, runs all evaluations,
# produces plots for learning curves and sample efficiency.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")    # headless — no display needed on cluster
import matplotlib.pyplot as plt

from agents.policy_net       import PolicyNetwork
from agents.rl_teacher       import RLTeacher
from gridworld_env.gridworld import GridWorld
from learners.standard_bc    import StandardBC
from learners.tv_bc          import TeacherAwareBC
from evaluation.metrics      import (evaluate_all,
                                      sample_efficiency_curve)

# ----------------------------------------------------------
# Load data
# ----------------------------------------------------------
with open("data/rollouts/teacher.pkl",       "rb") as f:
    teacher = pickle.load(f)
with open("data/rollouts/omega_star.pkl",    "rb") as f:
    omega_star = pickle.load(f)
with open("data/selected/train_states.pkl",  "rb") as f:
    states = pickle.load(f)
with open("data/selected/train_actions.pkl", "rb") as f:
    actions = pickle.load(f)

env = GridWorld(grid_size=8, omega_star=omega_star,
                max_steps=50, seed=42)

os.makedirs("results/plots", exist_ok=True)

# ----------------------------------------------------------
# Shared settings
# ----------------------------------------------------------
ETA        = 0.01
BETA       = 2.0
N_EPOCHS   = 100
BATCH_SIZE = 20
SEED_INIT  = 0
SEED_TRAIN = 7

def make_eval_fn(env, n_eps=30):
    from evaluation.metrics import policy_return
    def eval_fn(policy):
        return policy_return(policy, env, n_episodes=n_eps, seed=99)
    return eval_fn

print("=" * 60)
print("  Full Experiment: BC vs TV-BC")
print("=" * 60)
print(f"  eta={ETA}, beta={BETA}, epochs={N_EPOCHS}")

# ----------------------------------------------------------
# Experiment 1: Learning curves
# ----------------------------------------------------------
print("\n[Experiment 1] Learning curves...")

bc_policy = PolicyNetwork(64, 4, 64, seed=SEED_INIT)
bc        = StandardBC(policy=bc_policy, eta=ETA, seed=SEED_TRAIN)
bc.train(states, actions, N_EPOCHS, BATCH_SIZE,
         eval_fn=make_eval_fn(env), eval_every=5, verbose=True)

tvbc_policy = PolicyNetwork(64, 4, 64, seed=SEED_INIT)
tvbc        = TeacherAwareBC(policy=tvbc_policy, teacher=teacher,
                              eta=ETA, beta=BETA, seed=SEED_TRAIN)
tvbc.train(states, actions, N_EPOCHS, BATCH_SIZE,
           eval_fn=make_eval_fn(env), eval_every=5, verbose=True)

# Plot 1a: Training loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(bc.loss_history,   label="Standard BC",  color="#e07b39", lw=2)
axes[0].plot(tvbc.loss_history, label="TV-BC (ITAL)", color="#3a7ebf", lw=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("Training Loss: BC vs TV-BC")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 1b: Policy return curves
axes[1].plot(bc.eval_epochs,   bc.eval_history,
             label="Standard BC",  color="#e07b39", lw=2, marker="o", ms=4)
axes[1].plot(tvbc.eval_epochs, tvbc.eval_history,
             label="TV-BC (ITAL)", color="#3a7ebf", lw=2, marker="s", ms=4)
axes[1].axhline(y=47.9, color="green", linestyle="--",
                label="Teacher (~47.9)", lw=1.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Average Policy Return")
axes[1].set_title("Policy Return: BC vs TV-BC")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/plots/learning_curves.png", dpi=150)
plt.close()
print("  Saved: results/plots/learning_curves.png")

# ----------------------------------------------------------
# Experiment 2: Final metrics comparison
# ----------------------------------------------------------
print("\n[Experiment 2] Final metric comparison...")

bc_metrics   = evaluate_all(bc_policy,   teacher, env, n_eps=100)
tvbc_metrics = evaluate_all(tvbc_policy, teacher, env, n_eps=100)

print(f"\n  {'Metric':<22}  {'BC':>10}  {'TV-BC':>10}  {'Better':>8}")
print("  " + "-" * 55)
lower_better = {"value_alignment", "omega_recovery"}
for m in ["policy_return", "policy_agreement",
          "value_alignment", "omega_recovery"]:
    bc_v   = bc_metrics[m]
    tvbc_v = tvbc_metrics[m]
    if m in lower_better:
        better = "TV-BC" if tvbc_v < bc_v else "BC"
    else:
        better = "TV-BC" if tvbc_v > bc_v else "BC"
    print(f"  {m:<22}  {bc_v:>10.4f}  {tvbc_v:>10.4f}  {better:>8}")

# Bar chart of final metrics
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
metrics_to_plot = [
    ("policy_return",    "Policy Return",         False),
    ("policy_agreement", "Policy Agreement",      False),
    ("value_alignment",  "Value Alignment Error", True),
    ("omega_recovery",   "Omega Recovery Error",  True),
]
colors = ["#e07b39", "#3a7ebf"]

for ax, (key, label, lower_is_better) in zip(axes, metrics_to_plot):
    vals = [bc_metrics[key], tvbc_metrics[key]]
    bars = ax.bar(["BC", "TV-BC"], vals, color=colors, width=0.4)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel(label + ("\n(lower=better)" if lower_is_better
                           else "\n(higher=better)"), fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate bars with values
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Final Metric Comparison: BC vs TV-BC", fontsize=12)
plt.tight_layout()
plt.savefig("results/plots/final_metrics.png", dpi=150)
plt.close()
print("  Saved: results/plots/final_metrics.png")

# ----------------------------------------------------------
# Experiment 3: Sample efficiency
# ----------------------------------------------------------
print("\n[Experiment 3] Sample efficiency (this takes a few minutes)...")
fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

print("  Training BC at different data fractions...")
bc_eff = sample_efficiency_curve(
    learner_class  = StandardBC,
    learner_kwargs = {"eta": ETA, "seed": SEED_TRAIN},
    states=states, actions=actions, env=env,
    fractions=fractions, n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE, n_eval_eps=50, seed=SEED_INIT
)

print("  Training TV-BC at different data fractions...")
tvbc_eff = sample_efficiency_curve(
    learner_class  = TeacherAwareBC,
    learner_kwargs = {"teacher": teacher, "eta": ETA,
                      "beta": BETA, "seed": SEED_TRAIN},
    states=states, actions=actions, env=env,
    fractions=fractions, n_epochs=N_EPOCHS,
    batch_size=BATCH_SIZE, n_eval_eps=50, seed=SEED_INIT
)

# Plot sample efficiency
fig, ax = plt.subplots(figsize=(8, 5))
ns = [int(f * len(states)) for f in fractions]

ax.plot(ns, [bc_eff[f]   for f in fractions],
        label="Standard BC",  color="#e07b39", lw=2,
        marker="o", ms=6)
ax.plot(ns, [tvbc_eff[f] for f in fractions],
        label="TV-BC (ITAL)", color="#3a7ebf", lw=2,
        marker="s", ms=6)
ax.axhline(y=47.9, color="green", linestyle="--",
           label="Teacher (~47.9)", lw=1.5)

ax.set_xlabel("Number of Training (s,a) Pairs", fontsize=12)
ax.set_ylabel("Policy Return", fontsize=12)
ax.set_title("Sample Efficiency: BC vs TV-BC", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/sample_efficiency.png", dpi=150)
plt.close()
print("  Saved: results/plots/sample_efficiency.png")

# Print sample efficiency table
print(f"\n  {'Samples':>8}  {'BC return':>12}  {'TV-BC return':>12}  {'Diff':>8}")
print("  " + "-" * 46)
for f in fractions:
    n  = int(f * len(states))
    bc_r   = bc_eff[f]
    tvbc_r = tvbc_eff[f]
    print(f"  {n:>8}  {bc_r:>12.3f}  {tvbc_r:>12.3f}  "
          f"{tvbc_r - bc_r:>+8.3f}")

print("\n  All experiments complete.")
print("  Plots saved to results/plots/")
print("  Check sample efficiency — TV-BC advantage most visible")
print("  at low data fractions (10%, 25%)")
