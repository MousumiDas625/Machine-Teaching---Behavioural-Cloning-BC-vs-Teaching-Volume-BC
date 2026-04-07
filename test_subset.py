# test_subset.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
from agents.policy_net       import PolicyNetwork
from agents.rl_teacher       import RLTeacher
from teaching.teaching_volume  import compute_all_trajectory_tvs
from teaching.subset_selection import (softmax_with_beta,
                                        select_subset,
                                        flatten_trajectories,
                                        make_minibatches)

# Load data
with open("data/rollouts/teacher.pkl",   "rb") as f:
    teacher = pickle.load(f)
with open("data/rollouts/all_trajs.pkl", "rb") as f:
    all_trajs = pickle.load(f)
with open("data/rollouts/clean_trajs.pkl", "rb") as f:
    clean_trajs = pickle.load(f)

print(f"Loaded {len(all_trajs)} trajectories total")
print(f"  First 200 are clean, last 100 are noisy")

# Fresh learner
learner = PolicyNetwork(state_dim=64, action_dim=4, hidden_dim=64, seed=0)
eta     = 0.01

# ----------------------------------------------------------
# Step 1: Score all 300 trajectories
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("Step 1: Score all 300 trajectories with TV")
print("=" * 55)
tv_scores = compute_all_trajectory_tvs(
    all_trajs, learner, teacher, eta, aggregation="mean"
)

# Check: clean vs noisy TV distribution
clean_tvs = tv_scores[:200]
noisy_tvs = tv_scores[200:]
print(f"\nClean trajectory TVs — mean: {clean_tvs.mean():.5f}, "
      f"std: {clean_tvs.std():.5f}")
print(f"Noisy trajectory TVs — mean: {noisy_tvs.mean():.5f}, "
      f"std: {noisy_tvs.std():.5f}")
print("(Clean mean TV should be higher than noisy)")

# ----------------------------------------------------------
# Step 2: Softmax probabilities
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("Step 2: Softmax selection probabilities")
print("=" * 55)

for beta in [0.5, 2.0, 10.0]:
    probs = softmax_with_beta(tv_scores, beta)
    print(f"\n  beta={beta}:")
    print(f"    Max prob:  {probs.max():.5f}  "
          f"(index {np.argmax(probs)})")
    print(f"    Min prob:  {probs.min():.5f}")
    print(f"    Sum:       {probs.sum():.6f}  (must be 1.0)")
    print(f"    Is it a clean traj? index {np.argmax(probs)} < 200: "
          f"{np.argmax(probs) < 200}")

# ----------------------------------------------------------
# Step 3: Select K=80 trajectories
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("Step 3: Select K=80 trajectories (beta=2.0)")
print("=" * 55)

selected_trajs, selected_idx, probs = select_subset(
    trajectories = all_trajs,
    tv_scores    = tv_scores,
    K            = 80,
    beta         = 2.0,
    seed         = 0
)

# How many clean vs noisy were selected?
n_clean_selected = (selected_idx < 200).sum()
n_noisy_selected = (selected_idx >= 200).sum()
print(f"\n  Clean selected: {n_clean_selected}/80  "
      f"({100*n_clean_selected/80:.1f}%)")
print(f"  Noisy selected: {n_noisy_selected}/80  "
      f"({100*n_noisy_selected/80:.1f}%)")
print(f"  (Should be mostly clean — TV filters out noisy demos)")

# ----------------------------------------------------------
# Step 4: Flatten and make mini-batches
# ----------------------------------------------------------
print("\n" + "=" * 55)
print("Step 4: Flatten and mini-batch")
print("=" * 55)

states, actions = flatten_trajectories(selected_trajs)
print(f"  Total (s,a) pairs after flattening: {len(states)}")
print(f"  Expected: 80 trajs × 50 steps = {80*50}")

batches = make_minibatches(states, actions, batch_size=20,
                           rng=np.random.default_rng(0))
print(f"  Number of mini-batches (batch_size=20): {len(batches)}")
print(f"  Expected: {80*50} / 20 = {80*50//20}")
print(f"  First batch size: {len(batches[0][0])}")

# Save TV scores and selected data for use in training
import os
os.makedirs("data/selected", exist_ok=True)
import pickle
with open("data/selected/tv_scores.pkl", "wb") as f:
    pickle.dump(tv_scores, f)
with open("data/selected/selected_trajs.pkl", "wb") as f:
    pickle.dump(selected_trajs, f)
with open("data/selected/selected_idx.pkl", "wb") as f:
    pickle.dump(selected_idx, f)
with open("data/selected/train_states.pkl", "wb") as f:
    pickle.dump(states, f)
with open("data/selected/train_actions.pkl", "wb") as f:
    pickle.dump(actions, f)
print("\n  Saved selected data to data/selected/")
