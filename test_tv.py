# test_tv.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
from agents.policy_net    import PolicyNetwork
from agents.rl_teacher    import RLTeacher
from gridworld_env.gridworld import GridWorld
from teaching.teaching_volume import (compute_tv_single,
                                       compute_tv_batch,
                                       compute_trajectory_tv,
                                       compute_all_trajectory_tvs)

# Load saved teacher and rollouts
with open("data/rollouts/teacher.pkl", "rb") as f:
    teacher = pickle.load(f)
with open("data/rollouts/all_trajs.pkl", "rb") as f:
    all_trajs = pickle.load(f)

# Fresh learner (random weights — this is theta at the start)
learner = PolicyNetwork(state_dim=64, action_dim=4, hidden_dim=64, seed=0)
eta = 0.01

# ----------------------------------------------------------
# Test 1: TV for a single (s, a) pair
# ----------------------------------------------------------
print("=" * 50)
print("Test 1: TV for single (s, a) pairs")
print("=" * 50)

# Use cell 5 (the best cell, reward=1.0)
state = np.zeros(64, dtype=np.float32)
state[5] = 1.0

for action in range(4):
    tv = compute_tv_single(state, action, learner, teacher, eta)
    t_loss = teacher.loss_at(5, action)
    import torch
    with torch.no_grad():
        l_loss = learner.compute_loss(state, action).item()
    print(f"  action={action}: TV={tv:.5f}  "
          f"learner_loss={l_loss:.3f}  teacher_loss={t_loss:.3f}  "
          f"gap={l_loss - t_loss:.3f}")

# ----------------------------------------------------------
# Test 2: TV for a mini-batch
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("Test 2: TV for a mini-batch of 5 pairs")
print("=" * 50)

sample_traj = all_trajs[0][:5]   # first 5 steps of first trajectory
states  = [pair[0] for pair in sample_traj]
actions = [pair[1] for pair in sample_traj]
tv_batch = compute_tv_batch(states, actions, learner, teacher, eta)
print(f"  TV scores: {tv_batch}")
print(f"  Mean TV:   {tv_batch.mean():.5f}")
print(f"  All finite: {np.all(np.isfinite(tv_batch))}")

# ----------------------------------------------------------
# Test 3: TV for a full trajectory
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("Test 3: TV for full trajectory (50 steps)")
print("=" * 50)

traj_tv = compute_trajectory_tv(all_trajs[0], learner, teacher, eta)
print(f"  Trajectory 0 TV (mean): {traj_tv:.5f}")

# Compare clean vs noisy trajectory
clean_tv = compute_trajectory_tv(all_trajs[0],   learner, teacher, eta)
noisy_tv = compute_trajectory_tv(all_trajs[200], learner, teacher, eta)
print(f"\n  Clean traj TV:  {clean_tv:.5f}")
print(f"  Noisy traj TV:  {noisy_tv:.5f}")
print(f"  (Clean should generally be higher than noisy)")

# ----------------------------------------------------------
# Test 4: TV across a small pool
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("Test 4: TV across first 20 trajectories")
print("=" * 50)

small_pool = all_trajs[:20]
tv_scores  = compute_all_trajectory_tvs(small_pool, learner, teacher, eta)
print(f"\n  Top 3 trajectory indices by TV: "
      f"{np.argsort(tv_scores)[-3:][::-1]}")
print(f"  Bottom 3 trajectory indices by TV: "
      f"{np.argsort(tv_scores)[:3]}")
