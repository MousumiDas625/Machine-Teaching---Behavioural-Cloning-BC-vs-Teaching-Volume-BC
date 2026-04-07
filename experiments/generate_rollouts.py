# experiments/generate_rollouts.py
#
# Generate demonstration rollouts for the harder multi-goal GridWorld.
#
# Collects:
#   200 clean trajectories  (teacher acts from its optimal policy)
#   100 noisy trajectories  (teacher acts with 30% random noise)
#
# The harder environment makes noisy demos more dangerous because:
#   - Random actions near traps lead directly into -1.0 cells
#   - TV-BC must learn to identify and downweight these harmful demos
#   - BC will blindly try to clone trap-entering behaviour

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle

from gridworld_env.gridworld import GridWorld
from agents.rl_teacher       import RLTeacher

# ── Setup ──────────────────────────────────────────────────────────────
N_CLEAN      = 200
N_NOISY      = 100
NOISE_EPS    = 0.30    # 30% random actions in noisy trajectories
MAX_STEPS    = 50
GAMMA        = 0.99
SOFTMAX_TEMP = 0.1     # teacher policy temperature (near-deterministic)
SEED         = 42

os.makedirs("data/rollouts",  exist_ok=True)
os.makedirs("data/selected",  exist_ok=True)

# ── Build environment ──────────────────────────────────────────────────
# omega_star=None because the new GridWorld uses its fixed reward map
env = GridWorld(grid_size=8, omega_star=None,
                max_steps=MAX_STEPS, seed=SEED)

print(f"Environment: {env.n_states} states, {env.n_actions} actions")
print(f"Goal cells:  {env.GOAL_CELLS}")
print(f"Trap cells:  {list(env.TRAP_CELLS.keys())}")
print(f"omega_star range: [{env.omega_star.min():.3f}, "
      f"{env.omega_star.max():.3f}]")

# ── Train teacher via Value Iteration ──────────────────────────────────
print("\nTraining Value Iteration teacher...")
teacher = RLTeacher(env=env, gamma=GAMMA, softmax_temp=SOFTMAX_TEMP)
teacher.train()

# Verify teacher
print(f"V* range: [{teacher.V.min():.3f}, {teacher.V.max():.3f}]")
print(f"Best cell: {np.argmax(teacher.V)} "
      f"(V*={teacher.V.max():.3f})")

# Check teacher knows about all three goals
for cell, reward in env.GOAL_CELLS.items():
    row = cell // 8; col = cell % 8
    print(f"  Goal cell {cell} (row={row}, col={col}, "
          f"reward={reward}): V*={teacher.V[cell]:.3f}")

# Evaluate teacher
rng          = np.random.default_rng(SEED)
total_return = 0.0
N_EVAL       = 50
for _ in range(N_EVAL):
    state = env.reset()
    done  = False
    ep_r  = 0.0
    while not done:
        action         = teacher.act_greedy(state)
        state, r, done, _ = env.step(action)
        ep_r          += r
    total_return += ep_r
print(f"Teacher greedy return (avg over {N_EVAL} eps): "
      f"{total_return/N_EVAL:.3f}")

# ── Collect rollouts ───────────────────────────────────────────────────
def collect_trajectory(env, teacher, noisy=False, eps=0.30, seed=0):
    """
    Collect one trajectory of max_steps (s, a) pairs.

    Parameters
    ----------
    noisy : bool  — if True, take random action with prob eps
    eps   : float — noise probability
    seed  : int   — for reproducibility of noise

    Returns
    -------
    list of (state_np, action_int) tuples
    """
    rng_noise = np.random.default_rng(seed)
    state     = env.reset()
    traj      = []
    done      = False

    while not done:
        state_vec = env.get_state_vector(state)

        if noisy and rng_noise.random() < eps:
            action = int(rng_noise.integers(0, env.n_actions))
        else:
            action = teacher.act_greedy(state)

        traj.append((state_vec, action))
        state, _, done, _ = env.step(action)

    return traj

print(f"\nCollecting {N_CLEAN} clean trajectories...")
clean_trajs = []
for i in range(N_CLEAN):
    traj = collect_trajectory(env, teacher, noisy=False, seed=i)
    clean_trajs.append(traj)

print(f"Collecting {N_NOISY} noisy trajectories (eps={NOISE_EPS})...")
noisy_trajs = []
for i in range(N_NOISY):
    traj = collect_trajectory(env, teacher, noisy=True,
                               eps=NOISE_EPS, seed=N_CLEAN + i)
    noisy_trajs.append(traj)

all_trajs = clean_trajs + noisy_trajs

# ── Dataset statistics ─────────────────────────────────────────────────
print(f"\nDataset statistics:")
print(f"  Total trajectories: {len(all_trajs)} "
      f"({N_CLEAN} clean + {N_NOISY} noisy)")

# Count actions in clean vs noisy
action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
for label, trajs in [("Clean", clean_trajs), ("Noisy", noisy_trajs)]:
    counts = np.zeros(4, dtype=int)
    for traj in trajs:
        for (_, a) in traj:
            counts[a] += 1
    total = counts.sum()
    print(f"\n  {label} action distribution ({total} pairs):")
    for a, name in enumerate(action_names):
        print(f"    {name:6s}: {counts[a]:5d} ({100*counts[a]/total:.1f}%)")

# Count how often teacher enters trap cells
print(f"\n  Trap cell visits:")
for label, trajs in [("Clean", clean_trajs), ("Noisy", noisy_trajs)]:
    trap_visits = 0
    total_steps = 0
    for traj in trajs:
        for (state_np, _) in traj:
            cell = int(np.argmax(state_np))
            if cell in env.TRAP_CELLS:
                trap_visits += 1
            total_steps += 1
    print(f"    {label}: {trap_visits}/{total_steps} "
          f"({100*trap_visits/total_steps:.2f}%) steps in traps")

# ── Save ───────────────────────────────────────────────────────────────
with open("data/rollouts/teacher.pkl",      "wb") as f:
    pickle.dump(teacher, f)
with open("data/rollouts/omega_star.pkl",   "wb") as f:
    pickle.dump(env.omega_star, f)
with open("data/rollouts/clean_trajs.pkl",  "wb") as f:
    pickle.dump(clean_trajs, f)
with open("data/rollouts/noisy_trajs.pkl",  "wb") as f:
    pickle.dump(noisy_trajs, f)
with open("data/rollouts/all_trajs.pkl",    "wb") as f:
    pickle.dump(all_trajs, f)

print(f"\nSaved to data/rollouts/")
print(f"  teacher.pkl, omega_star.pkl")
print(f"  clean_trajs.pkl  ({N_CLEAN} trajectories)")
print(f"  noisy_trajs.pkl  ({N_NOISY} trajectories)")
print(f"  all_trajs.pkl    ({len(all_trajs)} trajectories)")
