# experiments/generate_rollouts.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
import torch
from gridworld_env.gridworld import GridWorld
from agents.rl_teacher       import RLTeacher


# ------------------------------------------------------------------
# Helper: save any Python object to disk using pickle
# ------------------------------------------------------------------
def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")


# ------------------------------------------------------------------
# Core function: collect rollouts from the teacher policy
# ------------------------------------------------------------------
def collect_rollouts(env       : GridWorld,
                     teacher   : RLTeacher,
                     n_rollouts: int,
                     noisy     : bool,
                     noise_eps : float,
                     seed      : int = 0) -> list:
    """
    Collect n_rollouts trajectories from the teacher policy.

    Each trajectory is a list of (state_vector, action) tuples.
    state_vector is a numpy array of shape (64,) — one-hot encoding.
    action is an int in {0, 1, 2, 3}.

    Parameters
    ----------
    env        : GridWorld instance
    teacher    : trained RLTeacher
    n_rollouts : how many trajectories to collect
    noisy      : if True, apply epsilon-greedy noise to teacher's actions
    noise_eps  : probability of random action (only used when noisy=True)
    seed       : random seed for start positions

    Returns
    -------
    trajectories : list of n_rollouts trajectories
                   each trajectory = list of (state_np, action_int) tuples
    """
    rng          = np.random.default_rng(seed)
    trajectories = []

    for i in range(n_rollouts):
        # Start from a random cell so we get diverse state coverage
        start_pos = int(rng.integers(0, env.n_states))
        state     = env.reset(start_pos=start_pos)

        trajectory = []
        done       = False

        while not done:
            # Get current cell index from one-hot state
            # np.argmax([0,0,1,0,...]) returns the index of the 1
            state_idx = int(np.argmax(state))

            # Teacher picks an action
            action = teacher.act(state_idx, noisy=noisy, eps=noise_eps)

            # Store BEFORE stepping — we want (state, action) pairs
            # not (next_state, action) pairs
            trajectory.append((state.copy(), action))

            # Step the environment
            state, reward, done, _ = env.step(action)

        trajectories.append(trajectory)

        # Progress update every 50 rollouts
        if (i + 1) % 50 == 0:
            print(f"    Collected {i+1}/{n_rollouts} rollouts...")

    return trajectories


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  Stage 1: GridWorld + Teacher Training + Rollouts")
    print("=" * 55)

    # ----------------------------------------------------------
    # 1. Create GridWorld
    # ----------------------------------------------------------
    print("\n[1] Creating 8x8 GridWorld (seed=42)...")
    env = GridWorld(grid_size=8, max_steps=50, seed=42)
    print(f"    n_states={env.n_states}, omega_star range: "
          f"[{env.omega_star.min():.3f}, {env.omega_star.max():.3f}]")

    # ----------------------------------------------------------
    # 2. Train Teacher
    # ----------------------------------------------------------
    print("\n[2] Training RL Teacher (Value Iteration)...")
    teacher = RLTeacher(
        omega_star   = env.omega_star,
        grid_size    = 8,
        gamma        = 0.99,
        conv_thresh  = 1e-6,
        softmax_temp = 0.1,
        seed         = 42
    )
    teacher.train(verbose=True)

    # ----------------------------------------------------------
    # 3. Collect clean rollouts (no noise)
    # ----------------------------------------------------------
    print("\n[3] Collecting 200 clean rollouts (eps=0.0)...")
    clean_trajs = collect_rollouts(
        env        = env,
        teacher    = teacher,
        n_rollouts = 200,
        noisy      = False,
        noise_eps  = 0.0,
        seed       = 0
    )
    print(f"    Done. {len(clean_trajs)} clean trajectories collected.")

    # ----------------------------------------------------------
    # 4. Collect noisy rollouts (eps=0.3)
    # ----------------------------------------------------------
    print("\n[4] Collecting 100 noisy rollouts (eps=0.3)...")
    noisy_trajs = collect_rollouts(
        env        = env,
        teacher    = teacher,
        n_rollouts = 100,
        noisy      = True,
        noise_eps  = 0.3,
        seed       = 1
    )
    print(f"    Done. {len(noisy_trajs)} noisy trajectories collected.")

    # ----------------------------------------------------------
    # 5. Combine and print statistics
    # ----------------------------------------------------------
    all_trajs = clean_trajs + noisy_trajs

    # Trajectory lengths (should all be 50 since max_steps=50)
    lengths = [len(t) for t in all_trajs]
    print(f"\n[5] Dataset statistics:")
    print(f"    Total trajectories : {len(all_trajs)}")
    print(f"    Trajectory length  : min={min(lengths)}, "
          f"max={max(lengths)}, avg={np.mean(lengths):.1f}")
    print(f"    Total (s,a) pairs  : {sum(lengths)}")

    # Count action distribution in clean vs noisy
    clean_actions = [a for traj in clean_trajs for (s, a) in traj]
    noisy_actions = [a for traj in noisy_trajs for (s, a) in traj]

    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    print(f"\n    Clean action distribution:")
    for i, name in enumerate(action_names):
        count = clean_actions.count(i)
        pct   = 100 * count / len(clean_actions)
        print(f"      {name:5s}: {count:5d}  ({pct:.1f}%)")

    print(f"\n    Noisy action distribution (should be more uniform):")
    for i, name in enumerate(action_names):
        count = noisy_actions.count(i)
        pct   = 100 * count / len(noisy_actions)
        print(f"      {name:5s}: {count:5d}  ({pct:.1f}%)")

    # ----------------------------------------------------------
    # 6. Save everything to data/rollouts/
    # ----------------------------------------------------------
    print("\n[6] Saving to data/rollouts/...")
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    save_pickle(env.omega_star,
                os.path.join(base, "data", "rollouts", "omega_star.pkl"))
    save_pickle(teacher,
                os.path.join(base, "data", "rollouts", "teacher.pkl"))
    save_pickle(clean_trajs,
                os.path.join(base, "data", "rollouts", "clean_trajs.pkl"))
    save_pickle(noisy_trajs,
                os.path.join(base, "data", "rollouts", "noisy_trajs.pkl"))
    save_pickle(all_trajs,
                os.path.join(base, "data", "rollouts", "all_trajs.pkl"))

    print("\n  Done. All rollout data saved to data/rollouts/")
    return env, teacher, clean_trajs, noisy_trajs, all_trajs


if __name__ == "__main__":
    main()
