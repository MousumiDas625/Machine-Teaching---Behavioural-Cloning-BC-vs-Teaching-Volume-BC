# experiments/collect_maze_demos.py
#
# Load trained PPO expert and collect demonstrations on Procgen Maze.
#
# Demonstration format: list of (obs, action, teacher_loss) tuples
# teacher_loss is stored at collection time — correct reference per level.
#
# Training demos: levels 0-499  (same as expert training range)
# Test levels:    1000-1019     (never seen by expert or learners)

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import pickle
import procgen

from agents.ppo_expert import CNNPolicy, DEVICE

# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════
N_DEMO_LEVELS   = 20      # collect demos from 50 training levels
TRAJS_PER_LEVEL = 5       # 5 demonstrations per level
MAX_STEPS       = 500     # max steps per trajectory
NOISE_EPS       = 0.10    # 10% random actions (lower than GridWorld
                           # because maze requires more precision)
EXPERT_PATH     = "results/data/maze_expert_best.pt"

os.makedirs("results/data", exist_ok=True)


def load_expert(path: str) -> CNNPolicy:
    policy = CNNPolicy(n_actions=15).to(DEVICE)
    policy.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True))
    policy.eval()
    print(f"  Expert loaded from {path}")
    return policy


def collect_one_trajectory(env, expert, eps=NOISE_EPS, seed=0):
    """
    Collect one trajectory on current environment level.
    Each step stores (obs, action, teacher_loss).
    teacher_loss = -log pi_expert(action|obs) stored immediately.
    """
    rng  = np.random.default_rng(seed)
    obs  = env.reset()['rgb'][0]   # shape (64, 64, 3)
    traj = []
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        if rng.random() < eps:
            action = int(rng.integers(0, 15))
        else:
            action = expert.act_greedy(obs)

        # Store teacher loss at collection time
        teacher_loss = expert.loss_at(obs, action)
        traj.append((obs.copy(), action, teacher_loss))

        obs_dict, reward_arr, done_arr, _ = env.step(np.array([action]))
        obs = obs_dict['rgb'][0]
        done = bool(done_arr[0])
        steps += 1

    return traj


def collect_all_demos(expert):
    """
    Collect demonstrations from N_DEMO_LEVELS different maze levels.
    Each level = one unique maze layout.
    Returns list of trajectories.
    """
    all_trajs = []
    traj_seed = 0

    print(f"\n  Collecting demonstrations...")
    print(f"  {'Level':>6} | {'Trajs':>5} | "
          f"{'Avg steps':>10} | {'Avg reward':>10}")
    print(f"  {'─'*40}")

    for level_idx in range(N_DEMO_LEVELS):
        level_seed = level_idx   # levels 0-49

        env = procgen.ProcgenEnv(
            num_envs          = 1,
            env_name          = 'maze',
            num_levels        = 1,
            start_level       = level_seed,
            distribution_mode = 'easy',
        )

        level_trajs = []
        total_steps  = 0
        total_reward = 0.0

        for _ in range(TRAJS_PER_LEVEL):
            traj = collect_one_trajectory(
                env, expert, eps=NOISE_EPS, seed=traj_seed)
            level_trajs.append(traj)
            traj_seed += 1

            # Compute approximate reward (non-zero if goal reached)
            # In Procgen maze: reward=10 if goal reached, 0 otherwise
            steps = len(traj)
            total_steps += steps

        all_trajs.extend(level_trajs)
        avg_steps = total_steps / TRAJS_PER_LEVEL

        print(f"  {level_seed:>6} | "
              f"{TRAJS_PER_LEVEL:>5} | "
              f"{avg_steps:>10.1f} | "
              f"  (see below)")

        env.close()

    print(f"\n  Total trajectories: {len(all_trajs)}")
    total_steps = sum(len(t) for t in all_trajs)
    print(f"  Total (obs,action) pairs: {total_steps:,}")
    return all_trajs


def main():
    print(f"{'='*55}")
    print(f"  Collecting Procgen Maze Demonstrations")
    print(f"{'='*55}")
    print(f"  Expert:         {EXPERT_PATH}")
    print(f"  Training levels: 0 to {N_DEMO_LEVELS-1}")
    print(f"  Trajs/level:    {TRAJS_PER_LEVEL}")
    print(f"  Noise:          {NOISE_EPS}")
    print(f"  Device:         {DEVICE}")

    # Load expert
    expert = load_expert(EXPERT_PATH)

    # Collect demonstrations
    all_trajs = collect_all_demos(expert)

    # Save
    demo_data = {
        "trajs"          : all_trajs,
        "N_DEMO_LEVELS"  : N_DEMO_LEVELS,
        "TRAJS_PER_LEVEL": TRAJS_PER_LEVEL,
        "NOISE_EPS"      : NOISE_EPS,
        "MAX_STEPS"      : MAX_STEPS,
        "total_pairs"    : sum(len(t) for t in all_trajs),
    }
    with open("results/data/maze_demos.pkl", "wb") as f:
        pickle.dump(demo_data, f)

    print(f"\n  Saved: results/data/maze_demos.pkl")
    print(f"  Ready for run_maze_generalisation.py")


if __name__ == "__main__":
    main()
