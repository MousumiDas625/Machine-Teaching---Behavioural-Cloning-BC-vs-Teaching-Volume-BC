# experiments/record_videos.py
#
# Record videos of:
#   1. Expert playing maze (reference behaviour)
#   2. BC policy playing maze (after training)
#   3. TV-BC policy playing maze (after training)
#
# Videos saved to results/videos/
# Requires: pip install imageio imageio-ffmpeg

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import imageio
import procgen
from agents.ppo_expert import CNNPolicy, DEVICE

os.makedirs("results/videos", exist_ok=True)

N_ACTIONS  = 15
MAX_STEPS  = 500
EXPERT_PATH   = "results/data/maze_expert_best.pt"
BC_PATH       = "results/data/bc_policy_maze.pt"
TVBC_PATH     = "results/data/tvbc_policy_maze.pt"

# Maze levels to record
# Training levels (seen during training)
TRAIN_LEVELS = [0, 5, 10]
# Test levels (never seen)
TEST_LEVELS  = [1000, 1001, 1002]


def load_policy(path, n_actions=N_ACTIONS):
    policy = CNNPolicy(n_actions=n_actions).to(DEVICE)
    policy.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True))
    policy.eval()
    return policy


def record_episode(policy, level_seed, label,
                   max_steps=MAX_STEPS, fps=10):
    """
    Record one episode of policy playing maze level.
    Saves as results/videos/{label}_level{level_seed}.mp4

    Parameters
    ----------
    policy     : CNNPolicy or None (None = random policy)
    level_seed : int — which maze layout to use
    label      : str — used in filename e.g. 'expert', 'bc', 'tvbc'
    fps        : int — frames per second in output video
    """
    env = procgen.ProcgenEnv(
        num_envs=1, env_name='maze',
        num_levels=1, start_level=level_seed,
        distribution_mode='easy')

    obs      = env.reset()['rgb'][0]   # (64, 64, 3) uint8
    frames   = [obs.copy()]
    done     = False
    steps    = 0
    total_r  = 0.0

    while not done and steps < max_steps:
        if policy is None:
            action = int(np.random.randint(0, N_ACTIONS))
        else:
            with torch.no_grad():
                probs  = policy.get_action_probs(obs).numpy()
            action = int(np.argmax(probs))

        obs_dict, reward_arr, done_arr, _ = env.step(np.array([action]))
        obs      = obs_dict['rgb'][0]
        total_r += float(reward_arr[0])
        done     = bool(done_arr[0])
        frames.append(obs.copy())
        steps   += 1

    env.close()

    # Save video
    solved    = total_r > 0
    fname     = (f"results/videos/{label}_level{level_seed}"
                 f"_{'SOLVED' if solved else 'FAILED'}.mp4")
    imageio.mimsave(fname, frames, fps=fps)
    print(f"  Saved: {fname}  "
          f"(steps={steps}, reward={total_r:.1f}, "
          f"{'SOLVED' if solved else 'failed'})")
    return fname, solved, total_r


def main():
    print("Recording videos for all policies and levels...")
    print("Videos saved to: results/videos/\n")

    # Install check
    try:
        import imageio
        import imageio.plugins.ffmpeg
    except ImportError:
        print("Installing imageio-ffmpeg...")
        os.system("pip install imageio imageio-ffmpeg --quiet")
        import imageio

    # Load policies
    policies = {}

    print("Loading policies...")
    if os.path.exists(EXPERT_PATH):
        policies['expert'] = load_policy(EXPERT_PATH)
        print(f"  Expert loaded")
    else:
        print(f"  Expert not found at {EXPERT_PATH}")
        print(f"  Run train_expert.py first")

    if os.path.exists(BC_PATH):
        policies['bc'] = load_policy(BC_PATH)
        print(f"  BC loaded")
    else:
        print(f"  BC policy not found — skipping")

    if os.path.exists(TVBC_PATH):
        policies['tvbc'] = load_policy(TVBC_PATH)
        print(f"  TV-BC loaded")
    else:
        print(f"  TV-BC policy not found — skipping")

    all_levels = TRAIN_LEVELS + TEST_LEVELS

    for label, policy in policies.items():
        print(f"\n--- Recording {label.upper()} ---")
        print(f"  Training levels (seen during training):")
        for level in TRAIN_LEVELS:
            record_episode(policy, level, label)

        print(f"  Test levels (NEVER seen during training):")
        for level in TEST_LEVELS:
            record_episode(policy, level, label)

    print(f"\nAll videos saved to results/videos/")
    print(f"Total files: "
          f"{len(os.listdir('results/videos/'))} videos")


if __name__ == "__main__":
    main()
