# experiments/collect_coinrun_demos.py
#
# Collect demonstrations from PPO expert on CoinRun level 10.
# Stores (obs, action, teacher_loss) triples.
# Teacher loss stored at collection time — correct reference.

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import pickle
import procgen

from agents.pavel_policy import load_pavel_expert, DEVICE

GAME        = 'coinrun'
FIXED_LEVEL = 10
N_TRAJS     = 50       # 50 demonstrations from level 10
MAX_STEPS   = 500
NOISE_EPS   = 0.05     # 5% noise — expert mostly optimal

EXPERT_PATH = "/scr/pavel/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_0p/2026-01-13__06-46-28__seed_6033/model_200015872.pth"
SAVE_DIR    = "results/coinrun/data"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_expert():
    return load_pavel_expert(EXPERT_PATH)


def collect_one_traj(expert, seed=0):
    rng = np.random.default_rng(seed)
    env = procgen.ProcgenEnv(
        num_envs=1, env_name=GAME,
        num_levels=1, start_level=FIXED_LEVEL,
        distribution_mode='easy')

    obs   = env.reset()['rgb'][0]
    traj  = []
    done  = False
    steps = 0

    while not done and steps < MAX_STEPS:
        if rng.random() < NOISE_EPS:
            action = int(rng.integers(0, 15))
        else:
            action = expert.act_greedy(obs)

        teacher_loss = expert.loss_at(obs, action)
        traj.append((obs.copy(), action, teacher_loss))

        obs_dict, _, done_arr, _ = env.step(np.array([action]))
        obs   = obs_dict['rgb'][0]
        done  = bool(done_arr[0])
        steps += 1

    env.close()
    return traj


def main():
    print(f"{'='*55}")
    print(f"  Collecting CoinRun Demos — Level {FIXED_LEVEL}")
    print(f"{'='*55}")
    print(f"  Trajectories: {N_TRAJS}")
    print(f"  Noise:        {NOISE_EPS}")

    expert    = load_expert()
    all_trajs = []
    solved    = 0

    print(f"\n  {'Traj':>5} | {'Steps':>6} | {'Solved':>7}")
    print(f"  {'─'*25}")

    for i in range(N_TRAJS):
        traj = collect_one_traj(expert, seed=i)
        all_trajs.append(traj)
        # Check if expert reached goal (any reward in traj)
        # CoinRun: reward=10 when coin collected
        s = len(traj)
        print(f"  {i+1:>5} | {s:>6} | "
              f"{'YES' if s < MAX_STEPS else 'NO':>7}")
        if s < MAX_STEPS:
            solved += 1

    print(f"\n  Solved: {solved}/{N_TRAJS} "
          f"({100*solved/N_TRAJS:.1f}%)")
    print(f"  Total pairs: {sum(len(t) for t in all_trajs):,}")

    demo_data = {
        "trajs"      : all_trajs,
        "GAME"       : GAME,
        "FIXED_LEVEL": FIXED_LEVEL,
        "N_TRAJS"    : N_TRAJS,
        "NOISE_EPS"  : NOISE_EPS,
    }
    with open(f"{SAVE_DIR}/coinrun_demos.pkl", "wb") as f:
        pickle.dump(demo_data, f)
    print(f"\n  Saved: {SAVE_DIR}/coinrun_demos.pkl")


if __name__ == "__main__":
    main()
