# experiments/train_coinrun_expert.py
#
# Train PPO expert on Procgen CoinRun — Level 10 ONLY
#
# Professor's instruction:
#   Pick one moderate level and stick with it.
#   Level 10 is moderate — has obstacles requiring real jumps.
#   Train until expert reaches 80%+ solve rate.
#
# CoinRun:
#   Agent runs right, must jump over enemies/obstacles
#   Reach the coin on the right side to win
#   Reward: +10 when coin collected, 0 otherwise
#   Teaching differs from optimal:
#     Optimal = shortest path to coin
#     Teaching = demonstrate obstacle avoidance clearly

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

from agents.ppo_expert import (
    CNNPolicy, PPOTrainer, collect_rollout, DEVICE)

# ── Parameters ────────────────────────────────────────────────────────
GAME            = 'coinrun'
FIXED_LEVEL     = 10            # single fixed level — professor's rule
N_ENVS          = 32
N_STEPS         = 256
TOTAL_TIMESTEPS = 25_000_000
N_EPOCHS        = 4
MINIBATCH_SIZE  = 512
LR              = 2.5e-4
GAMMA           = 0.999
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.01
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5

SAVE_DIR = "results/coinrun/data"
os.makedirs(SAVE_DIR, exist_ok=True)


def make_train_env():
    """Train on level 10 only — fixed level as professor instructed."""
    return procgen.ProcgenEnv(
        num_envs          = N_ENVS,
        env_name          = GAME,
        num_levels        = 50,          # diverse training
        start_level       = 0,           # start from easy levels
        distribution_mode = 'easy',
    )


def evaluate_expert(policy, n_episodes=50):
    """Evaluate on level 10 — check mastery of this specific level."""
    eval_env = procgen.ProcgenEnv(
        num_envs          = 1,
        env_name          = GAME,
        num_levels        = 1,
        start_level       = FIXED_LEVEL,
        distribution_mode = 'easy',
    )
    policy.eval()
    total_reward = 0.0
    n_solved     = 0
    obs          = eval_env.reset()['rgb']
    done_count   = 0
    ep_reward    = 0.0

    while done_count < n_episodes:
        obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs, _ = policy.forward(obs_t)
        action = int(torch.argmax(probs, dim=-1).item())
        obs_dict, reward_arr, done_arr, _ = eval_env.step(
            np.array([action]))
        obs       = obs_dict['rgb']
        reward    = float(reward_arr[0])
        done      = bool(done_arr[0])
        ep_reward += reward
        if done:
            total_reward += ep_reward
            if ep_reward > 0:
                n_solved += 1
            ep_reward  = 0.0
            done_count += 1

    eval_env.close()
    return total_reward / n_episodes, n_solved / n_episodes


def main():
    print(f"{'='*60}")
    print(f"  Training PPO Expert — {GAME.upper()} Level {FIXED_LEVEL}")
    print(f"{'='*60}")
    print(f"  Device:    {DEVICE}")
    print(f"  Game:      {GAME}")
    print(f"  Level:     {FIXED_LEVEL} (fixed — professor's rule)")
    print(f"  Target:    80%+ solve rate")
    print(f"  Steps:     {TOTAL_TIMESTEPS:,}")

    policy  = CNNPolicy(n_actions=15).to(DEVICE)
    trainer = PPOTrainer(
        policy=policy, lr=LR, clip_eps=CLIP_EPS,
        entropy_coef=ENTROPY_COEF, value_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM)
    print(f"  Params:    {policy.count_parameters():,}")

    env = make_train_env()

    steps_per_update = N_ENVS * N_STEPS
    n_updates        = TOTAL_TIMESTEPS // steps_per_update
    total_steps      = 0
    best_solve_rate  = 0.0

    reward_history = []
    solve_history  = []

    print(f"\n  {'Update':>7} | {'Steps':>10} | "
          f"{'Return':>8} | {'Solve%':>8} | {'Entropy':>8}")
    print(f"  {'─'*55}")

    for update in range(1, n_updates + 1):
        rollout = collect_rollout(
            env, policy, n_steps=N_STEPS,
            gamma=GAMMA, gae_lambda=GAE_LAMBDA)
        total_steps += steps_per_update

        metrics = trainer.update(
            rollout, n_epochs=N_EPOCHS,
            minibatch_size=MINIBATCH_SIZE)

        if update % 20 == 0 or update == 1:
            avg_ret, solve_rate = evaluate_expert(policy, n_episodes=50)
            reward_history.append(avg_ret)
            solve_history.append(solve_rate)

            print(f"  {update:>7} | {total_steps:>10,} | "
                  f"{avg_ret:>8.3f} | "
                  f"{solve_rate*100:>7.1f}% | "
                  f"{metrics['entropy']:>8.4f}")

            if solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                torch.save(policy.state_dict(),
                           f"{SAVE_DIR}/coinrun_expert_best.pt")

            # Stop early if expert is good enough
            if solve_rate >= 0.90:
                print(f"\n  Expert reached 90% — stopping early")
                break

    torch.save(policy.state_dict(),
               f"{SAVE_DIR}/coinrun_expert_final.pt")

    history = {
        "reward_history" : reward_history,
        "solve_history"  : solve_history,
        "best_solve_rate": best_solve_rate,
        "FIXED_LEVEL"    : FIXED_LEVEL,
        "GAME"           : GAME,
        "DEVICE"         : str(DEVICE),
    }
    with open(f"{SAVE_DIR}/coinrun_expert_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best solve rate: {best_solve_rate*100:.1f}%")
    print(f"  Saved: {SAVE_DIR}/coinrun_expert_best.pt")
    print(f"{'='*60}")
    env.close()


if __name__ == "__main__":
    main()
