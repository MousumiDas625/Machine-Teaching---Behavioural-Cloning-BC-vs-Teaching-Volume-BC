# experiments/train_expert.py
#
# Train PPO expert on Procgen Maze.
# Saves trained policy to results/data/maze_expert.pt
#
# Training levels: 0-499 (500 unique mazes)
# The expert learns general maze navigation.
# Later used to collect demonstrations for BC and TV-BC.
#
# Runtime: ~2-3 hours on CPU, ~30 mins on GPU

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

from agents.ppo_expert import CNNPolicy, PPOTrainer, collect_rollout, DEVICE

# ══════════════════════════════════════════════════════════════════════
# Training parameters
# ══════════════════════════════════════════════════════════════════════
N_ENVS          = 64       # parallel environments
N_STEPS         = 256      # steps per rollout per env
TOTAL_TIMESTEPS = 5_000_000  # total env steps
N_EPOCHS        = 4
MINIBATCH_SIZE  = 512
LR              = 5e-4
GAMMA           = 0.999
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
ENTROPY_COEF    = 0.01
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5

# Training on levels 0-499
TRAIN_START_LEVEL = 0
TRAIN_NUM_LEVELS  = 500

os.makedirs("results/data",  exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


def make_train_env():
    return procgen.ProcgenEnv(
        num_envs          = N_ENVS,
        env_name          = 'maze',
        num_levels        = TRAIN_NUM_LEVELS,
        start_level       = TRAIN_START_LEVEL,
        distribution_mode = 'easy',
        use_sequential_levels = False,
    )


def evaluate_expert(policy, n_episodes=50, seed=1000):
    """
    Evaluate expert on levels it was NOT trained on.
    Uses levels starting at seed=1000 (outside training range 0-499).
    """
    eval_env = procgen.ProcgenEnv(
        num_envs          = 1,
        env_name          = 'maze',
        num_levels        = 0,
        start_level       = seed,
        distribution_mode = 'easy',
    )
    policy.eval()
    total_reward = 0.0
    n_solved     = 0

    obs = eval_env.reset()['rgb']
    done_count = 0
    ep_reward  = 0.0

    while done_count < n_episodes:
        obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs, _ = policy.forward(obs_t)
        action = int(torch.argmax(probs, dim=-1).item())

        obs_dict, reward_arr, done_arr, _ = eval_env.step(np.array([action]))
        obs = obs_dict['rgb']
        reward = float(reward_arr[0])
        done = bool(done_arr[0])
        ep_reward += reward

        if done:
            total_reward += ep_reward
            if ep_reward > 0:
                n_solved += 1
            ep_reward = 0.0
            done_count += 1

    eval_env.close()
    avg_return  = total_reward / n_episodes
    solve_rate  = n_solved    / n_episodes
    return avg_return, solve_rate


def main():
    print(f"{'='*60}")
    print(f"  Training PPO Expert on Procgen Maze")
    print(f"{'='*60}")
    print(f"  Device:          {DEVICE}")
    print(f"  N_ENVS:          {N_ENVS}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Training levels: {TRAIN_START_LEVEL} to "
          f"{TRAIN_START_LEVEL + TRAIN_NUM_LEVELS - 1}")
    print(f"  Eval levels:     1000+ (never seen during training)")

    # Build policy and trainer
    policy  = CNNPolicy(n_actions=15).to(DEVICE)
    trainer = PPOTrainer(
        policy        = policy,
        lr            = LR,
        clip_eps      = CLIP_EPS,
        entropy_coef  = ENTROPY_COEF,
        value_coef    = VALUE_COEF,
        max_grad_norm = MAX_GRAD_NORM,
    )
    print(f"  Network params:  {policy.count_parameters():,}")

    # Build vectorised training environment
    env = make_train_env()

    # Training loop
    steps_per_update = N_ENVS * N_STEPS
    n_updates        = TOTAL_TIMESTEPS // steps_per_update
    total_steps      = 0

    reward_history  = []
    solve_history   = []
    update_history  = []

    print(f"\n  {'Update':>7} | {'Steps':>10} | "
          f"{'Eval return':>12} | {'Solve rate':>10} | "
          f"{'Entropy':>8}")
    print(f"  {'─'*60}")

    best_solve_rate = 0.0

    for update in range(1, n_updates + 1):

        # Collect rollout
        rollout     = collect_rollout(
            env, policy,
            n_steps    = N_STEPS,
            gamma      = GAMMA,
            gae_lambda = GAE_LAMBDA,
        )
        total_steps += steps_per_update

        # PPO update
        metrics = trainer.update(
            rollout,
            n_epochs       = N_EPOCHS,
            minibatch_size = MINIBATCH_SIZE,
        )

        # Evaluate every 50 updates
        if update % 50 == 0 or update == 1:
            avg_ret, solve_rate = evaluate_expert(
                policy, n_episodes=50, seed=1000)
            reward_history.append(avg_ret)
            solve_history.append(solve_rate)
            update_history.append(update)

            print(f"  {update:>7} | {total_steps:>10,} | "
                  f"{avg_ret:>12.3f} | "
                  f"{solve_rate*100:>9.1f}% | "
                  f"{metrics['entropy']:>8.4f}")

            # Save best model
            if solve_rate > best_solve_rate:
                best_solve_rate = solve_rate
                torch.save(policy.state_dict(),
                           "results/data/maze_expert_best.pt")

    # Save final model
    torch.save(policy.state_dict(),
               "results/data/maze_expert.pt")

    # Save training history
    history = {
        "reward_history" : reward_history,
        "solve_history"  : solve_history,
        "update_history" : update_history,
        "total_steps"    : total_steps,
        "best_solve_rate": best_solve_rate,
        "DEVICE"         : str(DEVICE),
    }
    with open("results/data/expert_training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best solve rate: {best_solve_rate*100:.1f}%")
    print(f"  Saved: results/data/maze_expert.pt")
    print(f"  Saved: results/data/maze_expert_best.pt")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
