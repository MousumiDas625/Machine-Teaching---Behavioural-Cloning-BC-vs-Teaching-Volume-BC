# visualise_policy.py
# Load trained BC and TV-BC policies and show their trajectories
# on the test maps. Prints the path the robot takes step by step.

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pickle
import torch

from agents.policy_net_mapobs       import PolicyNetworkMapObs
from agents.rl_teacher              import RLTeacher
from gridworld_env.gridworld_mapobs import GridWorldMapObs
from learners.standard_bc           import StandardBC
from learners.tv_bc                 import TeacherAwareBC
from experiments.run_generalisation import (
    make_env, make_teacher, collect_trajectory,
    BETA_TV, ETA, SEED_INIT, OBS_DIM, ACTION_DIM,
    N_GOALS, N_TRAPS, TRAJ_LENGTH, GAMMA, SOFTMAX_TEMP
)

ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
ACTION_ARROW = {0: '↑', 1: '↓', 2: '←', 3: '→'}

def print_grid_with_path(env, path_cells, title=""):
    """
    Print the 8x8 grid showing the robot's path.
    path_cells: list of cell indices visited in order
    """
    print(f"\n  {title}")
    print(f"  {'─'*45}")

    # Count visits per cell
    visit_count = {}
    for i, cell in enumerate(path_cells):
        visit_count[cell] = visit_count.get(cell, 0) + 1

    start_cell = path_cells[0]
    end_cell   = path_cells[-1]

    for row in range(env.grid_size):
        line = "  "
        for col in range(env.grid_size):
            cell = row * env.grid_size + col

            if cell == end_cell and cell in env.goal_cells:
                line += " [G] "   # reached goal
            elif cell == end_cell and cell in env.trap_cells:
                line += " [T] "   # ended in trap
            elif cell == start_cell:
                line += "  S  "   # start
            elif cell in env.goal_cells:
                line += f"G{env.goal_cells[cell]:+.1f}"
            elif cell in env.trap_cells:
                line += " -1  "
            elif cell in visit_count:
                # Show how many times visited
                line += f"  {'·'*min(visit_count[cell],3)}  "[:5]
            else:
                line += "  .  "
        print(line)

    print(f"\n  Path length: {len(path_cells)} steps")
    print(f"  Start: cell {start_cell} "
          f"(row={start_cell//8}, col={start_cell%8})")
    print(f"  End:   cell {end_cell} "
          f"(row={end_cell//8}, col={end_cell%8})")

    reached_goal = end_cell in env.goal_cells
    hit_trap     = any(c in env.trap_cells for c in path_cells[1:])
    trap_count   = sum(1 for c in path_cells[1:] if c in env.trap_cells)

    print(f"  Reached goal: {'YES ✓' if reached_goal else 'NO ✗'}")
    print(f"  Trap hits:    {trap_count}")

    total_reward = sum(env.omega_star[c] for c in path_cells[1:])
    print(f"  Total return: {total_reward:.3f}")


def rollout_and_show(policy, env, label, n_episodes=5, seed=99):
    """
    Run n_episodes and show the path for each.
    """
    rng = np.random.default_rng(seed)
    print(f"\n{'═'*55}")
    print(f"  {label} — {n_episodes} episodes on this map")
    print(f"{'═'*55}")

    total_return = 0.0
    n_goals      = 0
    trap_hits    = 0
    policy.eval()

    for ep in range(n_episodes):
        # Alternate between fixed start and random starts
        start = env.START_CELL if ep < n_episodes // 2 \
                else int(rng.integers(0, env.n_states))

        state     = env.reset(start_pos=start)
        path      = [state]
        actions   = []
        done      = False
        ep_return = 0.0
        reached   = False

        while not done:
            obs    = env.get_full_obs(state)
            probs  = policy.get_action_probs(obs).numpy()
            action = int(np.argmax(probs))
            state, reward, done, info = env.step(action)
            path.append(state)
            actions.append(action)
            ep_return += reward
            if info.get('reached_goal', False):
                reached = True

        total_return += ep_return
        n_goals      += 1 if reached else 0
        trap_hits    += sum(1 for c in path[1:] if c in env.trap_cells)

        # Print action sequence
        action_str = ' '.join(ACTION_ARROW[a] for a in actions[:20])
        if len(actions) > 20:
            action_str += f" ...+{len(actions)-20}"

        print(f"\n  Episode {ep+1}: return={ep_return:+.2f}  "
              f"goal={'YES' if reached else 'NO'}  "
              f"steps={len(actions)}")
        print(f"  Actions: {action_str}")

        # Show grid with path
        print_grid_with_path(
            env, path,
            title=f"Episode {ep+1} path "
                  f"({'reached goal' if reached else 'missed goal'})")

    print(f"\n  {'─'*45}")
    print(f"  {label} SUMMARY over {n_episodes} episodes:")
    print(f"  Avg return:  {total_return/n_episodes:.3f}")
    print(f"  Goal rate:   {n_goals}/{n_episodes} "
          f"({100*n_goals/n_episodes:.0f}%)")
    print(f"  Total trap hits: {trap_hits}")


def main():
    # ── Load trained policies ─────────────────────────────────────────
    # We need to retrain from scratch to get the final policy weights.
    # Load from saved results if you have them, otherwise retrain.
    
    # Check if saved policies exist
    bc_path   = "results/data/bc_policy.pt"
    tvbc_path = "results/data/tvbc_policy.pt"

    if os.path.exists(bc_path) and os.path.exists(tvbc_path):
        print("Loading saved policies...")
        bc_policy   = PolicyNetworkMapObs(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
        tvbc_policy = PolicyNetworkMapObs(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
        bc_policy.load_state_dict(  torch.load(bc_path))
        tvbc_policy.load_state_dict(torch.load(tvbc_path))
        print("Policies loaded.")
    else:
        print("No saved policies found.")
        print("Please run the experiment first, then save policies.")
        print("Add these lines at the end of run_generalisation.py:")
        print("  torch.save(bc_policy.state_dict(),   'results/data/bc_policy.pt')")
        print("  torch.save(tvbc_policy.state_dict(), 'results/data/tvbc_policy.pt')")
        return

    # ── Pick test maps to visualise ───────────────────────────────────
    # Change these seeds to see different test maps
    TEST_MAP_SEEDS = [5000, 5001, 5005, 5010, 5019]

    for seed in TEST_MAP_SEEDS:
        env = make_env(seed)
        tch = make_teacher(env)

        # Print map layout
        print(f"\n{'═'*55}")
        print(f"  TEST MAP (seed={seed}) — never used for training")
        print(f"{'═'*55}")
        env.print_map()

        # Teacher reference
        t_return = 0.0
        for ep in range(10):
            start = env.START_CELL if ep < 5 \
                    else np.random.randint(0, 64)
            state = env.reset(start_pos=start)
            done  = False; ep_r = 0.0
            while not done:
                state, r, done, _ = env.step(tch.act_greedy(state))
                ep_r += r
            t_return += ep_r
        print(f"  Teacher avg return (reference): {t_return/10:.3f}")

        # Show BC behaviour
        rollout_and_show(bc_policy,   env, "STANDARD BC",   n_episodes=5)

        # Show TV-BC behaviour
        rollout_and_show(tvbc_policy, env, "TV-BC (ITAL)",  n_episodes=5)

        print(f"\n{'═'*55}")
        input("  Press Enter to see next map...")


if __name__ == "__main__":
    main()
