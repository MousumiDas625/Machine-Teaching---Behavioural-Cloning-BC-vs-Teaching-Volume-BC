# evaluate_experts.py
#
# Evaluate Pavel coinrun checkpoints.
# Choose one or all checkpoints to evaluate on:
#   1. User-specified fixed levels
#   2. User-specified number of random levels
#
# Checkpoints:
#   0p  = 100% random goal during training
#   50p = 50% fixed goal, 50% random goal during training
#   100p = 100% fixed/deterministic goal during training
#
# Usage:
#   python3 evaluate_experts.py
#   (fully interactive — prompts for checkpoint, levels, episodes)
#
#   python3 evaluate_experts.py --checkpoint 0p
#   python3 evaluate_experts.py --checkpoint 50p
#   python3 evaluate_experts.py --checkpoint 100p
#   python3 evaluate_experts.py --checkpoint all
#
#   python3 evaluate_experts.py --checkpoint 100p \
#       --fixed_levels 0 1 5 10 20 50 \
#       --fixed_eps 20 \
#       --random_eps 100

import sys
import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".")))

from agents.pavel_policy import load_pavel_expert

# ── Checkpoint paths ──────────────────────────────────────────────────
CHECKPOINTS = {
    "0p   (100% random goal during training)": (
        "/scr/pavel/data/goal-misgen/policy/icml/coinrun/"
        "icml2_coinrun_exp0_0p/2026-01-13__06-46-28__seed_6033/"
        "model_200015872.pth"
    ),
    "50p  (50% fixed + 50% random goal during training)": (
        "/scr/pavel/data/goal-misgen/policy/icml/coinrun/"
        "icml2_coinrun_exp0_50p/2026-01-13__14-46-26__seed_6033/"
        "model_200015872.pth"
    ),
    "100p (100% fixed/deterministic goal during training)": (
        "/scr/pavel/data/goal-misgen/policy/icml/coinrun/"
        "icml2_100p/2026-01-13__14-46-27__seed_6033/"
        "model_200015872.pth"
    ),
}

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_FIXED_LEVELS  = [0, 1, 2, 5, 10, 20, 50, 100]
DEFAULT_FIXED_EPS     = 10     # episodes per fixed level
DEFAULT_RANDOM_EPS    = 50     # total random episodes
DEFAULT_MAX_STEPS     = 500    # max steps per episode


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Pavel coinrun experts")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        choices=["0p", "50p", "100p", "all"],
        help="Which checkpoint to evaluate: 0p, 50p, 100p, or all")
    parser.add_argument(
        "--fixed_levels", type=int, nargs="+", default=None,
        help=f"Level numbers to test (default: {DEFAULT_FIXED_LEVELS})")
    parser.add_argument(
        "--fixed_eps", type=int, default=None,
        help=f"Episodes per fixed level (default: {DEFAULT_FIXED_EPS})")
    parser.add_argument(
        "--random_eps", type=int, default=None,
        help=f"Total random level episodes (default: {DEFAULT_RANDOM_EPS})")
    parser.add_argument(
        "--max_steps", type=int, default=DEFAULT_MAX_STEPS,
        help=f"Max steps per episode (default: {DEFAULT_MAX_STEPS})")
    parser.add_argument(
        "--no_prompt", action="store_true",
        help="Skip interactive prompts and use defaults")
    return parser.parse_args()


def prompt_checkpoint():
    """Interactively ask user which checkpoint to evaluate."""
    print("\n  Available checkpoints:")
    print("  [1]  0p  — trained with 100% RANDOM goal location")
    print("  [2]  50p — trained with 50% fixed + 50% random goal")
    print("  [3]  100p — trained with 100% FIXED/deterministic goal")
    print("  [4]  all — evaluate all three side by side")
    print("\n  Enter 1, 2, 3, or 4 (default: 4 = all):")
    raw = input("  > ").strip()
    mapping = {"1": "0p", "2": "50p", "3": "100p", "4": "all", "": "all"}
    choice = mapping.get(raw, None)
    if choice is None:
        # Try direct name input e.g. "0p", "50p", "100p", "all"
        if raw in ["0p", "50p", "100p", "all"]:
            choice = raw
        else:
            print(f"  Invalid input '{raw}', using all checkpoints")
            choice = "all"
    print(f"  Selected: {choice}")
    return choice


def prompt_levels(default):
    print(f"\n  Fixed levels to test (default: {default})")
    print("  Enter space-separated level numbers, or press Enter for default:")
    raw = input("  > ").strip()
    if raw == "":
        return default
    try:
        levels = [int(x) for x in raw.split()]
        print(f"  Using levels: {levels}")
        return levels
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def prompt_int(name, default):
    print(f"\n  Number of {name} (default: {default}):")
    raw = input("  > ").strip()
    if raw == "":
        return default
    try:
        val = int(raw)
        print(f"  Using: {val}")
        return val
    except ValueError:
        print(f"  Invalid input, using default: {default}")
        return default


def eval_fixed_levels(expert, levels, n_eps, max_steps):
    """Evaluate expert on specific fixed levels."""
    import procgen
    results = {}
    for level in levels:
        env = procgen.ProcgenEnv(
            num_envs=1, env_name="coinrun",
            num_levels=1, start_level=level,
            distribution_mode="easy")
        solved = 0
        total_steps = 0
        ep_rewards = []
        for ep in range(n_eps):
            obs  = env.reset()["rgb"][0]
            done = False; steps = 0; ep_r = 0.0
            while not done and steps < max_steps:
                action = expert.act_greedy(obs)
                od, ra, da, _ = env.step(np.array([action]))
                obs   = od["rgb"][0]
                ep_r += float(ra[0])
                done  = bool(da[0])
                steps += 1
            ep_rewards.append(ep_r)
            if ep_r > 0:
                solved += 1
            total_steps += steps
        env.close()
        results[level] = {
            "solved"    : solved,
            "total"     : n_eps,
            "pct"       : solved * 100 // n_eps,
            "avg_steps" : total_steps / n_eps,
            "avg_reward": float(np.mean(ep_rewards)),
        }
    return results


def eval_random_levels(expert, n_eps, max_steps):
    """Evaluate expert on random levels (full distribution)."""
    import procgen
    env = procgen.ProcgenEnv(
        num_envs=1, env_name="coinrun",
        num_levels=0, start_level=0,
        distribution_mode="easy")
    solved = 0
    total_steps = 0
    ep_rewards = []
    for ep in range(n_eps):
        obs  = env.reset()["rgb"][0]
        done = False; steps = 0; ep_r = 0.0
        while not done and steps < max_steps:
            action = expert.act_greedy(obs)
            od, ra, da, _ = env.step(np.array([action]))
            obs   = od["rgb"][0]
            ep_r += float(ra[0])
            done  = bool(da[0])
            steps += 1
        ep_rewards.append(ep_r)
        if ep_r > 0:
            solved += 1
        total_steps += steps
    env.close()
    return {
        "solved"    : solved,
        "total"     : n_eps,
        "pct"       : solved * 100 // n_eps,
        "avg_steps" : total_steps / n_eps,
        "avg_reward": float(np.mean(ep_rewards)),
    }


def print_fixed_results(results, n_eps):
    print(f"\n  {'Level':>6} | {'Solved':>8} | {'Rate':>6} | "
          f"{'Avg Steps':>10} | {'Avg Reward':>11}")
    print(f"  {'─'*55}")
    for level, r in results.items():
        bar = "█" * (r["pct"] // 10) + "░" * (10 - r["pct"] // 10)
        print(f"  {level:>6} | "
              f"{r['solved']:>3}/{r['total']:<4} | "
              f"{r['pct']:>5}% | "
              f"{r['avg_steps']:>10.1f} | "
              f"{r['avg_reward']:>11.3f}  {bar}")


def print_random_results(r):
    bar = "█" * (r["pct"] // 10) + "░" * (10 - r["pct"] // 10)
    print(f"\n  Solved: {r['solved']}/{r['total']} ({r['pct']}%)  {bar}")
    print(f"  Avg steps/episode: {r['avg_steps']:.1f}")
    print(f"  Avg reward/episode: {r['avg_reward']:.3f}")


def main():
    args = get_args()

    print(f"\n{'='*65}")
    print(f"  Pavel CoinRun Expert Evaluation")
    print(f"{'='*65}")

    # ── Step 1: Choose checkpoint ─────────────────────────────────────
    if args.no_prompt and args.checkpoint is None:
        checkpoint_choice = "all"
    elif args.checkpoint is not None:
        checkpoint_choice = args.checkpoint
    else:
        checkpoint_choice = prompt_checkpoint()

    # Filter CHECKPOINTS dict based on choice
    KEY_MAP = {
        "0p" : "0p   (100% random goal during training)",
        "50p": "50p  (50% fixed + 50% random goal during training)",
        "100p":"100p (100% fixed/deterministic goal during training)",
    }
    if checkpoint_choice == "all":
        selected_checkpoints = CHECKPOINTS
        print(f"\n  Evaluating all 3 checkpoints")
    else:
        key = KEY_MAP[checkpoint_choice]
        selected_checkpoints = {key: CHECKPOINTS[key]}
        print(f"\n  Evaluating: {key}")

    # ── Step 2: Get evaluation parameters ────────────────────────────
    if args.no_prompt or args.fixed_levels is not None:
        fixed_levels = args.fixed_levels or DEFAULT_FIXED_LEVELS
        fixed_eps    = args.fixed_eps    or DEFAULT_FIXED_EPS
        random_eps   = args.random_eps   or DEFAULT_RANDOM_EPS
    else:
        print("\n  Configure evaluation parameters:")
        fixed_levels = prompt_levels(DEFAULT_FIXED_LEVELS)
        fixed_eps    = prompt_int("episodes per fixed level",
                                  DEFAULT_FIXED_EPS)
        random_eps   = prompt_int("random level episodes",
                                  DEFAULT_RANDOM_EPS)

    max_steps = args.max_steps

    print(f"\n  Settings:")
    print(f"    Fixed levels:       {fixed_levels}")
    print(f"    Episodes per level: {fixed_eps}")
    print(f"    Random episodes:    {random_eps}")
    print(f"    Max steps/episode:  {max_steps}")

    # ── Step 3: Run evaluation ────────────────────────────────────────
    summary = {}

    for name, ckpt_path in selected_checkpoints.items():
        print(f"\n{'='*65}")
        print(f"  Checkpoint: {name}")
        print(f"{'='*65}")

        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found at {ckpt_path}")
            print(f"  Skipping...")
            continue

        expert = load_pavel_expert(ckpt_path)

        print(f"\n  --- Fixed Levels ({fixed_eps} eps each) ---")
        fixed_results = eval_fixed_levels(
            expert, fixed_levels, fixed_eps, max_steps)
        print_fixed_results(fixed_results, fixed_eps)

        print(f"\n  --- Random Levels ({random_eps} episodes) ---")
        random_result = eval_random_levels(expert, random_eps, max_steps)
        print_random_results(random_result)

        summary[name] = {
            "fixed" : fixed_results,
            "random": random_result,
        }

    # ── Step 4: Summary (only shown when evaluating all) ─────────────
    if len(summary) > 1:
        print(f"\n{'='*65}")
        print(f"  SUMMARY COMPARISON")
        print(f"{'='*65}")
        print(f"\n  Random level solve rate (overall performance):")
        print(f"  {'Checkpoint':<50} | {'Solve Rate':>10}")
        print(f"  {'─'*63}")
        for name, res in summary.items():
            r = res["random"]
            print(f"  {name:<50} | "
                  f"{r['solved']}/{r['total']} ({r['pct']}%)")

        print(f"\n  Fixed level solve rates:")
        header = f"  {'Level':>6}"
        for name in summary:
            short = name.split("(")[0].strip()
            header += f" | {short:>8}"
        print(header)
        print(f"  {'─'*55}")
        for level in fixed_levels:
            row = f"  {level:>6}"
            for name, res in summary.items():
                pct = res["fixed"].get(level, {}).get("pct", "N/A")
                row += f" | {pct:>7}%"
            print(row)

        print(f"\n  Recommendation for teaching experiment:")
        best_name = max(summary.keys(),
                        key=lambda n: summary[n]["random"]["pct"])
        print(f"  Best overall: {best_name}")
        best_levels = [
            level for level in fixed_levels
            if all(summary[n]["fixed"].get(level, {}).get("pct", 0) >= 80
                   for n in summary)
        ]
        if best_levels:
            print(f"  Levels all checkpoints solve 80%+: {best_levels}")
            print(f"  Recommended training level: {best_levels[0]}")
        else:
            for name, res in summary.items():
                good = [l for l in fixed_levels
                        if res["fixed"].get(l, {}).get("pct", 0) >= 80]
                short = name.split("(")[0].strip()
                print(f"  {short} solves 80%+ on levels: {good}")


if __name__ == "__main__":
    main()
