# experiments/run_growing_terminal.py
#
# Growing dataset experiment using the TERMINAL GridWorld.
#
# Key difference from run_growing.py:
#   - Uses GridWorldTerminal — episode ends when agent reaches a goal
#   - Agent cannot farm reward by sitting at goal cell
#   - Task is harder: agent must find shortest safe path to goal
#   - Teacher return is now much lower (path length dependent)
#   - TV-BC advantage should be MORE visible because:
#       1. Noisy demos that walk into traps are actively harmful
#          (trap = -1.0 AND wastes steps toward goal)
#       2. There are fewer total steps per episode (episode ends at goal)
#          so every demonstration step carries more weight
#       3. The optimal path is non-trivial (must navigate around traps)
#
# Everything else is identical to run_growing.py:
#   - Growing dataset (frozen TV scores for old trajectories)
#   - TV recomputed for new trajectories using current θ
#   - Softmax selection for TV-BC, random selection for BC
#   - 200 training steps per round

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.policy_net              import PolicyNetwork
from agents.rl_teacher              import RLTeacher
from gridworld_env.gridworld_terminal import GridWorldTerminal
from learners.standard_bc           import StandardBC
from learners.tv_bc                 import TeacherAwareBC
from teaching.teaching_volume       import compute_tv_single

# ══════════════════════════════════════════════════════════════════════
# Configurable parameters
# ══════════════════════════════════════════════════════════════════════
N_ROUNDS        = 20
TRAJS_PER_ROUND = 5
TRAJ_LENGTH     = 50      # max steps per episode
NOISE_EPS       = 0.30
TRAIN_STEPS     = 200
BATCH_SIZE      = 20
K_SELECT        = 0       # 0 = use all accumulated trajectories
BETA_TV         = 5.0
ETA             = 0.01
GAMMA           = 0.99
SOFTMAX_TEMP    = 0.1
N_EVAL_EPS      = 50
STEP_PENALTY    = 0.0   # small penalty per non-goal step
MAP_MODE        = 'random'
N_MAPS          = 5
SEED_BASE       = 42
SEED_INIT       = 0

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Helper: collect one trajectory
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, noisy=True, eps=NOISE_EPS, seed=0):
    """
    Roll out teacher for one episode.
    Episode ends when goal reached OR max_steps hit.

    Returns list of (state_np, action_int) tuples.
    May be shorter than TRAJ_LENGTH if goal is reached early.
    """
    rng_noise = np.random.default_rng(seed)
    state     = env.reset(start_pos=env.START_CELL)
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

# ══════════════════════════════════════════════════════════════════════
# Helper: compute TV scores for new trajectories
# ══════════════════════════════════════════════════════════════════════
def compute_tv_for_trajectories(trajs, policy, teacher, eta=ETA):
    """
    One TV score per trajectory = mean TV across all steps.
    Uses CURRENT policy weights — called only for new trajectories.
    """
    tv_scores = []
    for traj in trajs:
        if len(traj) == 0:
            tv_scores.append(0.0)
            continue
        step_tvs = []
        for (state_np, action) in traj:
            tv = compute_tv_single(
                state_np = state_np,
                action   = action,
                learner  = policy,
                teacher  = teacher,
                eta      = eta
            )
            step_tvs.append(tv)
        tv_scores.append(float(np.mean(step_tvs)))
    return tv_scores

# ══════════════════════════════════════════════════════════════════════
# Helper: softmax selection
# ══════════════════════════════════════════════════════════════════════
def softmax_select(all_trajs, all_tv_scores, K, beta, rng):
    N = len(all_trajs)
    if K == 0 or K >= N:
        return all_trajs, list(range(N))
    tv_arr  = np.array(all_tv_scores, dtype=np.float64)
    tv_arr -= tv_arr.max()
    weights = np.exp(beta * tv_arr)
    probs   = weights / weights.sum()
    idx     = list(rng.choice(N, size=K, replace=False, p=probs))
    return [all_trajs[i] for i in idx], idx

# ══════════════════════════════════════════════════════════════════════
# Helper: flatten trajectories
# ══════════════════════════════════════════════════════════════════════
def flatten(trajs):
    states, actions = [], []
    for traj in trajs:
        for (s, a) in traj:
            states.append(s)
            actions.append(a)
    return states, actions

# ══════════════════════════════════════════════════════════════════════
# Helper: evaluate policy
# ══════════════════════════════════════════════════════════════════════
def evaluate_policy(policy, env, n_episodes=N_EVAL_EPS, seed=99):
    """
    Average return over n_episodes greedy rollouts.
    Also tracks: how often goal is reached, average steps to goal.
    """
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    goals_reached = 0
    total_steps   = 0
    policy.eval()

    for ep in range(n_episodes):
        start = env.START_CELL if ep < n_episodes // 2 \
                else int(rng.integers(0, env.n_states))
        state     = env.reset(start_pos=start)
        done      = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done:
            state_vec = env.get_state_vector(state)
            probs     = policy.get_action_probs(state_vec).numpy()
            action    = int(np.argmax(probs))
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps  += 1
            if info.get('reached_goal', False):
                goals_reached += 1

        total_reward += ep_reward
        total_steps  += ep_steps

    avg_return     = total_reward / n_episodes
    goal_rate      = goals_reached / n_episodes
    avg_steps      = total_steps   / n_episodes
    return avg_return, goal_rate, avg_steps

# ══════════════════════════════════════════════════════════════════════
# Single map experiment
# ══════════════════════════════════════════════════════════════════════
def run_one_map(map_seed, map_mode=MAP_MODE):
    print(f"\n{'═'*60}")
    print(f"  Map seed={map_seed}  mode={map_mode}  "
          f"(TERMINAL goals)")
    print(f"{'═'*60}")

    # Build terminal environment
    env = GridWorldTerminal(
        grid_size    = 8,
        max_steps    = TRAJ_LENGTH,
        seed         = map_seed,
        map_mode     = map_mode,
        step_penalty = STEP_PENALTY
    )
    env.print_map()

    # Train teacher on this map's reward
    teacher = RLTeacher(
        omega_star   = env.omega_star,
        grid_size    = env.grid_size,
        gamma        = GAMMA,
        softmax_temp = SOFTMAX_TEMP
    )
    teacher.train(verbose=True)

    # Evaluate teacher
    total = 0.0
    goals = 0
    rng_t = np.random.default_rng(99)
    for ep in range(N_EVAL_EPS):
        start = env.START_CELL if ep < N_EVAL_EPS//2 \
                else int(rng_t.integers(0, env.n_states))
        state = env.reset(start_pos=start)
        done  = False; ep_r = 0.0
        while not done:
            action = teacher.act_greedy(state)
            state, r, done, info = env.step(action)
            ep_r += r
            if info.get('reached_goal', False):
                goals += 1
        total += ep_r
    teacher_ret  = total / N_EVAL_EPS
    teacher_goal = goals / N_EVAL_EPS
    print(f"  Teacher return: {teacher_ret:.3f}  "
          f"goal_rate={teacher_goal:.2f}")

    # Initialise both learners with identical weights
    bc_policy   = PolicyNetwork(64, 4, 64, seed=SEED_INIT)
    tvbc_policy = PolicyNetwork(64, 4, 64, seed=SEED_INIT)
    bc_learner   = StandardBC(
        policy=bc_policy, eta=ETA, seed=SEED_BASE)
    tvbc_learner = TeacherAwareBC(
        policy=tvbc_policy, teacher=teacher,
        eta=ETA, beta=BETA_TV, seed=SEED_BASE)

    # Accumulators — TV scores frozen at collection time
    bc_all_trajs   = []
    bc_all_tv      = []
    tvbc_all_trajs = []
    tvbc_all_tv    = []

    bc_returns    = []
    tvbc_returns  = []
    bc_goal_rates = []
    tvbc_goal_rates = []
    round_sizes   = []

    traj_seed_ctr = 0

    print(f"\n  {'Round':>5} | {'Dataset':>7} | "
          f"{'BC ret':>8} | {'BC goal%':>8} | "
          f"{'TVBC ret':>8} | {'TVBC goal%':>10} | "
          f"{'TV(BC)':>8} | {'TV(TVBC)':>8}")
    print(f"  {'─'*85}")

    for round_idx in range(N_ROUNDS):

        # Step 1: Collect new trajectories
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            traj = collect_trajectory(
                env, teacher,
                noisy=True, eps=NOISE_EPS,
                seed=traj_seed_ctr)
            new_trajs.append(traj)
            traj_seed_ctr += 1

        # Step 2: Compute TV for new trajs using CURRENT θ
        # Old trajs keep their frozen TV scores
        new_bc_tv   = compute_tv_for_trajectories(
            new_trajs, bc_policy,   teacher, eta=ETA)
        new_tvbc_tv = compute_tv_for_trajectories(
            new_trajs, tvbc_policy, teacher, eta=ETA)

        # Step 3: Accumulate
        bc_all_trajs.extend(new_trajs)
        bc_all_tv.extend(new_bc_tv)
        tvbc_all_trajs.extend(new_trajs)
        tvbc_all_tv.extend(new_tvbc_tv)

        # Dataset size = total (s,a) pairs accumulated
        dataset_size = sum(len(t) for t in bc_all_trajs)
        round_sizes.append(dataset_size)

        # Step 4: Select trajectories for training
        rng_sel = np.random.default_rng(SEED_BASE + round_idx)

        tvbc_selected, _ = softmax_select(
            tvbc_all_trajs, tvbc_all_tv,
            K=K_SELECT, beta=BETA_TV, rng=rng_sel)

        n_select   = len(tvbc_selected)
        rng_bc_sel = np.random.default_rng(
            SEED_BASE + round_idx + 1000)
        bc_idx     = list(rng_bc_sel.choice(
            len(bc_all_trajs),
            size=min(n_select, len(bc_all_trajs)),
            replace=False))
        bc_selected = [bc_all_trajs[i] for i in bc_idx]

        # Step 5: Train both learners for TRAIN_STEPS
        pool_bc   = list(zip(*flatten(bc_selected))) \
                    if flatten(bc_selected)[0] else []
        pool_tvbc = list(zip(*flatten(tvbc_selected))) \
                    if flatten(tvbc_selected)[0] else []

        bc_pairs   = list(zip(*flatten(bc_selected)))
        tvbc_pairs = list(zip(*flatten(tvbc_selected)))

        bc_pool   = [(s, a) for s, a in
                     zip(flatten(bc_selected)[0],
                         flatten(bc_selected)[1])]
        tvbc_pool = [(s, a) for s, a in
                     zip(flatten(tvbc_selected)[0],
                         flatten(tvbc_selected)[1])]

        rng_train = np.random.default_rng(
            SEED_BASE + round_idx + 2000)

        for _ in range(TRAIN_STEPS):
            bs = min(BATCH_SIZE, len(bc_pool))
            ts = min(BATCH_SIZE, len(tvbc_pool))
            if bs == 0 or ts == 0:
                break

            idx_bc   = rng_train.choice(
                len(bc_pool),   size=bs, replace=False)
            idx_tvbc = rng_train.choice(
                len(tvbc_pool), size=ts, replace=False)

            batch_bc   = [bc_pool[i]   for i in idx_bc]
            batch_tvbc = [tvbc_pool[i] for i in idx_tvbc]

            bc_learner.step(batch_bc)
            tvbc_learner.step(batch_tvbc)

        # Step 6: Evaluate
        bc_ret, bc_gr, bc_steps     = evaluate_policy(
            bc_policy,   env)
        tvbc_ret, tvbc_gr, tvbc_steps = evaluate_policy(
            tvbc_policy, env)

        bc_returns.append(bc_ret)
        tvbc_returns.append(tvbc_ret)
        bc_goal_rates.append(bc_gr)
        tvbc_goal_rates.append(tvbc_gr)

        mean_bc_tv   = float(np.mean(bc_all_tv))
        mean_tvbc_tv = float(np.mean(tvbc_all_tv))

        print(f"  {round_idx+1:>5} | "
              f"{dataset_size:>7} | "
              f"{bc_ret:>8.3f} | "
              f"{bc_gr*100:>7.1f}% | "
              f"{tvbc_ret:>8.3f} | "
              f"{tvbc_gr*100:>9.1f}% | "
              f"{mean_bc_tv:>8.4f} | "
              f"{mean_tvbc_tv:.4f}")

    return (bc_returns, tvbc_returns,
            bc_goal_rates, tvbc_goal_rates,
            round_sizes, teacher_ret)

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print(f"{'═'*60}")
    print(f"  Growing Dataset Experiment — TERMINAL GridWorld")
    print(f"{'═'*60}")
    print(f"  Rounds:          {N_ROUNDS}")
    print(f"  Trajs per round: {TRAJS_PER_ROUND}")
    print(f"  Train steps:     {TRAIN_STEPS}")
    print(f"  Beta TV:         {BETA_TV}")
    print(f"  ETA:             {ETA}")
    print(f"  Step penalty:    {STEP_PENALTY}")
    print(f"  Map mode:        {MAP_MODE}  (N_MAPS={N_MAPS})")

    all_bc_ret   = []
    all_tvbc_ret = []
    all_bc_gr    = []
    all_tvbc_gr  = []
    all_teacher  = []

    for map_i in range(N_MAPS):
        seed = SEED_BASE + map_i * 100
        bc_r, tvbc_r, bc_gr, tvbc_gr, sizes, t_ret = \
            run_one_map(map_seed=seed, map_mode=MAP_MODE)

        all_bc_ret.append(bc_r)
        all_tvbc_ret.append(tvbc_r)
        all_bc_gr.append(bc_gr)
        all_tvbc_gr.append(tvbc_gr)
        all_teacher.append(t_ret)

    # Average over maps
    bc_mean    = np.mean(all_bc_ret,   axis=0)
    tvbc_mean  = np.mean(all_tvbc_ret, axis=0)
    bc_std     = np.std(all_bc_ret,    axis=0)
    tvbc_std   = np.std(all_tvbc_ret,  axis=0)
    bc_gr_mean = np.mean(all_bc_gr,    axis=0)
    tvbc_gr_mean = np.mean(all_tvbc_gr, axis=0)
    teacher_mean = float(np.mean(all_teacher))

    print(f"\n{'═'*60}")
    print(f"  FINAL RESULTS — TERMINAL GridWorld "
          f"(averaged over {N_MAPS} maps)")
    print(f"{'═'*60}")
    print(f"  Teacher return:      {teacher_mean:.3f}")
    print(f"  BC   final return:   {bc_mean[-1]:.3f}  "
          f"({100*bc_mean[-1]/teacher_mean:.1f}% of teacher)  "
          f"goal_rate={bc_gr_mean[-1]*100:.1f}%")
    print(f"  TV-BC final return:  {tvbc_mean[-1]:.3f}  "
          f"({100*tvbc_mean[-1]/teacher_mean:.1f}% of teacher)  "
          f"goal_rate={tvbc_gr_mean[-1]*100:.1f}%")
    print(f"  Difference (TV-BC - BC): "
          f"{tvbc_mean[-1]-bc_mean[-1]:+.3f}")

    # Save
    results = {
        "bc_mean"      : bc_mean.tolist(),
        "tvbc_mean"    : tvbc_mean.tolist(),
        "bc_std"       : bc_std.tolist(),
        "tvbc_std"     : tvbc_std.tolist(),
        "bc_gr_mean"   : bc_gr_mean.tolist(),
        "tvbc_gr_mean" : tvbc_gr_mean.tolist(),
        "round_sizes"  : sizes,
        "teacher_ret"  : teacher_mean,
        "N_ROUNDS"     : N_ROUNDS,
        "TRAJS_PER_ROUND": TRAJS_PER_ROUND,
        "BETA_TV"      : BETA_TV,
        "STEP_PENALTY" : STEP_PENALTY,
    }
    with open("results/data/growing_terminal_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/growing_terminal_results.pkl")

    # Plot
    rounds = np.arange(1, N_ROUNDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Return curves
    axes[0].plot(rounds, bc_mean,   color="#e07b39",
                 lw=2, label="Standard BC")
    axes[0].plot(rounds, tvbc_mean, color="#3a7ebf",
                 lw=2, label="TV-BC (ITAL)")
    if N_MAPS > 1:
        axes[0].fill_between(rounds,
            bc_mean - bc_std, bc_mean + bc_std,
            color="#e07b39", alpha=0.2)
        axes[0].fill_between(rounds,
            tvbc_mean - tvbc_std, tvbc_mean + tvbc_std,
            color="#3a7ebf", alpha=0.2)
    axes[0].axhline(y=teacher_mean, color="green",
                    linestyle="--", lw=1.5,
                    label=f"Teacher ({teacher_mean:.2f})")
    axes[0].set_xlabel("Teaching Round", fontsize=12)
    axes[0].set_ylabel("Average Return", fontsize=12)
    axes[0].set_title("Policy Return — Terminal GridWorld", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Goal rate curves
    axes[1].plot(rounds, bc_gr_mean,   color="#e07b39",
                 lw=2, label="Standard BC",  marker="o", ms=4)
    axes[1].plot(rounds, tvbc_gr_mean, color="#3a7ebf",
                 lw=2, label="TV-BC (ITAL)", marker="s", ms=4)
    axes[1].set_xlabel("Teaching Round", fontsize=12)
    axes[1].set_ylabel("Goal Reached Rate", fontsize=12)
    axes[1].set_title("How Often Agent Reaches Goal\n"
                      "(Terminal GridWorld)", fontsize=12)
    axes[1].set_ylim([0, 1.05])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Terminal GridWorld  |  step_penalty={STEP_PENALTY}  "
        f"beta={BETA_TV}  |  {N_MAPS} maps",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("results/plots/growing_terminal_curves.png", dpi=150)
    plt.close()
    print(f"  Plot: results/plots/growing_terminal_curves.png")
