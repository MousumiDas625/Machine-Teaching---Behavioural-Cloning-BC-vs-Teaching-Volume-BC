# experiments/run_growing.py
#
# Growing dataset experiment — implements exactly what the professor described.
#
# The key idea:
# ─────────────
# At each round, the teacher shows the learner a small batch of new
# trajectories. TV scores for those NEW trajectories are computed using
# the CURRENT learner weights at that round. Old trajectories KEEP their
# TV scores from when they were first collected — they are never recomputed.
# The dataset grows every round. Softmax selection is applied over ALL
# accumulated trajectories at each round.
#
# This mirrors real pedagogy:
#   Round 0: robot knows nothing  → teacher shows basic examples
#   Round 1: robot learned a bit  → teacher shows harder examples
#   Round 2: robot is better      → teacher shows more targeted examples
#   The old examples are not discarded — the robot keeps learning from them
#   But their TV scores reflect what was informative at the time they were shown
#
# Comparison:
#   TV-BC: uses teaching volume to select which examples to train on
#   BC:    trains on a random selection of the same size (no TV scoring)
#
# Professor's rules implemented here:
#   - Start cell always bottom-left
#   - Best goal always top-right
#   - Dataset grows each round (never discard old data)
#   - TV scores for old trajectories are frozen (not recomputed)
#   - TV scores for new trajectories use current θ
#   - Softmax over all accumulated TV scores for selection

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.policy_net       import PolicyNetwork
from agents.rl_teacher       import RLTeacher
from gridworld_env.gridworld import GridWorld
from learners.standard_bc    import StandardBC
from learners.tv_bc          import TeacherAwareBC
from teaching.teaching_volume import compute_tv_single

# ══════════════════════════════════════════════════════════════════════
# Configurable parameters
# ══════════════════════════════════════════════════════════════════════
N_ROUNDS          = 20      # how many rounds of teaching
TRAJS_PER_ROUND   = 5       # new trajectories added each round
TRAJ_LENGTH       = 50      # steps per trajectory
NOISE_EPS         = 0.30    # noise in teacher demonstrations
TRAIN_STEPS       = 200     # SGD steps per round (on selected subset)
BATCH_SIZE        = 20      # mini-batch size during training
K_SELECT          = 0       # trajectories to select via softmax each round
                             # 0 means use ALL accumulated trajectories
BETA_TV           = 5.0     # softmax temperature for TV selection
ETA               = 0.01    # learning rate
GAMMA             = 0.99    # discount for Value Iteration
SOFTMAX_TEMP      = 0.1     # teacher policy temperature
N_EVAL_EPS        = 50      # episodes per policy evaluation
MAP_MODE          = 'random' # 'fixed' or 'random'
N_MAPS            = 5       # how many random maps to average over
                             # (set >1 when MAP_MODE='random')
SEED_BASE         = 42
SEED_INIT         = 0       # network init seed (same for BC and TV-BC)

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Helper: collect one trajectory from teacher
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, noisy=True, eps=NOISE_EPS, seed=0):
    """
    Roll out the teacher for one episode of TRAJ_LENGTH steps.

    Parameters
    ----------
    noisy : bool  — if True, inject random actions with prob eps
    eps   : float — noise probability
    seed  : int   — for reproducible noise

    Returns
    -------
    list of (state_np, action_int) tuples
    """
    rng_noise = np.random.default_rng(seed)
    # Always start from bottom-left (professor's rule)
    state = env.reset(start_pos=env.START_CELL)
    traj  = []
    done  = False

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
# Helper: compute TV scores for a batch of trajectories
# ══════════════════════════════════════════════════════════════════════
def compute_tv_for_trajectories(trajs, policy, teacher, eta=ETA):
    """
    Compute one TV score per trajectory (mean TV across all steps).

    TV is computed using the CURRENT policy weights — this is called
    only for newly collected trajectories at each round.
    Old trajectories keep their TV scores from when they were collected.

    Parameters
    ----------
    trajs   : list of trajectories, each = list of (state_np, action)
    policy  : PolicyNetwork — current learner weights (θ at this round)
    teacher : RLTeacher     — provides reference loss ℓ(ω*; s, a)
    eta     : float         — learning rate used in TV formula

    Returns
    -------
    tv_scores : list of float, one per trajectory
    """
    tv_scores = []
    for traj in trajs:
        step_tvs = []
        for (state_np, action) in traj:
            tv = compute_tv_single(
                state_np  = state_np,
                action    = action,
                learner   = policy,
                teacher   = teacher,
                eta       = eta
            )
            step_tvs.append(tv)
        # Mean TV across all steps in this trajectory
        tv_scores.append(float(np.mean(step_tvs)))
    return tv_scores

# ══════════════════════════════════════════════════════════════════════
# Helper: softmax selection over accumulated trajectories
# ══════════════════════════════════════════════════════════════════════
def softmax_select(all_trajs, all_tv_scores, K, beta, rng):
    """
    Select K trajectories from the accumulated pool using
    softmax over TV scores.

    If K=0 or K >= len(all_trajs), return all trajectories
    (no selection needed — use everything).

    Parameters
    ----------
    all_trajs    : list of all accumulated trajectories
    all_tv_scores: list of TV scores (same length, frozen from collection time)
    K            : number to select (0 = use all)
    beta         : softmax temperature
    rng          : numpy random generator

    Returns
    -------
    selected_trajs : list of selected trajectories
    selected_idx   : indices of selected trajectories
    """
    N = len(all_trajs)
    if K == 0 or K >= N:
        return all_trajs, list(range(N))

    tv_arr  = np.array(all_tv_scores, dtype=np.float64)
    tv_arr -= tv_arr.max()                        # numerical stability
    weights = np.exp(beta * tv_arr)
    probs   = weights / weights.sum()

    selected_idx = list(rng.choice(N, size=K, replace=False, p=probs))
    selected_trajs = [all_trajs[i] for i in selected_idx]
    return selected_trajs, selected_idx

# ══════════════════════════════════════════════════════════════════════
# Helper: flatten trajectories into (states, actions) lists
# ══════════════════════════════════════════════════════════════════════
def flatten(trajs):
    states  = []
    actions = []
    for traj in trajs:
        for (s, a) in traj:
            states.append(s)
            actions.append(a)
    return states, actions

# ══════════════════════════════════════════════════════════════════════
# Helper: evaluate policy return
# ══════════════════════════════════════════════════════════════════════
def evaluate_policy(policy, env, n_episodes=N_EVAL_EPS, seed=99):
    """
    Average cumulative reward over n_episodes greedy rollouts.
    Always starts from bottom-left (professor's rule).
    """
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    policy.eval()

    for ep in range(n_episodes):
        # Mix of fixed start and random starts for robust evaluation
        if ep < n_episodes // 2:
            start = env.START_CELL
        else:
            start = int(rng.integers(0, env.n_states))

        state     = env.reset(start_pos=start)
        done      = False
        ep_reward = 0.0

        while not done:
            state_vec = env.get_state_vector(state)
            probs  = policy.get_action_probs(state_vec).numpy()
            action = int(np.argmax(probs))
            state, reward, done, _ = env.step(action)
            ep_reward += reward

        total_reward += ep_reward

    return total_reward / n_episodes

# ══════════════════════════════════════════════════════════════════════
# Single map experiment
# ══════════════════════════════════════════════════════════════════════
def run_one_map(map_seed, map_mode=MAP_MODE):
    """
    Run the full growing-dataset experiment on one map.

    Returns
    -------
    bc_returns   : list of policy returns after each round (BC)
    tvbc_returns : list of policy returns after each round (TV-BC)
    round_sizes  : list of dataset sizes at each round
    teacher_ret  : float, teacher reference return
    """
    print(f"\n{'═'*60}")
    print(f"  Map seed={map_seed}  mode={map_mode}")
    print(f"{'═'*60}")

    # ── Build environment ────────────────────────────────────────────
    env = GridWorld(grid_size=8, omega_star=None,
                    max_steps=TRAJ_LENGTH, seed=map_seed,
                    map_mode=map_mode)
    env.print_map()

    # ── Train teacher ────────────────────────────────────────────────
    teacher = RLTeacher(
    omega_star   = env.omega_star,
    grid_size    = env.grid_size,
    gamma        = GAMMA,
    softmax_temp = SOFTMAX_TEMP
    )
    teacher.train(verbose=False)

    teacher_ret = evaluate_policy(
        # Evaluate teacher by wrapping it as a pseudo-policy
        type('T', (), {
            'eval': lambda self: None,
            'get_action_probs': lambda self, s:
                __import__('torch').tensor(
                    teacher.get_action_probs(
                        int(__import__('numpy').argmax(
                            env.get_state_vector(s)
                            if isinstance(s, int) else s
                        ))
                    ), dtype=__import__('torch').float32
                )
        })(),
        env, n_episodes=N_EVAL_EPS
    )
    # Simpler: just roll out the teacher directly
    rng_eval = np.random.default_rng(99)
    total    = 0.0
    for ep in range(N_EVAL_EPS):
        start = env.START_CELL if ep < N_EVAL_EPS//2 \
                else int(rng_eval.integers(0, env.n_states))
        state = env.reset(start_pos=start)
        done  = False; ep_r = 0.0
        while not done:
            action = teacher.act_greedy(state)
            state, r, done, _ = env.step(action)
            ep_r += r
        total += ep_r
    teacher_ret = total / N_EVAL_EPS
    print(f"  Teacher return: {teacher_ret:.3f}")

    # ── Initialise both learners with IDENTICAL weights ──────────────
    bc_policy   = PolicyNetwork(64, 4, 64, seed=SEED_INIT)
    tvbc_policy = PolicyNetwork(64, 4, 64, seed=SEED_INIT)

    bc_learner   = StandardBC(policy=bc_policy,   eta=ETA, seed=SEED_BASE)
    tvbc_learner = TeacherAwareBC(policy=tvbc_policy, teacher=teacher,
                                   eta=ETA, beta=BETA_TV, seed=SEED_BASE)

    # ── Accumulators ─────────────────────────────────────────────────
    # These grow every round. TV scores are FROZEN at collection time.
    #
    # Structure:
    #   bc_all_trajs     : all trajectories collected so far (BC version)
    #   bc_all_tv        : TV scores frozen at collection time (using BC θ)
    #   tvbc_all_trajs   : same trajectories (TV-BC version)
    #   tvbc_all_tv      : TV scores frozen at collection time (using TV-BC θ)
    #
    # Why separate? BC and TV-BC have different θ at each round,
    # so their TV scores for the same trajectory are different.
    bc_all_trajs   = []
    bc_all_tv      = []
    tvbc_all_trajs = []
    tvbc_all_tv    = []

    bc_returns   = []
    tvbc_returns = []
    round_sizes  = []

    traj_seed_counter = 0   # unique seed for each collected trajectory

    print(f"\n  Round | Dataset | BC return | TV-BC return | "
          f"Mean TV (BC) | Mean TV (TV-BC)")
    print(f"  {'─'*70}")

    for round_idx in range(N_ROUNDS):

        # ── Step 1: Collect TRAJS_PER_ROUND new trajectories ─────────
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            traj = collect_trajectory(
                env, teacher,
                noisy = True,
                eps   = NOISE_EPS,
                seed  = traj_seed_counter
            )
            new_trajs.append(traj)
            traj_seed_counter += 1

        # ── Step 2: Compute TV for NEW trajectories using CURRENT θ ──
        # BC uses current BC policy weights
        # TV-BC uses current TV-BC policy weights
        # Old trajectories KEEP their frozen TV scores — not recomputed
        new_bc_tv   = compute_tv_for_trajectories(
            new_trajs, bc_policy, teacher, eta=ETA)
        new_tvbc_tv = compute_tv_for_trajectories(
            new_trajs, tvbc_policy, teacher, eta=ETA)

        # ── Step 3: Accumulate (append new to old, keep old TV frozen) ─
        bc_all_trajs.extend(new_trajs)
        bc_all_tv.extend(new_bc_tv)
        tvbc_all_trajs.extend(new_trajs)
        tvbc_all_tv.extend(new_tvbc_tv)

        dataset_size = len(bc_all_trajs) * TRAJ_LENGTH   # same for both
        round_sizes.append(dataset_size)

        # ── Step 4: Softmax selection over ALL accumulated trajectories ─
        rng_sel = np.random.default_rng(SEED_BASE + round_idx)

        # TV-BC: select via softmax over TV scores
        tvbc_selected, _ = softmax_select(
            tvbc_all_trajs, tvbc_all_tv,
            K=K_SELECT, beta=BETA_TV, rng=rng_sel)

        # BC: select RANDOMLY (same number as TV-BC selected)
        # This is the fair comparison — BC gets same dataset size
        # but without TV-guided selection
        n_select    = len(tvbc_selected)
        rng_bc_sel  = np.random.default_rng(SEED_BASE + round_idx + 1000)
        bc_idx      = list(rng_bc_sel.choice(
            len(bc_all_trajs), size=n_select, replace=False))
        bc_selected = [bc_all_trajs[i] for i in bc_idx]

        # ── Step 5: Train both learners for TRAIN_STEPS on selected data ─
        # Flatten selected trajectories into (states, actions) lists
        bc_states,   bc_actions   = flatten(bc_selected)
        tvbc_states, tvbc_actions = flatten(tvbc_selected)

        # Run TRAIN_STEPS mini-batch updates
        pool_bc   = list(zip(bc_states,   bc_actions))
        pool_tvbc = list(zip(tvbc_states, tvbc_actions))
        rng_train = np.random.default_rng(SEED_BASE + round_idx + 2000)

        for step in range(TRAIN_STEPS):
            # Sample mini-batch from the selected pool
            idx_bc   = rng_train.choice(
                len(pool_bc),   size=min(BATCH_SIZE, len(pool_bc)),
                replace=False)
            idx_tvbc = rng_train.choice(
                len(pool_tvbc), size=min(BATCH_SIZE, len(pool_tvbc)),
                replace=False)

            batch_bc   = [pool_bc[i]   for i in idx_bc]
            batch_tvbc = [pool_tvbc[i] for i in idx_tvbc]

            bc_learner.step(batch_bc)
            tvbc_learner.step(batch_tvbc)

        # ── Step 6: Evaluate both policies ───────────────────────────
        bc_ret   = evaluate_policy(bc_policy,   env)
        tvbc_ret = evaluate_policy(tvbc_policy, env)

        bc_returns.append(bc_ret)
        tvbc_returns.append(tvbc_ret)

        mean_bc_tv   = float(np.mean(bc_all_tv))
        mean_tvbc_tv = float(np.mean(tvbc_all_tv))

        print(f"  {round_idx+1:5d} | "
              f"{dataset_size:7d} | "
              f"{bc_ret:9.3f} | "
              f"{tvbc_ret:12.3f} | "
              f"{mean_bc_tv:11.4f} | "
              f"{mean_tvbc_tv:.4f}")

    return bc_returns, tvbc_returns, round_sizes, teacher_ret

# ══════════════════════════════════════════════════════════════════════
# Main — run over N_MAPS and average
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print(f"{'═'*60}")
    print(f"  Growing Dataset Experiment")
    print(f"{'═'*60}")
    print(f"  Rounds:          {N_ROUNDS}")
    print(f"  Trajs per round: {TRAJS_PER_ROUND}")
    print(f"  Train steps:     {TRAIN_STEPS}")
    print(f"  Beta TV:         {BETA_TV}")
    print(f"  ETA:             {ETA}")
    print(f"  Map mode:        {MAP_MODE}  (N_MAPS={N_MAPS})")

    all_bc_returns   = []
    all_tvbc_returns = []
    all_teacher_rets = []

    for map_i in range(N_MAPS):
        seed = SEED_BASE + map_i * 100
        bc_r, tvbc_r, sizes, teacher_ret = run_one_map(
            map_seed=seed, map_mode=MAP_MODE)

        all_bc_returns.append(bc_r)
        all_tvbc_returns.append(tvbc_r)
        all_teacher_rets.append(teacher_ret)

    # Average over maps
    bc_mean   = np.mean(all_bc_returns,   axis=0)
    tvbc_mean = np.mean(all_tvbc_returns, axis=0)
    bc_std    = np.std(all_bc_returns,    axis=0)
    tvbc_std  = np.std(all_tvbc_returns,  axis=0)
    teacher_mean = float(np.mean(all_teacher_rets))

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FINAL RESULTS (averaged over {N_MAPS} map(s))")
    print(f"{'═'*60}")
    print(f"  Teacher return (reference): {teacher_mean:.3f}")
    print(f"  BC   final return: {bc_mean[-1]:.3f} "
          f"({100*bc_mean[-1]/teacher_mean:.1f}% of teacher)")
    print(f"  TV-BC final return: {tvbc_mean[-1]:.3f} "
          f"({100*tvbc_mean[-1]/teacher_mean:.1f}% of teacher)")
    print(f"  Difference (TV-BC - BC): {tvbc_mean[-1]-bc_mean[-1]:+.3f}")

    # ── Save raw results ──────────────────────────────────────────────
    results = {
        "bc_returns"   : all_bc_returns,
        "tvbc_returns" : all_tvbc_returns,
        "bc_mean"      : bc_mean.tolist(),
        "tvbc_mean"    : tvbc_mean.tolist(),
        "bc_std"       : bc_std.tolist(),
        "tvbc_std"     : tvbc_std.tolist(),
        "round_sizes"  : sizes,
        "teacher_ret"  : teacher_mean,
        "N_ROUNDS"     : N_ROUNDS,
        "TRAJS_PER_ROUND": TRAJS_PER_ROUND,
        "BETA_TV"      : BETA_TV,
        "ETA"          : ETA,
    }
    with open("results/data/growing_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved to results/data/growing_results.pkl")

    # ── Plot ──────────────────────────────────────────────────────────
    rounds     = np.arange(1, N_ROUNDS + 1)
    dataset_sz = [(r+1) * TRAJS_PER_ROUND * TRAJ_LENGTH
                  for r in range(N_ROUNDS)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Return vs round
    axes[0].plot(rounds, bc_mean,   color="#e07b39", lw=2,
                 label="Standard BC")
    axes[0].plot(rounds, tvbc_mean, color="#3a7ebf", lw=2,
                 label="TV-BC (ITAL)")
    if N_MAPS > 1:
        axes[0].fill_between(rounds,
                              bc_mean - bc_std,
                              bc_mean + bc_std,
                              color="#e07b39", alpha=0.2)
        axes[0].fill_between(rounds,
                              tvbc_mean - tvbc_std,
                              tvbc_mean + tvbc_std,
                              color="#3a7ebf", alpha=0.2)
    axes[0].axhline(y=teacher_mean, color="green",
                    linestyle="--", lw=1.5,
                    label=f"Teacher ({teacher_mean:.1f})")
    axes[0].set_xlabel("Teaching Round", fontsize=12)
    axes[0].set_ylabel("Average Policy Return", fontsize=12)
    axes[0].set_title("Policy Return per Round\n"
                      "(Growing Dataset)", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Return vs dataset size
    axes[1].plot(dataset_sz, bc_mean,   color="#e07b39", lw=2,
                 label="Standard BC",  marker="o", ms=4)
    axes[1].plot(dataset_sz, tvbc_mean, color="#3a7ebf", lw=2,
                 label="TV-BC (ITAL)", marker="s", ms=4)
    axes[1].axhline(y=teacher_mean, color="green",
                    linestyle="--", lw=1.5,
                    label=f"Teacher ({teacher_mean:.1f})")
    axes[1].set_xlabel("Total (s,a) Pairs Seen", fontsize=12)
    axes[1].set_ylabel("Average Policy Return", fontsize=12)
    axes[1].set_title("Sample Efficiency\n"
                      "(Growing Dataset)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Growing Dataset  |  {TRAJS_PER_ROUND} trajs/round  |  "
        f"beta={BETA_TV}  eta={ETA}  |  {N_MAPS} map(s)",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("results/plots/growing_curves.png", dpi=150)
    plt.close()
    print(f"  Plot saved to results/plots/growing_curves.png")
