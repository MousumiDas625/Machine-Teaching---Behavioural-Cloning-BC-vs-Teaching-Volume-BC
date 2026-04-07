# experiments/run_growing_mapobs_recency.py
#
# Growing dataset with map observation + RECENCY WEIGHTING.
#
# Problem this fixes:
#   In run_growing_mapobs.py, frozen TV scores from old maps
#   stay high because they were informative THEN.
#   TV-BC keeps selecting old-map trajectories because their
#   frozen TV scores are high, reinforcing wrong behaviour.
#
# Solution — recency weighting:
#   When selecting trajectories for training, multiply each
#   trajectory's TV score by a recency weight based on which
#   map it came from:
#
#     Current map:     weight = 1.0   (full weight)
#     1 map ago:       weight = 0.5   (half weight)
#     2 maps ago:      weight = 0.25
#     3 maps ago:      weight = 0.125
#     4+ maps ago:     weight = 0.0625
#
#   Effective TV = frozen_TV * recency_weight
#
# This preserves the professor's rule (never discard data)
# while reducing the influence of stale old-map trajectories.
# Old data is kept and can still be selected — it just has
# lower effective TV score than recent data.
#
# BC comparison:
#   BC uses the same recency weighting for its RANDOM selection
#   probability — so both learners have the same bias toward
#   recent data. The only difference remains the TV scoring.

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.rl_teacher              import RLTeacher
from agents.policy_net_mapobs       import PolicyNetworkMapObs
from gridworld_env.gridworld_mapobs import GridWorldMapObs
from learners.standard_bc           import StandardBC
from learners.tv_bc                 import TeacherAwareBC

# ══════════════════════════════════════════════════════════════════════
# Configurable parameters
# ══════════════════════════════════════════════════════════════════════
N_ROUNDS          = 20
TRAJS_PER_ROUND   = 5
TRAJ_LENGTH       = 50
NOISE_EPS         = 0.30
TRAIN_STEPS       = 200      # increased from 200 — more steps per map
BATCH_SIZE        = 20
K_SELECT          = 0        # 0 = use all (weighted)
BETA_TV           = 2.0      # reduced from 2.0 — more stable
ETA               = 0.01
GAMMA             = 0.99
SOFTMAX_TEMP      = 0.1
N_EVAL_EPS        = 50
TERMINAL          = False
STEP_PENALTY      = 0.0
MAP_CHANGE_EVERY  = 4
N_GOALS           = 3
N_TRAPS           = 16
SEED_BASE         = 42
SEED_INIT         = 0

# Recency decay factor
# Weight for trajectory from k maps ago = RECENCY_DECAY^k
# 0.5 means: current=1.0, 1 ago=0.5, 2 ago=0.25, 3 ago=0.125
RECENCY_DECAY     = 0.5

OBS_DIM    = 128
ACTION_DIM = 4

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Helper: collect one trajectory
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, noisy=True,
                        eps=NOISE_EPS, seed=0):
    rng_noise = np.random.default_rng(seed)
    state     = env.reset(start_pos=env.START_CELL)
    traj      = []
    done      = False
    while not done:
        obs    = env.get_full_obs(state)
        if noisy and rng_noise.random() < eps:
            action = int(rng_noise.integers(0, env.n_actions))
        else:
            action = teacher.act_greedy(state)
        traj.append((obs, action))
        state, _, done, _ = env.step(action)
    return traj

# ══════════════════════════════════════════════════════════════════════
# Helper: compute TV scores
# ══════════════════════════════════════════════════════════════════════
def compute_tv_for_trajectories(trajs, policy, teacher, eta=ETA):
    tv_scores = []
    for traj in trajs:
        if len(traj) == 0:
            tv_scores.append(0.0)
            continue
        step_tvs = []
        for (obs_128, action) in traj:
            pos_vec   = obs_128[64:]
            state_idx = int(np.argmax(pos_vec))

            policy.zero_grad()
            loss_t = policy.compute_loss(obs_128, action)
            loss_t.backward()

            grads = []
            for p in policy.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().view(-1))
                else:
                    grads.append(torch.zeros(p.numel()))
            flat_grad    = torch.cat(grads)
            grad_norm_sq = float(flat_grad.dot(flat_grad).item())
            term1        = -(eta ** 2) * grad_norm_sq

            learner_loss = loss_t.item()
            teacher_loss = teacher.loss_at(state_idx, action)
            term2        = 2.0 * eta * (learner_loss - teacher_loss)
            step_tvs.append(term1 + term2)

        tv_scores.append(float(np.mean(step_tvs)))
    return tv_scores

# ══════════════════════════════════════════════════════════════════════
# Helper: recency-weighted softmax selection
# ══════════════════════════════════════════════════════════════════════
def recency_weighted_select(all_trajs, all_tv_scores,
                             all_map_ids, current_map_id,
                             K, beta, rng,
                             decay=RECENCY_DECAY):
    """
    Select K trajectories using TV scores weighted by recency.

    For each trajectory:
      maps_ago = current_map_id - trajectory_map_id
      recency_weight = decay ^ maps_ago
      effective_tv   = frozen_tv * recency_weight

    Then apply softmax over effective_tv scores.

    Parameters
    ----------
    all_trajs      : list of all accumulated trajectories
    all_tv_scores  : list of frozen TV scores
    all_map_ids    : list of map_id when each traj was collected
    current_map_id : int — which map is active right now
    K              : number to select (0 = use all)
    beta           : softmax temperature
    rng            : numpy random generator
    decay          : recency decay factor (0.5 default)

    Returns
    -------
    selected_trajs  : list of selected trajectories
    selected_idx    : indices of selected trajectories
    effective_tvs   : the weighted TV scores used for selection
    """
    N = len(all_trajs)

    # Compute recency weights for each trajectory
    recency_weights = np.array([
        decay ** (current_map_id - m_id)
        for m_id in all_map_ids
    ], dtype=np.float64)

    # Effective TV = frozen TV score * recency weight
    tv_arr    = np.array(all_tv_scores, dtype=np.float64)
    eff_tv    = tv_arr * recency_weights

    if K == 0 or K >= N:
        # Use all — but still weight the selection probability
        # for BC comparison (BC uses recency weights too)
        return all_trajs, list(range(N)), eff_tv

    # Softmax over effective TV scores
    eff_tv_stable  = eff_tv - eff_tv.max()
    weights        = np.exp(beta * eff_tv_stable)
    probs          = weights / weights.sum()
    idx            = list(rng.choice(N, size=K,
                                      replace=False, p=probs))
    return [all_trajs[i] for i in idx], idx, eff_tv

# ══════════════════════════════════════════════════════════════════════
# Helper: recency-weighted random selection for BC
# ══════════════════════════════════════════════════════════════════════
def recency_weighted_bc_select(all_trajs, all_map_ids,
                                current_map_id, n_select,
                                rng, decay=RECENCY_DECAY):
    """
    BC random selection weighted by recency.

    BC does not use TV scores but still uses recency weights
    so that both learners have the same bias toward recent data.
    The only difference between BC and TV-BC remains whether
    TV scores are used for selection.

    BC selection probability:
      p(traj_i) proportional to decay^(maps_ago)
    """
    N = len(all_trajs)
    recency_weights = np.array([
        decay ** (current_map_id - m_id)
        for m_id in all_map_ids
    ], dtype=np.float64)

    probs = recency_weights / recency_weights.sum()
    idx   = list(rng.choice(N,
                             size=min(n_select, N),
                             replace=False,
                             p=probs))
    return [all_trajs[i] for i in idx], idx

# ══════════════════════════════════════════════════════════════════════
# Helper: flatten trajectories
# ══════════════════════════════════════════════════════════════════════
def flatten(trajs):
    obs_list, act_list = [], []
    for traj in trajs:
        for (obs, a) in traj:
            obs_list.append(obs)
            act_list.append(a)
    return obs_list, act_list

# ══════════════════════════════════════════════════════════════════════
# Helper: evaluate policy
# ══════════════════════════════════════════════════════════════════════
def evaluate_policy(policy, env,
                    n_episodes=N_EVAL_EPS, seed=99):
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    goals_reached = 0
    policy.eval()

    for ep in range(n_episodes):
        start = env.START_CELL if ep < n_episodes // 2 \
                else int(rng.integers(0, env.n_states))
        state           = env.reset(start_pos=start)
        done            = False
        ep_reward       = 0.0
        reached_this_ep = False

        while not done:
            obs    = env.get_full_obs(state)
            probs  = policy.get_action_probs(obs).numpy()
            action = int(np.argmax(probs))
            state, reward, done, info = env.step(action)
            ep_reward += reward
            if info.get('reached_goal', False):
                reached_this_ep = True

        total_reward  += ep_reward
        goals_reached += 1 if reached_this_ep else 0

    return total_reward / n_episodes, goals_reached / n_episodes

# ══════════════════════════════════════════════════════════════════════
# Helper: evaluate teacher
# ══════════════════════════════════════════════════════════════════════
def eval_teacher(teacher, env, n_episodes=30):
    rng   = np.random.default_rng(99)
    total = 0.0
    for ep in range(n_episodes):
        start = env.START_CELL if ep < n_episodes // 2 \
                else int(rng.integers(0, env.n_states))
        state = env.reset(start_pos=start)
        done  = False; ep_r = 0.0
        while not done:
            action = teacher.act_greedy(state)
            state, r, done, _ = env.step(action)
            ep_r += r
        total += ep_r
    return total / n_episodes

# ══════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════
def run_experiment():

    print(f"{'═'*65}")
    print(f"  Growing Dataset — Map Obs — Recency Weighting")
    print(f"{'═'*65}")
    print(f"  Rounds:          {N_ROUNDS}")
    print(f"  Trajs/round:     {TRAJS_PER_ROUND}")
    print(f"  Map changes:     every {MAP_CHANGE_EVERY} rounds")
    print(f"  Train steps:     {TRAIN_STEPS} (increased)")
    print(f"  Beta TV:         {BETA_TV} (reduced for stability)")
    print(f"  Recency decay:   {RECENCY_DECAY}")
    print(f"    current map:   weight=1.0")
    print(f"    1 map ago:     weight={RECENCY_DECAY:.2f}")
    print(f"    2 maps ago:    weight={RECENCY_DECAY**2:.3f}")
    print(f"    3 maps ago:    weight={RECENCY_DECAY**3:.4f}")
    print(f"    4 maps ago:    weight={RECENCY_DECAY**4:.4f}")
    print(f"  ETA:             {ETA}")
    print(f"  Obs dim:         {OBS_DIM}")

    # ── Build initial environment ──────────────────────────────────────
    env = GridWorldMapObs(
        grid_size    = 8,
        max_steps    = TRAJ_LENGTH,
        seed         = SEED_BASE,
        n_goals      = N_GOALS,
        n_traps      = N_TRAPS,
        terminal     = TERMINAL,
        step_penalty = STEP_PENALTY
    )
    env.print_map()

    # ── Train initial teacher ──────────────────────────────────────────
    teacher = RLTeacher(
        omega_star   = env.omega_star,
        grid_size    = env.grid_size,
        gamma        = GAMMA,
        softmax_temp = SOFTMAX_TEMP
    )
    teacher.train(verbose=True)
    t_ret = eval_teacher(teacher, env)
    print(f"  Teacher return (map 1): {t_ret:.3f}\n")

    # ── Initialise both learners with IDENTICAL weights ────────────────
    bc_policy   = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
    tvbc_policy = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)

    bc_learner   = StandardBC(
        policy=bc_policy,   eta=ETA, seed=SEED_BASE)
    tvbc_learner = TeacherAwareBC(
        policy=tvbc_policy, teacher=teacher,
        eta=ETA, beta=BETA_TV, seed=SEED_BASE)

    print(f"  Parameters: {bc_policy.count_parameters():,}\n")

    # ── Accumulators ───────────────────────────────────────────────────
    # Each trajectory stores: (traj_data, frozen_tv, map_id_when_collected)
    bc_all_trajs   = []
    bc_all_tv      = []
    bc_all_map_ids = []   # which map was active when collected

    tvbc_all_trajs   = []
    tvbc_all_tv      = []
    tvbc_all_map_ids = []

    bc_returns      = []
    tvbc_returns    = []
    bc_goal_rates   = []
    tvbc_goal_rates = []
    teacher_rets    = []
    round_sizes     = []
    eff_tv_means    = []   # mean effective TV each round

    traj_seed_ctr  = 0
    current_map_id = 1

    print(f"  {'Rnd':>3} | {'Map':>3} | {'Dataset':>7} | "
          f"{'BC ret':>8} | {'BC%':>6} | "
          f"{'TVBC ret':>8} | {'TVBC%':>6} | "
          f"{'Teacher':>7} | {'EffTV':>7}")
    print(f"  {'─'*80}")

    for round_idx in range(N_ROUNDS):

        # ── Map change check ───────────────────────────────────────────
        if round_idx > 0 and round_idx % MAP_CHANGE_EVERY == 0:
            current_map_id += 1
            new_seed = SEED_BASE + (current_map_id - 1) * 100

            print(f"\n  ── MAP CHANGE → Map {current_map_id} "
                  f"(seed={new_seed}) ──")
            env.change_map(new_seed)
            env.print_map()

            teacher = RLTeacher(
                omega_star   = env.omega_star,
                grid_size    = env.grid_size,
                gamma        = GAMMA,
                softmax_temp = SOFTMAX_TEMP
            )
            teacher.train(verbose=True)
            tvbc_learner.teacher = teacher

            t_ret = eval_teacher(teacher, env)
            print(f"  Teacher return (map {current_map_id}): "
                  f"{t_ret:.3f}\n")

        # ── Step 1: Collect new trajectories ──────────────────────────
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            traj = collect_trajectory(
                env, teacher, noisy=True,
                eps=NOISE_EPS, seed=traj_seed_ctr)
            new_trajs.append(traj)
            traj_seed_ctr += 1

        # ── Step 2: Compute TV for new trajs (current theta) ──────────
        new_bc_tv   = compute_tv_for_trajectories(
            new_trajs, bc_policy,   teacher)
        new_tvbc_tv = compute_tv_for_trajectories(
            new_trajs, tvbc_policy, teacher)

        # ── Step 3: Accumulate with map_id tag ────────────────────────
        bc_all_trajs.extend(new_trajs)
        bc_all_tv.extend(new_bc_tv)
        bc_all_map_ids.extend([current_map_id] * len(new_trajs))

        tvbc_all_trajs.extend(new_trajs)
        tvbc_all_tv.extend(new_tvbc_tv)
        tvbc_all_map_ids.extend([current_map_id] * len(new_trajs))

        dataset_size = sum(len(t) for t in bc_all_trajs)
        round_sizes.append(dataset_size)

        # ── Step 4: Recency-weighted selection ────────────────────────
        rng_sel = np.random.default_rng(SEED_BASE + round_idx)

        # TV-BC: softmax over recency-weighted TV scores
        tvbc_selected, _, eff_tv = recency_weighted_select(
            tvbc_all_trajs, tvbc_all_tv, tvbc_all_map_ids,
            current_map_id, K=K_SELECT,
            beta=BETA_TV, rng=rng_sel)

        # BC: recency-weighted random selection (same size)
        n_select = len(tvbc_selected)
        rng_bc   = np.random.default_rng(
            SEED_BASE + round_idx + 1000)
        bc_selected, _ = recency_weighted_bc_select(
            bc_all_trajs, bc_all_map_ids,
            current_map_id, n_select, rng_bc)

        eff_tv_means.append(float(np.mean(eff_tv)))

        # ── Step 5: Train both learners ────────────────────────────────
        bc_obs,   bc_acts   = flatten(bc_selected)
        tvbc_obs, tvbc_acts = flatten(tvbc_selected)

        bc_pool   = list(zip(bc_obs,   bc_acts))
        tvbc_pool = list(zip(tvbc_obs, tvbc_acts))

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
            bc_learner.step([bc_pool[i]   for i in idx_bc])
            tvbc_learner.step([tvbc_pool[i] for i in idx_tvbc])

        # ── Step 6: Evaluate on CURRENT map ────────────────────────────
        bc_ret,   bc_gr   = evaluate_policy(bc_policy,   env)
        tvbc_ret, tvbc_gr = evaluate_policy(tvbc_policy, env)
        t_ret             = eval_teacher(teacher, env)

        bc_returns.append(bc_ret)
        tvbc_returns.append(tvbc_ret)
        bc_goal_rates.append(bc_gr)
        tvbc_goal_rates.append(tvbc_gr)
        teacher_rets.append(t_ret)

        print(f"  {round_idx+1:>3} | "
              f"{current_map_id:>3} | "
              f"{dataset_size:>7} | "
              f"{bc_ret:>8.3f} | "
              f"{bc_gr*100:>5.1f}% | "
              f"{tvbc_ret:>8.3f} | "
              f"{tvbc_gr*100:>5.1f}% | "
              f"{t_ret:>7.3f} | "
              f"{eff_tv_means[-1]:>7.4f}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  FINAL RESULTS (Recency Weighted)")
    print(f"{'═'*65}")
    print(f"  BC   final: {bc_returns[-1]:.3f}  "
          f"goal={bc_goal_rates[-1]*100:.1f}%")
    print(f"  TV-BC final: {tvbc_returns[-1]:.3f}  "
          f"goal={tvbc_goal_rates[-1]*100:.1f}%")
    print(f"  Diff (TV-BC - BC): "
          f"{tvbc_returns[-1]-bc_returns[-1]:+.3f}")
    print(f"  Teacher final: {teacher_rets[-1]:.3f}")

    # ── Compare with previous run ──────────────────────────────────────
    print(f"\n  Comparison with no recency weighting:")
    print(f"  {'':20s}  {'BC':>8}  {'TV-BC':>8}")
    print(f"  {'No recency (prev)':20s}  {'34.815':>8}  {'29.755':>8}")
    print(f"  {'With recency':20s}  "
          f"{bc_returns[-1]:>8.3f}  {tvbc_returns[-1]:>8.3f}")

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "bc_returns"     : bc_returns,
        "tvbc_returns"   : tvbc_returns,
        "bc_goal_rates"  : bc_goal_rates,
        "tvbc_goal_rates": tvbc_goal_rates,
        "teacher_rets"   : teacher_rets,
        "round_sizes"    : round_sizes,
        "eff_tv_means"   : eff_tv_means,
        "N_ROUNDS"       : N_ROUNDS,
        "MAP_CHANGE_EVERY": MAP_CHANGE_EVERY,
        "RECENCY_DECAY"  : RECENCY_DECAY,
        "BETA_TV"        : BETA_TV,
        "TRAIN_STEPS"    : TRAIN_STEPS,
    }
    with open("results/data/growing_mapobs_recency_results.pkl",
              "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/"
          f"growing_mapobs_recency_results.pkl")

    # ── Plot ───────────────────────────────────────────────────────────
    rounds     = np.arange(1, N_ROUNDS + 1)
    map_colors = ['#e8f4f8','#fef9e7','#e8f8e8',
                  '#fdf2f8','#f2f3f4']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for map_i in range(5):
        s = map_i * MAP_CHANGE_EVERY + 1
        e = min((map_i+1) * MAP_CHANGE_EVERY, N_ROUNDS)
        for ax in axes:
            ax.axvspan(s-0.5, e+0.5,
                       alpha=0.4,
                       color=map_colors[map_i],
                       label=f'Map {map_i+1}')

    axes[0].plot(rounds, bc_returns,
                 color="#e07b39", lw=2,
                 label="BC", marker="o", ms=4)
    axes[0].plot(rounds, tvbc_returns,
                 color="#3a7ebf", lw=2,
                 label="TV-BC", marker="s", ms=4)
    axes[0].plot(rounds, teacher_rets,
                 color="green", lw=1.5,
                 linestyle="--", label="Teacher")
    for cr in range(MAP_CHANGE_EVERY, N_ROUNDS, MAP_CHANGE_EVERY):
        axes[0].axvline(x=cr+0.5, color='red',
                        linestyle=':', lw=1.5, alpha=0.7)
    axes[0].set_xlabel("Teaching Round", fontsize=12)
    axes[0].set_ylabel("Average Return",  fontsize=12)
    axes[0].set_title("Return — Recency Weighted\n"
                      "(red = map change)", fontsize=11)
    axes[0].legend(fontsize=9, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, [r*100 for r in bc_goal_rates],
                 color="#e07b39", lw=2,
                 label="BC",    marker="o", ms=4)
    axes[1].plot(rounds, [r*100 for r in tvbc_goal_rates],
                 color="#3a7ebf", lw=2,
                 label="TV-BC", marker="s", ms=4)
    for cr in range(MAP_CHANGE_EVERY, N_ROUNDS, MAP_CHANGE_EVERY):
        axes[1].axvline(x=cr+0.5, color='red',
                        linestyle=':', lw=1.5, alpha=0.7)
    axes[1].set_xlabel("Teaching Round",   fontsize=12)
    axes[1].set_ylabel("Goal Rate (%)",    fontsize=12)
    axes[1].set_title("Goal Rate — Recency Weighted",
                      fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Recency Weighted  |  decay={RECENCY_DECAY}  "
        f"beta={BETA_TV}  steps={TRAIN_STEPS}  "
        f"obs={OBS_DIM}d",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(
        "results/plots/growing_mapobs_recency_curves.png",
        dpi=150)
    plt.close()
    print(f"  Plot: results/plots/"
          f"growing_mapobs_recency_curves.png")


if __name__ == "__main__":
    run_experiment()
