# experiments/run_generalisation.py
#
# Generalisation experiment — faithful to Algorithm 1 (Yuan et al., NeurIPS 2021)
#
# Three design principles:
#   1. EQUAL WEIGHTAGE TO ALL HISTORY:
#      TV-BC uses softmax over frozen TV scores only — no recency decay.
#      All accumulated demonstrations compete equally based on TV score.
#      High TV = selected more. Low TV = selected less. No age penalty.
#      BC selects uniformly at random from all accumulated history.
#
#   2. NEW MAP EVERY ROUND:
#      Each of the 50 training rounds uses a completely new random map.
#      Robot never trains on the same map twice.
#
#   3. TEST ON 20 UNSEEN MAPS:
#      20 held-out maps fixed before training starts.
#      Robot never sees demonstrations from these maps.
#      Evaluated after every training round.

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
# Parameters
# ══════════════════════════════════════════════════════════════════════
N_ROUNDS        = 50       # training rounds = unique training maps
TRAJS_PER_ROUND = 5        # demonstrations per round
TRAJ_LENGTH     = 50       # max steps per trajectory
NOISE_EPS       = 0.30     # teacher noise
TRAIN_STEPS     = 500      # SGD steps per round
BATCH_SIZE      = 20
BETA_TV         = 2.0
ETA             = 0.01
GAMMA           = 0.99
SOFTMAX_TEMP    = 0.1
N_EVAL_EPS      = 50
N_GOALS         = 3
N_TRAPS         = 16
SEED_INIT       = 0

# Training map seeds — one per round, never reused
TRAIN_SEEDS = [1000 + i * 13 for i in range(N_ROUNDS)]

# Test map seeds — completely separate, NEVER used for training
TEST_SEEDS = {f'test_{i}': 5000 + i for i in range(20)}

OBS_DIM    = 128
ACTION_DIM = 4

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# Verify no overlap
assert all(s not in TRAIN_SEEDS for s in TEST_SEEDS.values()), \
    "ERROR: test seed overlaps with training seed"

# ══════════════════════════════════════════════════════════════════════
# Environment and teacher builders
# ══════════════════════════════════════════════════════════════════════
def make_env(seed):
    return GridWorldMapObs(
        grid_size=8, max_steps=TRAJ_LENGTH, seed=seed,
        n_goals=N_GOALS, n_traps=N_TRAPS,
        terminal=False, step_penalty=0.0)

def make_teacher(env):
    t = RLTeacher(omega_star=env.omega_star,
                  grid_size=env.grid_size,
                  gamma=GAMMA, softmax_temp=SOFTMAX_TEMP)
    t.train(verbose=False)
    return t

# ══════════════════════════════════════════════════════════════════════
# Trajectory collection
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, eps=NOISE_EPS, seed=0):
    """
    Collect one demonstration trajectory using 128-dim map observation.
    obs = [omega_star_THIS_MAP | one_hot_position]
    """
    rng   = np.random.default_rng(seed)
    state = env.reset(start_pos=env.START_CELL)
    traj  = []
    done  = False
    while not done:
        obs = env.get_full_obs(state)
        if rng.random() < eps:
            action = int(rng.integers(0, env.n_actions))
        else:
            action = teacher.act_greedy(state)
        traj.append((obs, action))
        state, _, done, _ = env.step(action)
    return traj

# ══════════════════════════════════════════════════════════════════════
# TV computation
# ══════════════════════════════════════════════════════════════════════
def compute_tv_batch(trajs, policy, teacher, eta=ETA):
    """
    Compute TV score per trajectory = mean TV across all steps.
    Uses CURRENT policy theta. Frozen immediately after.
    state_idx extracted from obs[64:] (position part) not full obs.
    """
    scores = []
    for traj in trajs:
        if len(traj) == 0:
            scores.append(0.0)
            continue
        step_tvs = []
        for (obs, action) in traj:
            state_idx = int(np.argmax(obs[64:]))
            policy.zero_grad()
            loss = policy.compute_loss(obs, action)
            loss.backward()
            grads = [p.grad.detach().view(-1) if p.grad is not None
                     else torch.zeros(p.numel())
                     for p in policy.parameters()]
            flat      = torch.cat(grads)
            gnorm_sq  = float(flat.dot(flat).item())
            term1     = -(eta ** 2) * gnorm_sq
            learner_l = loss.item()
            teacher_l = teacher.loss_at(state_idx, action)
            term2     = 2.0 * eta * (learner_l - teacher_l)
            step_tvs.append(term1 + term2)
        scores.append(float(np.mean(step_tvs)))
    return scores

# ══════════════════════════════════════════════════════════════════════
# Selection functions — equal weightage to all history
# ══════════════════════════════════════════════════════════════════════
def select_tvbc(all_trajs, all_tv, beta, rng):
    """
    TV-BC selection: softmax over frozen TV scores.
    EQUAL WEIGHTAGE to all history — no recency decay.
    All accumulated demonstrations compete based on TV score only.
    High TV = selected more often.
    Low TV (trap-entering) = almost never selected.
    This follows Algorithm 1 from Yuan et al. NeurIPS 2021.
    """
    N = len(all_trajs)
    if N == 0:
        return [], [], np.array([])

    tv_arr  = np.array(all_tv, dtype=np.float64)
    tv_s    = tv_arr - tv_arr.max()          # numerical stability
    probs   = np.exp(beta * tv_s)
    probs  /= probs.sum()
    idx     = list(rng.choice(N, size=N, replace=False, p=probs))
    return [all_trajs[i] for i in idx], idx, tv_arr


def select_bc(all_trajs, n_select, rng):
    """
    BC selection: uniform random from all accumulated history.
    EQUAL WEIGHTAGE — no recency, no TV.
    Matches uniform D_t sampling for the naive learner in the paper.
    """
    N   = len(all_trajs)
    idx = list(rng.choice(N, size=min(n_select, N), replace=False))
    return [all_trajs[i] for i in idx], idx

# ══════════════════════════════════════════════════════════════════════
# Flatten trajectories
# ══════════════════════════════════════════════════════════════════════
def flatten(trajs):
    obs_l, act_l = [], []
    for traj in trajs:
        for (o, a) in traj:
            obs_l.append(o)
            act_l.append(a)
    return obs_l, act_l

# ══════════════════════════════════════════════════════════════════════
# Policy evaluation
# ══════════════════════════════════════════════════════════════════════
def evaluate(policy, env, n_eps=N_EVAL_EPS, seed=99):
    """
    Evaluate policy on given environment.
    Works on both training maps and unseen test maps.
    Returns avg_return, goal_rate.
    """
    rng     = np.random.default_rng(seed)
    total   = 0.0
    n_goals = 0
    policy.eval()
    for ep in range(n_eps):
        start = env.START_CELL if ep < n_eps // 2 \
                else int(rng.integers(0, env.n_states))
        state   = env.reset(start_pos=start)
        done    = False
        ep_r    = 0.0
        reached = False
        while not done:
            obs    = env.get_full_obs(state)
            probs  = policy.get_action_probs(obs).numpy()
            action = int(np.argmax(probs))
            state, r, done, info = env.step(action)
            ep_r += r
            if info.get('reached_goal', False):
                reached = True
        total   += ep_r
        n_goals += 1 if reached else 0
    return total / n_eps, n_goals / n_eps


def eval_teacher_return(teacher, env, n_eps=30):
    rng   = np.random.default_rng(99)
    total = 0.0
    for ep in range(n_eps):
        start = env.START_CELL if ep < n_eps // 2 \
                else int(rng.integers(0, env.n_states))
        state = env.reset(start_pos=start)
        done  = False; ep_r = 0.0
        while not done:
            state, r, done, _ = env.step(teacher.act_greedy(state))
            ep_r += r
        total += ep_r
    return total / n_eps

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def run():
    print(f"{'═'*70}")
    print(f"  Generalisation Experiment")
    print(f"  Algorithm 1 (Yuan et al. NeurIPS 2021) — faithful implementation")
    print(f"{'═'*70}")
    print(f"  Training maps:    {N_ROUNDS} unique maps, one per round")
    print(f"  Test maps:        {len(TEST_SEEDS)} held-out maps (seeds 5000-5019)")
    print(f"  Selection:        equal weightage — TV drives TV-BC, uniform for BC")
    print(f"  N_TRAPS:          {N_TRAPS}")
    print(f"  Noise:            {NOISE_EPS}")
    print(f"  Beta:             {BETA_TV}")
    print(f"  Train steps/rnd:  {TRAIN_STEPS}")
    print(f"  Obs dim:          {OBS_DIM} (map 64 + position 64)")

    # ── Build 20 held-out test environments ───────────────────────────
    print(f"\n{'─'*70}")
    print(f"  HELD-OUT TEST MAPS (robot will NEVER train on these)")
    print(f"{'─'*70}")
    test_envs     = {}
    test_teachers = {}
    test_refs     = {}
    for name, seed in TEST_SEEDS.items():
        env = make_env(seed)
        tch = make_teacher(env)
        ref = eval_teacher_return(tch, env)
        test_envs[name]     = env
        test_teachers[name] = tch
        test_refs[name]     = ref

    avg_ref = float(np.mean(list(test_refs.values())))
    print(f"  Teacher return range: "
          f"[{min(test_refs.values()):.2f}, {max(test_refs.values()):.2f}]")
    print(f"  Teacher return avg:   {avg_ref:.3f}")

    # ── Initialise both learners with IDENTICAL weights ────────────────
    bc_policy   = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
    tvbc_policy = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)

    bc_learner   = StandardBC(policy=bc_policy, eta=ETA, seed=42)
    tvbc_learner = TeacherAwareBC(
        policy=tvbc_policy, teacher=None,
        eta=ETA, beta=BETA_TV, seed=42)

    print(f"\n  Network parameters: {bc_policy.count_parameters():,}")
    print(f"  Both networks: same seed={SEED_INIT} (identical start)\n")

    # ── Accumulators ───────────────────────────────────────────────────
    # TV scores frozen at collection time — equal weight in selection
    bc_pool_trajs  = [];  bc_pool_tv  = []
    tvbc_pool_trajs = []; tvbc_pool_tv = []

    # Per-round results averaged over 20 test maps
    bc_avg_ret    = []
    tvbc_avg_ret  = []
    bc_avg_goal   = []
    tvbc_avg_goal = []
    pool_sizes    = []
    mean_tv_log   = []
    traj_seed     = 0

    names = list(TEST_SEEDS.keys())
    print(f"  {'Rnd':>3} | {'Pool':>6} | "
          f"{'BC avg':>8} | {'TV-BC avg':>9} | "
          f"{'BC goal':>7} | {'TV goal':>7} | "
          f"{'MeanTV':>8}")
    print(f"  {'─'*70}")

    # ══════════════════════════════════════════════════════════════════
    # Training loop — one new map per round
    # ══════════════════════════════════════════════════════════════════
    for rnd in range(N_ROUNDS):

        # ── NEW training map this round ────────────────────────────────
        train_seed = TRAIN_SEEDS[rnd]
        train_env  = make_env(train_seed)
        train_tch  = make_teacher(train_env)
        tvbc_learner.teacher = train_tch

        # ── Collect 5 demonstrations ───────────────────────────────────
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            t = collect_trajectory(train_env, train_tch,
                                    eps=NOISE_EPS, seed=traj_seed)
            new_trajs.append(t)
            traj_seed += 1

        # ── Compute TV using CURRENT theta — freeze immediately ────────
        bc_new_tv   = compute_tv_batch(new_trajs, bc_policy,   train_tch)
        tvbc_new_tv = compute_tv_batch(new_trajs, tvbc_policy, train_tch)

        # ── Add to pool (equal weight — no round tag needed) ──────────
        bc_pool_trajs.extend(new_trajs);   bc_pool_tv.extend(bc_new_tv)
        tvbc_pool_trajs.extend(new_trajs); tvbc_pool_tv.extend(tvbc_new_tv)

        pool_size = sum(len(t) for t in bc_pool_trajs)
        pool_sizes.append(pool_size)

        # ── Select — equal weightage to all history ────────────────────
        rng_s = np.random.default_rng(42 + rnd)

        # TV-BC: softmax over ALL frozen TV scores (no recency)
        tvbc_sel, _, tv_arr = select_tvbc(
            tvbc_pool_trajs, tvbc_pool_tv,
            beta=BETA_TV, rng=rng_s)

        # BC: uniform random from ALL history (no recency)
        bc_sel, _ = select_bc(
            bc_pool_trajs,
            n_select=len(tvbc_sel),
            rng=np.random.default_rng(42 + rnd + 1000))

        mean_tv_log.append(float(np.mean(tv_arr)))

        # ── Train 500 steps ────────────────────────────────────────────
        bc_obs,   bc_acts   = flatten(bc_sel)
        tvbc_obs, tvbc_acts = flatten(tvbc_sel)
        bc_flat   = list(zip(bc_obs,   bc_acts))
        tvbc_flat = list(zip(tvbc_obs, tvbc_acts))

        rng_t = np.random.default_rng(42 + rnd + 2000)
        for _ in range(TRAIN_STEPS):
            bs = min(BATCH_SIZE, len(bc_flat))
            ts = min(BATCH_SIZE, len(tvbc_flat))
            if bs == 0 or ts == 0:
                break
            bi = rng_t.choice(len(bc_flat),   size=bs, replace=False)
            ti = rng_t.choice(len(tvbc_flat), size=ts, replace=False)
            bc_learner.step(  [bc_flat[i]   for i in bi])
            tvbc_learner.step([tvbc_flat[i] for i in ti])

        # ── Evaluate on ALL 20 held-out test maps ──────────────────────
        bc_rets   = []
        tvbc_rets = []
        bc_goals  = []
        tvbc_goals = []

        for name in names:
            bc_r,   bc_g   = evaluate(bc_policy,   test_envs[name])
            tvbc_r, tvbc_g = evaluate(tvbc_policy, test_envs[name])
            bc_rets.append(bc_r);    bc_goals.append(bc_g)
            tvbc_rets.append(tvbc_r); tvbc_goals.append(tvbc_g)

        bc_avg   = float(np.mean(bc_rets))
        tvbc_avg = float(np.mean(tvbc_rets))
        bc_g_avg   = float(np.mean(bc_goals))
        tvbc_g_avg = float(np.mean(tvbc_goals))

        bc_avg_ret.append(bc_avg)
        tvbc_avg_ret.append(tvbc_avg)
        bc_avg_goal.append(bc_g_avg)
        tvbc_avg_goal.append(tvbc_g_avg)

        print(f"  {rnd+1:>3} | {pool_size:>6} | "
              f"{bc_avg:>8.3f} | {tvbc_avg:>9.3f} | "
              f"{bc_g_avg*100:>6.1f}% | {tvbc_g_avg*100:>6.1f}% | "
              f"{mean_tv_log[-1]:>8.4f}")

    # ══════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL RESULTS — averaged over 20 held-out test maps")
    print(f"  Robot trained on {N_ROUNDS} maps, tested on 20 unseen maps")
    print(f"{'═'*70}")
    print(f"  Teacher avg return:    {avg_ref:.3f}")
    print(f"  BC   avg return:       {bc_avg_ret[-1]:.3f}  "
          f"({100*bc_avg_ret[-1]/avg_ref:.1f}% of teacher)  "
          f"goal={bc_avg_goal[-1]*100:.1f}%")
    print(f"  TV-BC avg return:      {tvbc_avg_ret[-1]:.3f}  "
          f"({100*tvbc_avg_ret[-1]/avg_ref:.1f}% of teacher)  "
          f"goal={tvbc_avg_goal[-1]*100:.1f}%")
    print(f"  Difference (TV-BC-BC): {tvbc_avg_ret[-1]-bc_avg_ret[-1]:+.3f}")
    winner = "TV-BC" if tvbc_avg_ret[-1] > bc_avg_ret[-1] else "BC"
    print(f"  Winner:                {winner}")

    # Best round for each method
    bc_best_rnd   = int(np.argmax(bc_avg_ret)) + 1
    tvbc_best_rnd = int(np.argmax(tvbc_avg_ret)) + 1
    print(f"\n  BC   best avg return: {max(bc_avg_ret):.3f} at round {bc_best_rnd}")
    print(f"  TV-BC best avg return: {max(tvbc_avg_ret):.3f} at round {tvbc_best_rnd}")

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "bc_avg_ret"    : bc_avg_ret,
        "tvbc_avg_ret"  : tvbc_avg_ret,
        "bc_avg_goal"   : bc_avg_goal,
        "tvbc_avg_goal" : tvbc_avg_goal,
        "pool_sizes"    : pool_sizes,
        "mean_tv_log"   : mean_tv_log,
        "test_refs"     : test_refs,
        "avg_ref"       : avg_ref,
        "TEST_SEEDS"    : TEST_SEEDS,
        "TRAIN_SEEDS"   : TRAIN_SEEDS,
        "N_ROUNDS"      : N_ROUNDS,
        "N_TRAPS"       : N_TRAPS,
        "BETA_TV"       : BETA_TV,
    }
    with open("results/data/generalisation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/generalisation_results.pkl")

    # ── Plot ───────────────────────────────────────────────────────────
    rounds = np.arange(1, N_ROUNDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rounds, bc_avg_ret,
                 color="#e07b39", lw=2, label="BC", marker="o", ms=3)
    axes[0].plot(rounds, tvbc_avg_ret,
                 color="#3a7ebf", lw=2, label="TV-BC", marker="s", ms=3)
    axes[0].axhline(y=avg_ref, color="green", linestyle="--",
                    lw=1.5, label=f"Teacher avg ({avg_ref:.1f})")
    axes[0].set_xlabel("Training Round (unique maps seen)", fontsize=11)
    axes[0].set_ylabel("Avg Return over 20 Test Maps", fontsize=11)
    axes[0].set_title("Generalisation to Unseen Maps\n"
                      "(avg over 20 held-out maps)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, [r*100 for r in bc_avg_goal],
                 color="#e07b39", lw=2, label="BC", marker="o", ms=3)
    axes[1].plot(rounds, [r*100 for r in tvbc_avg_goal],
                 color="#3a7ebf", lw=2, label="TV-BC", marker="s", ms=3)
    axes[1].set_xlabel("Training Round", fontsize=11)
    axes[1].set_ylabel("Avg Goal Rate (%) over 20 Test Maps", fontsize=11)
    axes[1].set_title("Goal Rate — Unseen Maps", fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Generalisation  |  {N_ROUNDS} training maps  |  "
        f"20 test maps  |  {N_TRAPS} traps  |  "
        f"equal weight selection  |  beta={BETA_TV}",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("results/plots/generalisation_curves.png", dpi=150)
    plt.close()
    print(f"  Plot: results/plots/generalisation_curves.png")


if __name__ == "__main__":
    run()
