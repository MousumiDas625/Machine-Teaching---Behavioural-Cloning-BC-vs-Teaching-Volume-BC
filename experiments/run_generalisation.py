# experiments/run_generalisation.py
#
# Generalisation experiment.
#
# TRAINING:
#   One new random map per round (20 unique training maps total).
#   Robot sees each training map exactly once.
#   Demonstrations collected, TV computed, pool grows with recency weighting.
#   Robot is NEVER evaluated on training maps.
#
# TESTING:
#   3 held-out maps created ONCE before training starts.
#   Seeds chosen to never overlap with training seeds.
#   After every training round, both policies are evaluated
#   on all 3 test maps.
#   Robot has NEVER seen any demonstrations from test maps.
#   Robot must use the 128-dim map observation to navigate.
#
# This directly answers:
#   "Did the robot learn a general navigation policy,
#    or did it just memorise specific maps?"
#
# If TV-BC generalises better than BC:
#   TV-BC learned cleaner, more general demonstrations
#   BC learned from contaminated demos including trap-entering examples
#   which confused its understanding of the general goal/trap structure

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
K_SELECT        = 0        # 0 = use all accumulated trajectories
BETA_TV         = 2.0
ETA             = 0.01
GAMMA           = 0.99
SOFTMAX_TEMP    = 0.1
N_EVAL_EPS      = 50       # episodes per test map evaluation
N_GOALS         = 3
N_TRAPS         = 16       # hard task
RECENCY_DECAY   = 0.9
SEED_INIT       = 0        # network init seed

# Training map seeds — one per round
# Using prime step to avoid patterns
TRAIN_SEEDS = [1000 + i * 13 for i in range(N_ROUNDS)]
# = [1000, 1013, 1026, 1039, ...]

# Test map seeds — completely separate range from training
# These NEVER appear in TRAIN_SEEDS
TEST_SEEDS = {f'test_{i}': 5000 + i for i in range(20)}

OBS_DIM    = 128
ACTION_DIM = 4

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Verify no seed overlap between train and test
# ══════════════════════════════════════════════════════════════════════
assert all(s not in TRAIN_SEEDS for s in TEST_SEEDS.values()), \
    "ERROR: test seed overlaps with training seed"

# ══════════════════════════════════════════════════════════════════════
# Environment builder
# ══════════════════════════════════════════════════════════════════════
def make_env(seed):
    """Build GridWorldMapObs with given seed."""
    return GridWorldMapObs(
        grid_size    = 8,
        max_steps    = TRAJ_LENGTH,
        seed         = seed,
        n_goals      = N_GOALS,
        n_traps      = N_TRAPS,
        terminal     = False,
        step_penalty = 0.0
    )

# ══════════════════════════════════════════════════════════════════════
# Teacher builder
# ══════════════════════════════════════════════════════════════════════
def make_teacher(env):
    """Train Value Iteration teacher on env's reward map."""
    t = RLTeacher(
        omega_star   = env.omega_star,
        grid_size    = env.grid_size,
        gamma        = GAMMA,
        softmax_temp = SOFTMAX_TEMP
    )
    t.train(verbose=False)
    return t

# ══════════════════════════════════════════════════════════════════════
# Trajectory collection
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, eps=NOISE_EPS, seed=0):
    """
    Collect one demonstration trajectory using 128-dim map observation.
    Each (obs, action) pair contains the map encoding for THIS map.
    """
    rng   = np.random.default_rng(seed)
    state = env.reset(start_pos=env.START_CELL)
    traj  = []
    done  = False

    while not done:
        obs    = env.get_full_obs(state)   # 128-dim
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
    Compute TV score for each trajectory using current policy theta.
    TV score = mean TV across all (obs, action) pairs in trajectory.
    """
    scores = []
    for traj in trajs:
        if len(traj) == 0:
            scores.append(0.0)
            continue

        step_tvs = []
        for (obs, action) in traj:
            # Extract cell index from position part of obs
            state_idx = int(np.argmax(obs[64:]))

            # Compute gradient norm squared (Term 1 of TV)
            policy.zero_grad()
            loss  = policy.compute_loss(obs, action)
            loss.backward()
            grads = [p.grad.detach().view(-1)
                     if p.grad is not None
                     else torch.zeros(p.numel())
                     for p in policy.parameters()]
            flat       = torch.cat(grads)
            gnorm_sq   = float(flat.dot(flat).item())
            term1      = -(eta ** 2) * gnorm_sq

            # Loss gap (Term 2 of TV)
            learner_l  = loss.item()
            teacher_l  = teacher.loss_at(state_idx, action)
            term2      = 2.0 * eta * (learner_l - teacher_l)

            step_tvs.append(term1 + term2)

        scores.append(float(np.mean(step_tvs)))
    return scores

# ══════════════════════════════════════════════════════════════════════
# Recency-weighted selection
# ══════════════════════════════════════════════════════════════════════
def select_tvbc(all_trajs, all_tv, all_rounds, current_round,
                beta, rng, decay=RECENCY_DECAY):
    """
    TV-BC selection: softmax over TV * recency_weight.
    recency_weight = decay ^ (current_round - collection_round)
    """
    N    = len(all_trajs)
    ages = np.array([current_round - r for r in all_rounds],
                     dtype=np.float64)
    w    = decay ** ages
    etv  = np.array(all_tv, dtype=np.float64) * w

    if N == 0:
        return [], [], etv

    etv_s = etv - etv.max()
    probs = np.exp(beta * etv_s)
    probs /= probs.sum()
    idx   = list(rng.choice(N, size=N, replace=False, p=probs)) \
            if N > 0 else []
    return [all_trajs[i] for i in idx], idx, etv


def select_bc(all_trajs, all_rounds, current_round,
              n_select, rng, decay=RECENCY_DECAY):
    """
    BC selection: random but recency-weighted.
    Same recency bias as TV-BC — only difference is no TV scoring.
    """
    N    = len(all_trajs)
    ages = np.array([current_round - r for r in all_rounds],
                     dtype=np.float64)
    w    = decay ** ages
    p    = w / w.sum()
    idx  = list(rng.choice(N, size=min(n_select, N),
                            replace=False, p=p))
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
# Policy evaluation — works on ANY environment
# ══════════════════════════════════════════════════════════════════════
def evaluate(policy, env, n_eps=N_EVAL_EPS, seed=99):
    """
    Evaluate policy on given environment.
    Uses env.get_full_obs() so map encoding is always current.
    Works on both training maps and unseen test maps.

    Returns avg_return, goal_rate.
    """
    rng      = np.random.default_rng(seed)
    total    = 0.0
    n_goals  = 0
    policy.eval()

    for ep in range(n_eps):
        # Mix of fixed start and random starts
        start = env.START_CELL if ep < n_eps // 2 \
                else int(rng.integers(0, env.n_states))
        state           = env.reset(start_pos=start)
        done            = False
        ep_r            = 0.0
        reached         = False

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
    """Evaluate teacher greedily on env."""
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
    print(f"{'═'*70}")
    print(f"  Training maps:   {N_ROUNDS} unique maps, one per round")
    print(f"    Seeds:         {TRAIN_SEEDS[:5]} ...")
    print(f"  Test maps:       3 held-out maps, NEVER used for training")
    print(f"    Seeds:         {TEST_SEEDS}")
    print(f"  N_TRAPS:         {N_TRAPS}")
    print(f"  Noise:           {NOISE_EPS}")
    print(f"  Beta:            {BETA_TV}")
    print(f"  Recency decay:   {RECENCY_DECAY}")
    print(f"  Train steps/rnd: {TRAIN_STEPS}")

    # ── Build test environments ONCE — never change ────────────────────
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
        print(f"  [{name}] seed={seed}  teacher_return={ref:.3f}")
        env.print_map()

    # ── Initialise learners ────────────────────────────────────────────
    bc_policy   = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
    tvbc_policy = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)

    bc_learner   = StandardBC(
        policy=bc_policy, eta=ETA, seed=42)
    tvbc_learner = TeacherAwareBC(
        policy=tvbc_policy, teacher=None,
        eta=ETA, beta=BETA_TV, seed=42)

    print(f"\n  Network parameters: {bc_policy.count_parameters():,}")
    print(f"  Both networks initialised with same seed={SEED_INIT}\n")

    # ── Accumulators ───────────────────────────────────────────────────
    bc_pool_trajs   = []
    bc_pool_tv      = []
    bc_pool_rounds  = []

    tvbc_pool_trajs  = []
    tvbc_pool_tv     = []
    tvbc_pool_rounds = []

    # Results logged per round
    # test_ret[name] = [round1_return, round2_return, ...]
    bc_test_ret  = {n: [] for n in TEST_SEEDS}
    bc_test_goal = {n: [] for n in TEST_SEEDS}
    tvbc_test_ret  = {n: [] for n in TEST_SEEDS}
    tvbc_test_goal = {n: [] for n in TEST_SEEDS}
    pool_sizes    = []
    eff_tv_log    = []
    traj_seed     = 0

    # Print header
    names = list(TEST_SEEDS.keys())
    print(f"  {'Rnd':>3} | {'Pool':>5} | "
          + " | ".join(
              f"{'BC_'+n[:3]:>7} {'TV_'+n[:3]:>7}"
              for n in names)
          + f" | {'EffTV':>7}")
    print(f"  {'─'*75}")

    # ══════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════
    for rnd in range(N_ROUNDS):

        train_seed = TRAIN_SEEDS[rnd]

        # ── Build training environment for this round ──────────────────
        # Different map every round — robot never trained on this before
        train_env = make_env(train_seed)
        train_tch = make_teacher(train_env)
        tvbc_learner.teacher = train_tch

        # ── Collect demonstrations ─────────────────────────────────────
        # Observations encode THIS round's map layout
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            t = collect_trajectory(
                train_env, train_tch,
                eps=NOISE_EPS, seed=traj_seed)
            new_trajs.append(t)
            traj_seed += 1

        # ── Compute TV for new trajectories using current theta ────────
        # Freeze immediately after computation
        bc_new_tv   = compute_tv_batch(
            new_trajs, bc_policy,   train_tch)
        tvbc_new_tv = compute_tv_batch(
            new_trajs, tvbc_policy, train_tch)

        # ── Add to pool ────────────────────────────────────────────────
        bc_pool_trajs.extend(new_trajs)
        bc_pool_tv.extend(bc_new_tv)
        bc_pool_rounds.extend([rnd] * len(new_trajs))

        tvbc_pool_trajs.extend(new_trajs)
        tvbc_pool_tv.extend(tvbc_new_tv)
        tvbc_pool_rounds.extend([rnd] * len(new_trajs))

        pool_size = sum(len(t) for t in bc_pool_trajs)
        pool_sizes.append(pool_size)

        # ── Select training subset ─────────────────────────────────────
        rng_s = np.random.default_rng(42 + rnd)

        tvbc_sel, _, eff_tv = select_tvbc(
            tvbc_pool_trajs, tvbc_pool_tv, tvbc_pool_rounds,
            current_round=rnd, beta=BETA_TV, rng=rng_s)

        bc_sel, _ = select_bc(
            bc_pool_trajs, bc_pool_rounds,
            current_round=rnd,
            n_select=len(tvbc_sel),
            rng=np.random.default_rng(42 + rnd + 1000))

        eff_tv_log.append(float(np.mean(eff_tv)))

        # ── Train ──────────────────────────────────────────────────────
        bc_obs,   bc_acts   = flatten(bc_sel)
        tvbc_obs, tvbc_acts = flatten(tvbc_sel)
        bc_pool_flat   = list(zip(bc_obs,   bc_acts))
        tvbc_pool_flat = list(zip(tvbc_obs, tvbc_acts))

        rng_t = np.random.default_rng(42 + rnd + 2000)
        for _ in range(TRAIN_STEPS):
            bs = min(BATCH_SIZE, len(bc_pool_flat))
            ts = min(BATCH_SIZE, len(tvbc_pool_flat))
            if bs == 0 or ts == 0:
                break
            bi = rng_t.choice(len(bc_pool_flat),   size=bs, replace=False)
            ti = rng_t.choice(len(tvbc_pool_flat), size=ts, replace=False)
            bc_learner.step(  [bc_pool_flat[i]   for i in bi])
            tvbc_learner.step([tvbc_pool_flat[i] for i in ti])

        # ── Evaluate on held-out test maps ─────────────────────────────
        # Robot has NEVER trained on these maps
        # This measures genuine generalisation
        row = f"  {rnd+1:>3} | {pool_size:>5} | "
        for name in names:
            bc_r,   bc_g   = evaluate(bc_policy,   test_envs[name])
            tvbc_r, tvbc_g = evaluate(tvbc_policy, test_envs[name])
            bc_test_ret[name].append(bc_r)
            bc_test_goal[name].append(bc_g)
            tvbc_test_ret[name].append(tvbc_r)
            tvbc_test_goal[name].append(tvbc_g)
            row += f"{bc_r:>7.2f} {tvbc_r:>7.2f} | "
        row += f"{eff_tv_log[-1]:>7.4f}"
        print(row)

    # ══════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  FINAL RESULTS ON HELD-OUT TEST MAPS")
    print(f"  Robot trained on {N_ROUNDS} maps, tested on 3 unseen maps")
    print(f"{'═'*70}")

    for name in names:
        bc_r   = bc_test_ret[name][-1]
        tvbc_r = tvbc_test_ret[name][-1]
        bc_g   = bc_test_goal[name][-1]
        tvbc_g = tvbc_test_goal[name][-1]
        ref    = test_refs[name]
        winner = "TV-BC" if tvbc_r > bc_r else "BC"

        print(f"\n  Test map '{name}' (seed={TEST_SEEDS[name]}):")
        print(f"    Teacher return:   {ref:.3f}")
        print(f"    BC   return:  {bc_r:7.3f}  "
              f"({100*bc_r/ref:5.1f}% of teacher)  "
              f"goal={bc_g*100:.1f}%")
        print(f"    TV-BC return: {tvbc_r:7.3f}  "
              f"({100*tvbc_r/ref:5.1f}% of teacher)  "
              f"goal={tvbc_g*100:.1f}%")
        print(f"    Difference: {tvbc_r-bc_r:+.3f}  →  {winner} wins")

    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "bc_test_ret"   : bc_test_ret,
        "tvbc_test_ret" : tvbc_test_ret,
        "bc_test_goal"  : bc_test_goal,
        "tvbc_test_goal": tvbc_test_goal,
        "pool_sizes"    : pool_sizes,
        "eff_tv_log"    : eff_tv_log,
        "test_refs"     : test_refs,
        "TEST_SEEDS"    : TEST_SEEDS,
        "TRAIN_SEEDS"   : TRAIN_SEEDS,
        "N_ROUNDS"      : N_ROUNDS,
        "N_TRAPS"       : N_TRAPS,
        "BETA_TV"       : BETA_TV,
        "RECENCY_DECAY" : RECENCY_DECAY,
    }
    with open("results/data/generalisation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/generalisation_results.pkl")

    # ── Plots ──────────────────────────────────────────────────────────
    rounds = np.arange(1, N_ROUNDS + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = {"easy": "#2ecc71", "medium": "#e67e22", "hard": "#e74c3c"}

    for col, name in enumerate(names):
        ref = test_refs[name]

        # Row 0: return curves
        ax = axes[0][col]
        ax.plot(rounds, bc_test_ret[name],
                color="#e07b39", lw=2,
                label="BC", marker="o", ms=3)
        ax.plot(rounds, tvbc_test_ret[name],
                color="#3a7ebf", lw=2,
                label="TV-BC", marker="s", ms=3)
        ax.axhline(y=ref, color=colors[name],
                   linestyle="--", lw=1.5,
                   label=f"Teacher ({ref:.1f})")
        ax.set_title(
            f"{name.capitalize()} Test Map\n"
            f"(seed={TEST_SEEDS[name]} — never trained on)",
            fontsize=10)
        ax.set_xlabel("Training rounds completed", fontsize=9)
        ax.set_ylabel("Return on test map", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 1: goal rate curves
        ax2 = axes[1][col]
        ax2.plot(rounds,
                 [r*100 for r in bc_test_goal[name]],
                 color="#e07b39", lw=2,
                 label="BC", marker="o", ms=3)
        ax2.plot(rounds,
                 [r*100 for r in tvbc_test_goal[name]],
                 color="#3a7ebf", lw=2,
                 label="TV-BC", marker="s", ms=3)
        ax2.set_title(f"Goal Rate — {name.capitalize()} Test Map",
                      fontsize=10)
        ax2.set_xlabel("Training rounds completed", fontsize=9)
        ax2.set_ylabel("Goal rate (%)", fontsize=9)
        ax2.set_ylim([0, 105])
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"Generalisation to Unseen Maps  |  "
        f"{N_ROUNDS} training maps (one per round)  |  "
        f"{N_TRAPS} traps  |  noise={NOISE_EPS}  "
        f"beta={BETA_TV}  decay={RECENCY_DECAY}",
        fontsize=11)
    plt.tight_layout()
    plt.savefig("results/plots/generalisation_curves.png", dpi=150)
    plt.close()
    print(f"  Plot: results/plots/generalisation_curves.png")


if __name__ == "__main__":
    run()
