# experiments/run_generalisation_v2.py
#
# Generalisation experiment v2 — multiple seeds + larger map
#
# Changes from v1:
#   1. MULTIPLE SEEDS: runs 5 times with different network inits
#      averages results, plots with shaded error bands
#   2. LARGER MAP: 16x16 grid (256 cells) instead of 8x8 (64 cells)
#   3. MORE MAPS: 100 training maps, 50 test maps
#   4. MORE OBSTACLES: 60 traps on 16x16 grid
#   5. Larger network: 512->512->256->4 (auto-scaled)

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
GRID_SIZE       = 16       # 16x16 grid = 256 cells
N_STATES        = GRID_SIZE * GRID_SIZE   # 256
OBS_DIM         = N_STATES * 2            # 512 = 256 map + 256 position
ACTION_DIM      = 4

N_ROUNDS        = 100      # 100 unique training maps
TRAJS_PER_ROUND = 5
TRAJ_LENGTH     = 100      # longer episodes for larger grid
NOISE_EPS       = 0.30
TRAIN_STEPS     = 500
BATCH_SIZE      = 20
BETA_TV         = 1.0
ETA             = 0.01
GAMMA           = 0.99
SOFTMAX_TEMP    = 0.1
N_EVAL_EPS      = 30       # slightly fewer for speed
N_GOALS         = 5        # more goals on larger grid
N_TRAPS         = 60       # more obstacles on larger grid

# Multiple seeds for statistical robustness
SEEDS = [0, 1, 2, 3, 4]

# Training map seeds
TRAIN_SEEDS = [1000 + i * 13 for i in range(N_ROUNDS)]

# 50 held-out test maps
TEST_SEEDS = {f'test_{i}': 5000 + i for i in range(50)}

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

assert all(s not in TRAIN_SEEDS for s in TEST_SEEDS.values())

# ══════════════════════════════════════════════════════════════════════
# Environment and teacher
# ══════════════════════════════════════════════════════════════════════
def make_env(seed):
    return GridWorldMapObs(
        grid_size=GRID_SIZE, max_steps=TRAJ_LENGTH,
        seed=seed, n_goals=N_GOALS, n_traps=N_TRAPS,
        terminal=False, step_penalty=0.0)

def make_teacher(env):
    t = RLTeacher(omega_star=env.omega_star,
                  grid_size=env.grid_size,
                  gamma=GAMMA, softmax_temp=SOFTMAX_TEMP)
    t.train(verbose=False)
    return t

# ══════════════════════════════════════════════════════════════════════
# Trajectory collection
# Stores (obs, action, teacher_loss) — correct teacher ref per map
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, eps=NOISE_EPS, seed=0):
    rng   = np.random.default_rng(seed)
    state = env.reset(start_pos=env.START_CELL)
    traj  = []
    done  = False
    while not done:
        obs      = env.get_full_obs(state)
        cell_idx = int(np.argmax(obs[N_STATES:]))
        if rng.random() < eps:
            action = int(rng.integers(0, env.n_actions))
        else:
            action = teacher.act_greedy(state)
        teacher_loss = teacher.loss_at(cell_idx, action)
        traj.append((obs, action, teacher_loss))
        state, _, done, _ = env.step(action)
    return traj

# ══════════════════════════════════════════════════════════════════════
# TV computation — uses stored teacher loss
# ══════════════════════════════════════════════════════════════════════
def compute_tv_batch(trajs, policy, eta=ETA):
    scores = []
    for traj in trajs:
        if len(traj) == 0:
            scores.append(0.0)
            continue
        step_tvs = []
        for (obs, action, teacher_loss) in traj:
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
            term2     = 2.0 * eta * (learner_l - teacher_loss)
            step_tvs.append(term1 + term2)
        scores.append(float(np.mean(step_tvs)))
    return scores

# ══════════════════════════════════════════════════════════════════════
# Selection — equal weightage to all history
# ══════════════════════════════════════════════════════════════════════
def select_tvbc(all_trajs, all_tv, beta, rng):
    N = len(all_trajs)
    if N == 0:
        return [], [], np.array([])
    tv_arr = np.array(all_tv, dtype=np.float64)
    tv_s   = tv_arr - tv_arr.max()
    probs  = np.exp(beta * tv_s)
    probs /= probs.sum()
    idx    = list(rng.choice(N, size=N, replace=False, p=probs))
    return [all_trajs[i] for i in idx], idx, tv_arr

def select_bc(all_trajs, n_select, rng):
    N   = len(all_trajs)
    idx = list(rng.choice(N, size=min(n_select, N), replace=False))
    return [all_trajs[i] for i in idx], idx

def flatten(trajs):
    result = []
    for traj in trajs:
        for step in traj:
            result.append(step)
    return result

# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════
def evaluate(policy, env, n_eps=N_EVAL_EPS, seed=99):
    rng     = np.random.default_rng(seed)
    total   = 0.0
    n_goals = 0
    policy.eval()
    for ep in range(n_eps):
        start = env.START_CELL if ep < n_eps // 2 \
                else int(rng.integers(0, env.n_states))
        state   = env.reset(start_pos=start)
        done    = False; ep_r = 0.0; reached = False
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

def eval_teacher_return(teacher, env, n_eps=20):
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
# Single seed run
# ══════════════════════════════════════════════════════════════════════
def run_one_seed(seed_init, test_envs, test_refs,
                 avg_ref, names, verbose=True):
    """Run entire experiment for one network initialisation seed."""

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  Seed {seed_init}")
        print(f"  {'─'*60}")

    bc_policy   = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=seed_init)
    tvbc_policy = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=seed_init)

    bc_learner   = StandardBC(policy=bc_policy, eta=ETA, seed=42)
    tvbc_learner = TeacherAwareBC(
        policy=tvbc_policy, teacher=None,
        eta=ETA, beta=BETA_TV, seed=42)

    bc_pool   = []; bc_tv   = []
    tvbc_pool = []; tvbc_tv = []

    bc_avg_ret    = []; tvbc_avg_ret  = []
    bc_avg_goal   = []; tvbc_avg_goal = []
    traj_seed     = seed_init * 10000  # unique per seed

    for rnd in range(N_ROUNDS):

        train_env = make_env(TRAIN_SEEDS[rnd])
        train_tch = make_teacher(train_env)
        tvbc_learner.teacher = train_tch

        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            t = collect_trajectory(train_env, train_tch,
                                    eps=NOISE_EPS, seed=traj_seed)
            new_trajs.append(t)
            traj_seed += 1

        bc_new_tv   = compute_tv_batch(new_trajs, bc_policy)
        tvbc_new_tv = compute_tv_batch(new_trajs, tvbc_policy)

        bc_pool.extend(new_trajs);   bc_tv.extend(bc_new_tv)
        tvbc_pool.extend(new_trajs); tvbc_tv.extend(tvbc_new_tv)

        rng_s = np.random.default_rng(42 + rnd + seed_init * 1000)
        tvbc_sel, _, _ = select_tvbc(
            tvbc_pool, tvbc_tv, beta=BETA_TV, rng=rng_s)
        bc_sel, _ = select_bc(
            bc_pool, n_select=len(tvbc_sel),
            rng=np.random.default_rng(
                42 + rnd + seed_init * 1000 + 500))

        bc_flat   = flatten(bc_sel)
        tvbc_flat = flatten(tvbc_sel)

        rng_t = np.random.default_rng(
            42 + rnd + seed_init * 1000 + 2000)
        for _ in range(TRAIN_STEPS):
            bs = min(BATCH_SIZE, len(bc_flat))
            ts = min(BATCH_SIZE, len(tvbc_flat))
            if bs == 0 or ts == 0:
                break
            bi = rng_t.choice(len(bc_flat),   size=bs, replace=False)
            ti = rng_t.choice(len(tvbc_flat), size=ts, replace=False)
            bc_batch   = [(bc_flat[i][0], bc_flat[i][1]) for i in bi]
            tvbc_batch = [tvbc_flat[i] for i in ti]
            bc_learner.step(bc_batch)
            tvbc_learner.step(tvbc_batch)

        # Evaluate on all 50 test maps
        bc_rets = []; tvbc_rets = []
        bc_goals = []; tvbc_goals = []
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

        if verbose and (rnd + 1) % 10 == 0:
            print(f"    Round {rnd+1:>3} | "
                  f"BC={bc_avg:>7.3f} ({bc_g_avg*100:.1f}%) | "
                  f"TV={tvbc_avg:>7.3f} ({tvbc_g_avg*100:.1f}%)")

    return bc_avg_ret, tvbc_avg_ret, bc_avg_goal, tvbc_avg_goal

# ══════════════════════════════════════════════════════════════════════
# Main — run over all seeds and average
# ══════════════════════════════════════════════════════════════════════
def run():
    print(f"{'='*65}")
    print(f"  Generalisation Experiment v2")
    print(f"  16x16 grid | 100 training maps | 50 test maps")
    print(f"  {len(SEEDS)} seeds for statistical robustness")
    print(f"{'='*65}")
    print(f"  Grid:         {GRID_SIZE}x{GRID_SIZE} = {N_STATES} cells")
    print(f"  Obs dim:      {OBS_DIM} ({N_STATES} map + {N_STATES} pos)")
    print(f"  N_TRAPS:      {N_TRAPS}  N_GOALS: {N_GOALS}")
    print(f"  N_ROUNDS:     {N_ROUNDS}")
    print(f"  Test maps:    {len(TEST_SEEDS)}")
    print(f"  Seeds:        {SEEDS}")
    print(f"  Beta:         {BETA_TV}  ETA: {ETA}")
    print(f"  Train steps:  {TRAIN_STEPS}/round")

    # Build 50 held-out test environments once
    print(f"\n  Building {len(TEST_SEEDS)} held-out test environments...")
    test_envs = {}; test_refs = {}
    names = list(TEST_SEEDS.keys())
    for name, seed in TEST_SEEDS.items():
        env = make_env(seed)
        tch = make_teacher(env)
        test_envs[name] = env
        test_refs[name] = eval_teacher_return(tch, env)

    avg_ref = float(np.mean(list(test_refs.values())))
    print(f"  Teacher avg return: {avg_ref:.3f}")
    print(f"  Teacher range: [{min(test_refs.values()):.2f}, "
          f"{max(test_refs.values()):.2f}]")

    # Check network size
    sample_net = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=0)
    print(f"  Network params: {sample_net.count_parameters():,}")
    del sample_net

    # Run over all seeds
    all_bc_ret    = []
    all_tvbc_ret  = []
    all_bc_goal   = []
    all_tvbc_goal = []

    for seed_init in SEEDS:
        bc_r, tvbc_r, bc_g, tvbc_g = run_one_seed(
            seed_init=seed_init,
            test_envs=test_envs,
            test_refs=test_refs,
            avg_ref=avg_ref,
            names=names,
            verbose=True)
        all_bc_ret.append(bc_r)
        all_tvbc_ret.append(tvbc_r)
        all_bc_goal.append(bc_g)
        all_tvbc_goal.append(tvbc_g)

    # Average across seeds
    bc_mean    = np.mean(all_bc_ret,    axis=0)
    tvbc_mean  = np.mean(all_tvbc_ret,  axis=0)
    bc_std     = np.std(all_bc_ret,     axis=0)
    tvbc_std   = np.std(all_tvbc_ret,   axis=0)
    bc_g_mean  = np.mean(all_bc_goal,   axis=0)
    tvbc_g_mean= np.mean(all_tvbc_goal, axis=0)
    bc_g_std   = np.std(all_bc_goal,    axis=0)
    tvbc_g_std = np.std(all_tvbc_goal,  axis=0)

    # Final summary
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS (avg over {len(SEEDS)} seeds x 50 test maps)")
    print(f"{'='*65}")
    print(f"  Teacher avg return:    {avg_ref:.3f}")
    print(f"  BC   avg return:       {bc_mean[-1]:.3f} "
          f"(+/-{bc_std[-1]:.3f})  "
          f"({100*bc_mean[-1]/avg_ref:.1f}%)  "
          f"goal={bc_g_mean[-1]*100:.1f}%")
    print(f"  TV-BC avg return:      {tvbc_mean[-1]:.3f} "
          f"(+/-{tvbc_std[-1]:.3f})  "
          f"({100*tvbc_mean[-1]/avg_ref:.1f}%)  "
          f"goal={tvbc_g_mean[-1]*100:.1f}%")
    print(f"  Difference (TV-BC-BC): "
          f"{tvbc_mean[-1]-bc_mean[-1]:+.3f}")
    winner = "TV-BC" if tvbc_mean[-1] > bc_mean[-1] else "BC"
    print(f"  Winner:                {winner}")

    # Save
    results = {
        "bc_mean": bc_mean.tolist(),
        "tvbc_mean": tvbc_mean.tolist(),
        "bc_std": bc_std.tolist(),
        "tvbc_std": tvbc_std.tolist(),
        "bc_g_mean": bc_g_mean.tolist(),
        "tvbc_g_mean": tvbc_g_mean.tolist(),
        "bc_g_std": bc_g_std.tolist(),
        "tvbc_g_std": tvbc_g_std.tolist(),
        "all_bc_ret": all_bc_ret,
        "all_tvbc_ret": all_tvbc_ret,
        "avg_ref": avg_ref,
        "SEEDS": SEEDS,
        "N_ROUNDS": N_ROUNDS,
        "GRID_SIZE": GRID_SIZE,
        "N_TRAPS": N_TRAPS,
        "N_GOALS": N_GOALS,
        "BETA_TV": BETA_TV,
    }
    with open("results/data/generalisation_v2_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/generalisation_v2_results.pkl")

    # Plot with error bands
    rounds = np.arange(1, N_ROUNDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Return curves with shaded std bands
    axes[0].plot(rounds, bc_mean,   color="#e07b39",
                 lw=2, label="BC",    marker="o", ms=2)
    axes[0].plot(rounds, tvbc_mean, color="#3a7ebf",
                 lw=2, label="TV-BC", marker="s", ms=2)
    axes[0].fill_between(rounds,
                          bc_mean - bc_std,
                          bc_mean + bc_std,
                          color="#e07b39", alpha=0.2)
    axes[0].fill_between(rounds,
                          tvbc_mean - tvbc_std,
                          tvbc_mean + tvbc_std,
                          color="#3a7ebf", alpha=0.2)
    axes[0].axhline(y=avg_ref, color="green", linestyle="--",
                    lw=1.5, label=f"Teacher ({avg_ref:.1f})")
    axes[0].set_xlabel("Training Round", fontsize=11)
    axes[0].set_ylabel("Avg Return (50 test maps)", fontsize=11)
    axes[0].set_title(
        f"Generalisation — {GRID_SIZE}x{GRID_SIZE} Grid\n"
        f"(mean ± std over {len(SEEDS)} seeds)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Goal rate curves
    axes[1].plot(rounds, [r*100 for r in bc_g_mean],
                 color="#e07b39", lw=2, label="BC",    marker="o", ms=2)
    axes[1].plot(rounds, [r*100 for r in tvbc_g_mean],
                 color="#3a7ebf", lw=2, label="TV-BC", marker="s", ms=2)
    axes[1].fill_between(rounds,
                          [(r-s)*100 for r,s in
                           zip(bc_g_mean, bc_g_std)],
                          [(r+s)*100 for r,s in
                           zip(bc_g_mean, bc_g_std)],
                          color="#e07b39", alpha=0.2)
    axes[1].fill_between(rounds,
                          [(r-s)*100 for r,s in
                           zip(tvbc_g_mean, tvbc_g_std)],
                          [(r+s)*100 for r,s in
                           zip(tvbc_g_mean, tvbc_g_std)],
                          color="#3a7ebf", alpha=0.2)
    axes[1].set_xlabel("Training Round", fontsize=11)
    axes[1].set_ylabel("Avg Goal Rate (%) — 50 test maps", fontsize=11)
    axes[1].set_title("Goal Rate — Unseen Maps", fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"{GRID_SIZE}x{GRID_SIZE} grid | {N_ROUNDS} train maps | "
        f"50 test maps | {N_TRAPS} traps | "
        f"{len(SEEDS)} seeds | beta={BETA_TV}",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("results/plots/generalisation_v2_curves.png", dpi=150)
    plt.close()
    print(f"  Plot: results/plots/generalisation_v2_curves.png")


if __name__ == "__main__":
    run()
