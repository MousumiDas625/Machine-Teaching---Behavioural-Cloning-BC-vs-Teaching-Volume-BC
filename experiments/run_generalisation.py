# experiments/run_generalisation.py
#
# Generalisation experiment — faithful to Algorithm 1 (Yuan et al., NeurIPS 2021)
#
# KEY FIX from previous version:
#   Trajectories now store (obs, action, teacher_loss) instead of (obs, action).
#   The teacher_loss is computed at collection time using the CORRECT teacher
#   for that specific map. This means the ITAL correction always uses the
#   right reference loss regardless of which round the example came from.
#   Previously, self.teacher in tv_bc.py was always the CURRENT round's teacher,
#   which gave wrong TV scores for examples from old maps — causing oscillation.
#
# Three design principles:
#   1. EQUAL WEIGHTAGE: softmax over frozen TV scores, no recency decay
#   2. NEW MAP EVERY ROUND: 50 unique training maps
#   3. TEST ON 20 UNSEEN MAPS: evaluation on held-out maps only

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
N_ROUNDS        = 50
TRAJS_PER_ROUND = 5
TRAJ_LENGTH     = 50
NOISE_EPS       = 0.30
TRAIN_STEPS     = 500
BATCH_SIZE      = 20
BETA_TV         = 1.0      # reduced from 2.0 for stability
ETA             = 0.01
GAMMA           = 0.99
SOFTMAX_TEMP    = 0.1
N_EVAL_EPS      = 50
N_GOALS         = 3
N_TRAPS         = 16
SEED_INIT       = 0

TRAIN_SEEDS = [1000 + i * 13 for i in range(N_ROUNDS)]
TEST_SEEDS  = {f'test_{i}': 5000 + i for i in range(20)}

OBS_DIM    = 128
ACTION_DIM = 4

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

assert all(s not in TRAIN_SEEDS for s in TEST_SEEDS.values())

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
# KEY CHANGE: each step stores (obs, action, teacher_loss)
# teacher_loss = -log pi_teacher(action|cell) for THIS map's teacher
# Stored at collection time so ITAL correction always uses correct ref
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, eps=NOISE_EPS, seed=0):
    """
    Collect one trajectory. Each step stores:
      obs          : 128-dim [map_encoding | position]
      action       : int
      teacher_loss : -log pi_teacher(action|cell) for THIS map
    """
    rng   = np.random.default_rng(seed)
    state = env.reset(start_pos=env.START_CELL)
    traj  = []
    done  = False
    while not done:
        obs          = env.get_full_obs(state)
        cell_idx     = int(np.argmax(obs[64:]))
        if rng.random() < eps:
            action = int(rng.integers(0, env.n_actions))
        else:
            action = teacher.act_greedy(state)
        # Store teacher loss at collection time — correct for THIS map
        teacher_loss = teacher.loss_at(cell_idx, action)
        traj.append((obs, action, teacher_loss))
        state, _, done, _ = env.step(action)
    return traj

# ══════════════════════════════════════════════════════════════════════
# TV computation
# Uses stored teacher_loss — no teacher object needed
# ══════════════════════════════════════════════════════════════════════
def compute_tv_batch(trajs, policy, eta=ETA):
    """
    Compute TV score per trajectory using current policy theta.
    Uses teacher_loss stored in each step — no teacher needed here.
    TV = -eta² ||grad||² + 2eta [loss_learner - stored_teacher_loss]
    """
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
    """
    TV-BC: softmax over ALL frozen TV scores. No recency decay.
    Equal weightage to entire history. High TV selected more.
    """
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
    """BC: uniform random from all history. No recency, no TV."""
    N   = len(all_trajs)
    idx = list(rng.choice(N, size=min(n_select, N), replace=False))
    return [all_trajs[i] for i in idx], idx

# ══════════════════════════════════════════════════════════════════════
# Flatten trajectories
# ══════════════════════════════════════════════════════════════════════
def flatten(trajs):
    """Returns list of (obs, action, teacher_loss) tuples."""
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
    print(f"{'='*70}")
    print(f"  Generalisation Experiment — Correct Teacher Reference Fix")
    print(f"{'='*70}")
    print(f"  Training maps:    {N_ROUNDS} unique maps, one per round")
    print(f"  Test maps:        {len(TEST_SEEDS)} held-out maps (seeds 5000-5019)")
    print(f"  Selection:        equal weightage, TV drives TV-BC")
    print(f"  Teacher ref:      stored at collection time (correct per map)")
    print(f"  N_TRAPS:          {N_TRAPS}  Beta: {BETA_TV}  ETA: {ETA}")
    print(f"  Train steps/rnd:  {TRAIN_STEPS}")

    # Build 20 held-out test environments
    test_envs = {}; test_refs = {}
    for name, seed in TEST_SEEDS.items():
        env = make_env(seed)
        tch = make_teacher(env)
        test_envs[name] = env
        test_refs[name] = eval_teacher_return(tch, env)

    avg_ref = float(np.mean(list(test_refs.values())))
    print(f"\n  Teacher avg return: {avg_ref:.3f}")
    print(f"  Teacher range:      [{min(test_refs.values()):.2f}, "
          f"{max(test_refs.values()):.2f}]")

    # Initialise both learners with identical weights
    bc_policy   = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
    tvbc_policy = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
    bc_learner   = StandardBC(policy=bc_policy, eta=ETA, seed=42)
    tvbc_learner = TeacherAwareBC(
        policy=tvbc_policy, teacher=None,
        eta=ETA, beta=BETA_TV, seed=42)

    print(f"  Network params: {bc_policy.count_parameters():,} "
          f"(both identical, seed={SEED_INIT})\n")

    # Pools — store (obs, action, teacher_loss) tuples
    bc_pool   = []; bc_tv   = []
    tvbc_pool = []; tvbc_tv = []

    bc_avg_ret = []; tvbc_avg_ret = []
    bc_avg_goal = []; tvbc_avg_goal = []
    pool_sizes = []; mean_tv_log = []
    traj_seed  = 0

    names = list(TEST_SEEDS.keys())
    print(f"  {'Rnd':>3} | {'Pool':>6} | {'BC avg':>8} | "
          f"{'TV-BC avg':>9} | {'BC%':>6} | {'TV%':>6} | {'MeanTV':>8}")
    print(f"  {'-'*65}")

    for rnd in range(N_ROUNDS):

        # New training map this round
        train_env = make_env(TRAIN_SEEDS[rnd])
        train_tch = make_teacher(train_env)

        # Collect 5 demonstrations — each step stores teacher_loss
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            t = collect_trajectory(train_env, train_tch,
                                    eps=NOISE_EPS, seed=traj_seed)
            new_trajs.append(t)
            traj_seed += 1

        # Compute TV using current theta and STORED teacher losses
        # No teacher object needed — teacher_loss already in each step
        bc_new_tv   = compute_tv_batch(new_trajs, bc_policy)
        tvbc_new_tv = compute_tv_batch(new_trajs, tvbc_policy)

        # Add to pool — equal weight, no round tag
        bc_pool.extend(new_trajs);   bc_tv.extend(bc_new_tv)
        tvbc_pool.extend(new_trajs); tvbc_tv.extend(tvbc_new_tv)

        pool_size = sum(len(t) for t in bc_pool)
        pool_sizes.append(pool_size)

        # Select — equal weightage to all history
        rng_s = np.random.default_rng(42 + rnd)
        tvbc_sel, _, tv_arr = select_tvbc(
            tvbc_pool, tvbc_tv, beta=BETA_TV, rng=rng_s)
        bc_sel, _ = select_bc(
            bc_pool, n_select=len(tvbc_sel),
            rng=np.random.default_rng(42 + rnd + 1000))

        mean_tv_log.append(float(np.mean(tv_arr)))

        # Flatten: list of (obs, action, teacher_loss) tuples
        bc_flat   = flatten(bc_sel)
        tvbc_flat = flatten(tvbc_sel)

        # Train 500 steps
        rng_t = np.random.default_rng(42 + rnd + 2000)
        for _ in range(TRAIN_STEPS):
            bs = min(BATCH_SIZE, len(bc_flat))
            ts = min(BATCH_SIZE, len(tvbc_flat))
            if bs == 0 or ts == 0:
                break
            bi = rng_t.choice(len(bc_flat),   size=bs, replace=False)
            ti = rng_t.choice(len(tvbc_flat), size=ts, replace=False)
            # BC gets (obs, action) only
            bc_batch = [(bc_flat[i][0], bc_flat[i][1]) for i in bi]
            # TV-BC gets (obs, action, teacher_loss) for correct ITAL
            tvbc_batch = [tvbc_flat[i] for i in ti]
            bc_learner.step(bc_batch)
            tvbc_learner.step(tvbc_batch)

        # Evaluate on all 20 held-out test maps
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

        print(f"  {rnd+1:>3} | {pool_size:>6} | {bc_avg:>8.3f} | "
              f"{tvbc_avg:>9.3f} | {bc_g_avg*100:>5.1f}% | "
              f"{tvbc_g_avg*100:>5.1f}% | {mean_tv_log[-1]:>8.4f}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS (avg over 20 held-out test maps)")
    print(f"{'='*70}")
    print(f"  Teacher avg:   {avg_ref:.3f}")
    print(f"  BC   avg:      {bc_avg_ret[-1]:.3f}  "
          f"({100*bc_avg_ret[-1]/avg_ref:.1f}%)  "
          f"goal={bc_avg_goal[-1]*100:.1f}%")
    print(f"  TV-BC avg:     {tvbc_avg_ret[-1]:.3f}  "
          f"({100*tvbc_avg_ret[-1]/avg_ref:.1f}%)  "
          f"goal={tvbc_avg_goal[-1]*100:.1f}%")
    print(f"  Diff:          {tvbc_avg_ret[-1]-bc_avg_ret[-1]:+.3f}")
    winner = "TV-BC" if tvbc_avg_ret[-1] > bc_avg_ret[-1] else "BC"
    print(f"  Winner:        {winner}")
    print(f"\n  BC   best: {max(bc_avg_ret):.3f} at round "
          f"{int(np.argmax(bc_avg_ret))+1}")
    print(f"  TV-BC best: {max(tvbc_avg_ret):.3f} at round "
          f"{int(np.argmax(tvbc_avg_ret))+1}")

    # Save
    results = {
        "bc_avg_ret": bc_avg_ret, "tvbc_avg_ret": tvbc_avg_ret,
        "bc_avg_goal": bc_avg_goal, "tvbc_avg_goal": tvbc_avg_goal,
        "pool_sizes": pool_sizes, "mean_tv_log": mean_tv_log,
        "test_refs": test_refs, "avg_ref": avg_ref,
        "TEST_SEEDS": TEST_SEEDS, "TRAIN_SEEDS": TRAIN_SEEDS,
        "N_ROUNDS": N_ROUNDS, "N_TRAPS": N_TRAPS, "BETA_TV": BETA_TV,
    }
    with open("results/data/generalisation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/generalisation_results.pkl")

    # Plot
    rounds = np.arange(1, N_ROUNDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rounds, bc_avg_ret,   color="#e07b39", lw=2,
                 label="BC",    marker="o", ms=3)
    axes[0].plot(rounds, tvbc_avg_ret, color="#3a7ebf", lw=2,
                 label="TV-BC", marker="s", ms=3)
    axes[0].axhline(y=avg_ref, color="green", linestyle="--",
                    lw=1.5, label=f"Teacher ({avg_ref:.1f})")
    axes[0].set_xlabel("Training Round", fontsize=11)
    axes[0].set_ylabel("Avg Return over 20 Test Maps", fontsize=11)
    axes[0].set_title("Generalisation to Unseen Maps", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rounds, [r*100 for r in bc_avg_goal],
                 color="#e07b39", lw=2, label="BC",    marker="o", ms=3)
    axes[1].plot(rounds, [r*100 for r in tvbc_avg_goal],
                 color="#3a7ebf", lw=2, label="TV-BC", marker="s", ms=3)
    axes[1].set_xlabel("Training Round", fontsize=11)
    axes[1].set_ylabel("Avg Goal Rate (%)", fontsize=11)
    axes[1].set_title("Goal Rate — Unseen Maps", fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(
        f"Generalisation | {N_ROUNDS} train maps | 20 test maps | "
        f"{N_TRAPS} traps | correct teacher ref | beta={BETA_TV}",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("results/plots/generalisation_curves.png", dpi=150)
    plt.close()
    print(f"  Plot: results/plots/generalisation_curves.png")


if __name__ == "__main__":
    run()
