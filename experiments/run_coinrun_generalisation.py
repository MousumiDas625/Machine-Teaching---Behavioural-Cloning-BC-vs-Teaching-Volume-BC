# experiments/run_coinrun_generalisation.py
#
# TV-BC vs BC on CoinRun.
# Train on level 10 demonstrations.
# Test on levels 11-30 (never seen during training).
#
# Why coinrun shows teaching behaviour:
#   Obstacles force detours — teacher demonstrates avoidance
#   TV formula: trap-entering steps get very negative TV
#   TV-BC filters these out, BC does not
#   Result: TV-BC should generalise better to unseen levels

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import procgen

from agents.ppo_expert import CNNPolicy, DEVICE
from agents.pavel_policy import load_pavel_expert, PavelPolicy

# ── Parameters ────────────────────────────────────────────────────────
GAME        = 'coinrun'
FIXED_LEVEL = 10
N_ROUNDS    = 50
TRAJS_PER_ROUND = 5
MAX_STEPS   = 500
NOISE_EPS   = 0.05
TRAIN_STEPS = 200
BATCH_SIZE  = 16
BETA_TV     = 1.0
ETA         = 1e-4
N_EVAL_EPS  = 20
N_ACTIONS   = 15
SEEDS       = [0, 1, 2]

# Train on level 10 only
TRAIN_LEVEL = 10

# Test on levels 11-30 — never seen during training
TEST_LEVELS = list(range(11, 31))

EXPERT_PATH = "/scr/pavel/data/goal-misgen/policy/icml/coinrun/icml2_coinrun_exp0_0p/2026-01-13__06-46-28__seed_6033/model_200015872.pth"
SAVE_DIR    = "results/coinrun/data"
PLOT_DIR    = "results/coinrun/plots"
VIDEO_DIR   = "results/coinrun/videos"

os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)


def load_expert():
    return load_pavel_expert(EXPERT_PATH)


def collect_trajectory(expert, seed=0):
    rng = np.random.default_rng(seed)
    env = procgen.ProcgenEnv(
        num_envs=1, env_name=GAME,
        num_levels=1, start_level=TRAIN_LEVEL,
        distribution_mode='easy')
    obs   = env.reset()['rgb'][0]
    traj  = []
    done  = False
    steps = 0
    while not done and steps < MAX_STEPS:
        if rng.random() < NOISE_EPS:
            action = int(rng.integers(0, N_ACTIONS))
        else:
            action = expert.act_greedy(obs)
        teacher_loss = expert.loss_at(obs, action)
        traj.append((obs.copy(), action, teacher_loss))
        obs_dict, _, done_arr, _ = env.step(np.array([action]))
        obs   = obs_dict['rgb'][0]
        done  = bool(done_arr[0])
        steps += 1
    env.close()
    return traj


def compute_tv_batch(trajs, policy):
    scores = []
    dev = next(policy.parameters()).device
    for traj in trajs:
        if len(traj) == 0:
            scores.append(0.0)
            continue
        sample = traj[::max(1, len(traj)//10)][:10]
        policy.zero_grad()
        total_loss  = torch.tensor(0.0, device=dev)
        avg_teach_l = 0.0
        for (obs, action, tl) in sample:
            loss        = policy.compute_loss(obs, action)
            total_loss  = total_loss + loss
            avg_teach_l += tl
        avg_loss    = total_loss / len(sample)
        avg_teach_l = avg_teach_l / len(sample)
        avg_loss.backward()
        grads = [p.grad.detach().view(-1) if p.grad is not None
                 else torch.zeros(p.numel(), device=dev)
                 for p in policy.parameters()]
        flat     = torch.cat(grads)
        gnorm_sq = float(flat.dot(flat).item())
        term1    = -(ETA ** 2) * gnorm_sq
        term2    = 2.0 * ETA * (avg_loss.item() - avg_teach_l)
        scores.append(term1 + term2)
    return scores


def select_tvbc(all_trajs, all_tv, rng):
    N = len(all_trajs)
    if N == 0:
        return [], np.array([])
    tv_arr = np.array(all_tv, dtype=np.float64)
    tv_s   = tv_arr - tv_arr.max()
    probs  = np.exp(BETA_TV * tv_s)
    probs /= probs.sum()
    idx    = list(rng.choice(N, size=N, replace=False, p=probs))
    return [all_trajs[i] for i in idx], tv_arr


def select_bc(all_trajs, rng):
    N   = len(all_trajs)
    idx = list(rng.choice(N, size=N, replace=False))
    return [all_trajs[i] for i in idx]


def flatten(trajs):
    return [step for traj in trajs for step in traj]


def bc_step(policy, optimiser, batch):
    policy.train()
    optimiser.zero_grad()
    dev  = next(policy.parameters()).device
    loss = torch.tensor(0.0, device=dev)
    for (obs, action, _) in batch:
        loss = loss + policy.compute_loss(obs, action)
    avg  = loss / len(batch)
    avg.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()
    return avg.item()


def tvbc_step(policy, batch, rng):
    dev = next(policy.parameters()).device
    n   = len(batch)

    # Step 1: TV for all
    tv_scores = []
    for (obs, action, tl) in batch:
        policy.zero_grad()
        loss = policy.compute_loss(obs, action)
        loss.backward()
        grads = [p.grad.detach().view(-1) if p.grad is not None
                 else torch.zeros(p.numel(), device=dev)
                 for p in policy.parameters()]
        flat     = torch.cat(grads)
        gnorm_sq = float(flat.dot(flat).item())
        term1    = -(ETA ** 2) * gnorm_sq
        term2    = 2.0 * ETA * (loss.item() - tl)
        tv_scores.append(term1 + term2)

    # Step 2: Sample chosen
    tv_arr  = np.array(tv_scores, dtype=np.float64)
    tv_arr -= tv_arr.max()
    probs   = np.exp(BETA_TV * tv_arr)
    probs  /= probs.sum()
    t       = int(rng.choice(n, p=probs))
    obs_t, a_t, tl_t = batch[t]

    # Step 3: Naive BC step → theta_hat
    policy.zero_grad()
    loss_t = policy.compute_loss(obs_t, a_t)
    loss_t.backward()
    grads_t = [p.grad.detach().view(-1) if p.grad is not None
               else torch.zeros(p.numel(), device=dev)
               for p in policy.parameters()]
    g_t = torch.cat(grads_t)

    theta_hat = policy.clone()
    with torch.no_grad():
        offset = 0
        for p_o, p_h in zip(policy.parameters(),
                             theta_hat.parameters()):
            numel = p_o.numel()
            p_h.data.copy_(
                p_o.data - ETA * g_t[offset:offset+numel].view(p_o.shape))
            offset += numel

    # Step 4: Recompute TV at theta_hat
    tv_hat = []
    for (obs, action, tl) in batch:
        theta_hat.zero_grad()
        loss = theta_hat.compute_loss(obs, action)
        loss.backward()
        grads = [p.grad.detach().view(-1) if p.grad is not None
                 else torch.zeros(p.numel(), device=dev)
                 for p in theta_hat.parameters()]
        flat     = torch.cat(grads)
        gnorm_sq = float(flat.dot(flat).item())
        tv_hat.append(-(ETA**2)*gnorm_sq + 2*ETA*(loss.item()-tl))
    tv_hat_arr  = np.array(tv_hat, dtype=np.float64)
    tv_hat_arr -= tv_hat_arr.max()
    q_hat       = np.exp(BETA_TV * tv_hat_arr)
    q_hat      /= q_hat.sum()

    # Step 5: g_q
    n_params = sum(p.numel() for p in theta_hat.parameters())
    g_q      = torch.zeros(n_params, device=dev)
    for i, (obs, action, _) in enumerate(batch):
        theta_hat.zero_grad()
        loss_i = theta_hat.compute_loss(obs, action)
        loss_i.backward()
        grads_i = [p.grad.detach().view(-1) if p.grad is not None
                   else torch.zeros(p.numel(), device=dev)
                   for p in theta_hat.parameters()]
        g_q += float(q_hat[i]) * torch.cat(grads_i)

    # Step 6: Correction
    theta_hat.zero_grad()
    loss_th = theta_hat.compute_loss(obs_t, a_t)
    loss_th.backward()
    grads_th = [p.grad.detach().view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=dev)
                for p in theta_hat.parameters()]
    g_th       = torch.cat(grads_th)
    correction = 2.0 * BETA_TV * (ETA**2) * (g_th - g_q)
    corr_norm  = float(correction.norm().item())
    if corr_norm > 1.0:
        correction = correction / corr_norm

    with torch.no_grad():
        offset = 0
        for p_m, p_h in zip(policy.parameters(),
                             theta_hat.parameters()):
            numel = p_m.numel()
            p_m.data.copy_(
                p_h.data - correction[offset:offset+numel].view(p_m.shape))
            offset += numel


def evaluate_on_levels(policy, levels, n_eps=N_EVAL_EPS):
    policy.eval()
    all_returns = []
    all_solved  = []
    for level in levels:
        env = procgen.ProcgenEnv(
            num_envs=1, env_name=GAME,
            num_levels=1, start_level=level,
            distribution_mode='easy')
        ep_returns = []
        for _ in range(n_eps):
            obs   = env.reset()['rgb'][0]
            done  = False; ep_r = 0.0; steps = 0
            while not done and steps < MAX_STEPS:
                with torch.no_grad():
                    probs = policy.get_action_probs(obs).numpy()
                action = int(np.argmax(probs))
                od, ra, da, _ = env.step(np.array([action]))
                obs   = od['rgb'][0]
                ep_r += float(ra[0])
                done  = bool(da[0])
                steps += 1
            ep_returns.append(ep_r)
        env.close()
        all_returns.append(float(np.mean(ep_returns)))
        all_solved.append(float(np.mean([r > 0 for r in ep_returns])))
    return float(np.mean(all_returns)), float(np.mean(all_solved))


def run_one_seed(seed_init, expert, expert_ret, expert_solved):
    print(f"\n  {'─'*55}")
    print(f"  Seed {seed_init}")
    print(f"  {'─'*55}")

    bc_policy   = CNNPolicy(n_actions=N_ACTIONS).to(DEVICE)
    tvbc_policy = CNNPolicy(n_actions=N_ACTIONS).to(DEVICE)
    torch.manual_seed(seed_init)
    for p in bc_policy.parameters():
        if p.dim() > 1:
            nn.init.orthogonal_(p)
    tvbc_policy.load_state_dict(bc_policy.state_dict())

    bc_opt  = optim.Adam(bc_policy.parameters(),   lr=ETA)
    tvbc_rng = np.random.default_rng(seed_init)

    bc_pool = []; bc_tv = []
    tv_pool = []; tv_tv = []

    bc_rets = []; tv_rets = []
    bc_sols = []; tv_sols = []
    traj_seed = seed_init * 10000

    for rnd in range(N_ROUNDS):
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            t = collect_trajectory(expert, seed=traj_seed)
            new_trajs.append(t)
            traj_seed += 1

        bc_new_tv = compute_tv_batch(new_trajs, bc_policy)
        tv_new_tv = compute_tv_batch(new_trajs, tvbc_policy)

        bc_pool.extend(new_trajs); bc_tv.extend(bc_new_tv)
        tv_pool.extend(new_trajs); tv_tv.extend(tv_new_tv)

        rng_s = np.random.default_rng(42 + rnd + seed_init*1000)
        tv_sel, _ = select_tvbc(tv_pool, tv_tv, rng_s)
        bc_sel    = select_bc(bc_pool,
                    np.random.default_rng(42+rnd+seed_init*1000+500))

        bc_flat  = flatten(bc_sel)
        tv_flat  = flatten(tv_sel)

        rng_t = np.random.default_rng(42+rnd+seed_init*1000+2000)
        for _ in range(TRAIN_STEPS):
            bs = min(BATCH_SIZE, len(bc_flat))
            ts = min(BATCH_SIZE, len(tv_flat))
            if bs == 0 or ts == 0:
                break
            bi = rng_t.choice(len(bc_flat),  size=bs, replace=False)
            ti = rng_t.choice(len(tv_flat),  size=ts, replace=False)
            bc_step(bc_policy, bc_opt, [bc_flat[i] for i in bi])
            tvbc_step(tvbc_policy,     [tv_flat[i] for i in ti],
                      tvbc_rng)

        bc_r,  bc_s  = evaluate_on_levels(bc_policy,   TEST_LEVELS)
        tv_r,  tv_s  = evaluate_on_levels(tvbc_policy, TEST_LEVELS)

        bc_rets.append(bc_r);  tv_rets.append(tv_r)
        bc_sols.append(bc_s);  tv_sols.append(tv_s)

        if (rnd+1) % 10 == 0:
            print(f"    Round {rnd+1:>3} | "
                  f"BC={bc_r:.3f} ({bc_s*100:.1f}%) | "
                  f"TV={tv_r:.3f} ({tv_s*100:.1f}%)")

    torch.save(bc_policy.state_dict(),
               f"{SAVE_DIR}/bc_policy_coinrun_seed{seed_init}.pt")
    torch.save(tvbc_policy.state_dict(),
               f"{SAVE_DIR}/tvbc_policy_coinrun_seed{seed_init}.pt")
    return bc_rets, tv_rets, bc_sols, tv_sols


def run():
    print(f"{'='*60}")
    print(f"  CoinRun — TV-BC vs BC Generalisation")
    print(f"{'='*60}")
    print(f"  Train level:  {TRAIN_LEVEL} (fixed)")
    print(f"  Test levels:  {TEST_LEVELS[0]}–{TEST_LEVELS[-1]}")
    print(f"  N_ROUNDS:     {N_ROUNDS}")
    print(f"  Seeds:        {SEEDS}")
    print(f"  Device:       {DEVICE}")

    expert = load_expert()
    expert_ret, expert_solved = evaluate_on_levels(
        expert, TEST_LEVELS, n_eps=20)
    print(f"\n  Expert on test levels: "
          f"return={expert_ret:.3f} "
          f"solve={expert_solved*100:.1f}%")

    all_bc_ret = []; all_tv_ret = []
    all_bc_sol = []; all_tv_sol = []

    for seed in SEEDS:
        bc_r, tv_r, bc_s, tv_s = run_one_seed(
            seed, expert, expert_ret, expert_solved)
        all_bc_ret.append(bc_r); all_tv_ret.append(tv_r)
        all_bc_sol.append(bc_s); all_tv_sol.append(tv_s)

    bc_mean  = np.mean(all_bc_ret, axis=0)
    tv_mean  = np.mean(all_tv_ret, axis=0)
    bc_std   = np.std(all_bc_ret,  axis=0)
    tv_std   = np.std(all_tv_ret,  axis=0)
    bc_sm    = np.mean(all_bc_sol, axis=0)
    tv_sm    = np.mean(all_tv_sol, axis=0)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — avg {len(SEEDS)} seeds x "
          f"{len(TEST_LEVELS)} test levels")
    print(f"{'='*60}")
    print(f"  Expert:  {expert_ret:.3f}  "
          f"solve={expert_solved*100:.1f}%")
    print(f"  BC:      {bc_mean[-1]:.3f} "
          f"(+/-{bc_std[-1]:.3f})  "
          f"solve={bc_sm[-1]*100:.1f}%")
    print(f"  TV-BC:   {tv_mean[-1]:.3f} "
          f"(+/-{tv_std[-1]:.3f})  "
          f"solve={tv_sm[-1]*100:.1f}%")
    print(f"  Diff:    {tv_mean[-1]-bc_mean[-1]:+.3f}")
    winner = "TV-BC" if tv_mean[-1] > bc_mean[-1] else "BC"
    print(f"  Winner:  {winner}")

    results = {
        "bc_mean": bc_mean.tolist(), "tv_mean": tv_mean.tolist(),
        "bc_std":  bc_std.tolist(),  "tv_std":  tv_std.tolist(),
        "bc_sm":   bc_sm.tolist(),   "tv_sm":   tv_sm.tolist(),
        "expert_ret": expert_ret, "expert_solved": expert_solved,
        "SEEDS": SEEDS, "N_ROUNDS": N_ROUNDS,
        "TRAIN_LEVEL": TRAIN_LEVEL, "TEST_LEVELS": TEST_LEVELS,
        "BETA_TV": BETA_TV, "ETA": ETA,
    }
    with open(f"{SAVE_DIR}/coinrun_generalisation_results.pkl",
              "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: {SAVE_DIR}/coinrun_generalisation_results.pkl")

    rounds = np.arange(1, N_ROUNDS+1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(rounds, bc_mean, color="#e07b39", lw=2,
                 label="BC", marker="o", ms=2)
    axes[0].plot(rounds, tv_mean, color="#3a7ebf", lw=2,
                 label="TV-BC", marker="s", ms=2)
    axes[0].fill_between(rounds, bc_mean-bc_std, bc_mean+bc_std,
                          color="#e07b39", alpha=0.2)
    axes[0].fill_between(rounds, tv_mean-tv_std, tv_mean+tv_std,
                          color="#3a7ebf", alpha=0.2)
    axes[0].axhline(y=expert_ret, color="green", linestyle="--",
                    lw=1.5, label=f"Expert ({expert_ret:.2f})")
    axes[0].set_xlabel("Training Round", fontsize=11)
    axes[0].set_ylabel("Avg Return — 20 test levels", fontsize=11)
    axes[0].set_title(f"CoinRun Level {TRAIN_LEVEL} → "
                      f"Generalisation to Levels 11-30", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rounds, [r*100 for r in bc_sm],
                 color="#e07b39", lw=2, label="BC", marker="o", ms=2)
    axes[1].plot(rounds, [r*100 for r in tv_sm],
                 color="#3a7ebf", lw=2, label="TV-BC", marker="s", ms=2)
    axes[1].axhline(y=expert_solved*100, color="green",
                    linestyle="--", lw=1.5)
    axes[1].set_xlabel("Training Round", fontsize=11)
    axes[1].set_ylabel("Solve Rate (%) — 20 test levels", fontsize=11)
    axes[1].set_title("CoinRun Solve Rate", fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(
        f"CoinRun | Train level {TRAIN_LEVEL} | "
        f"Test levels 11-30 | {len(SEEDS)} seeds | "
        f"beta={BETA_TV}",
        fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/coinrun_generalisation.png", dpi=150)
    plt.close()
    print(f"  Plot: {PLOT_DIR}/coinrun_generalisation.png")


if __name__ == "__main__":
    run()
