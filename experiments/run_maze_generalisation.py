# experiments/run_maze_generalisation.py
#
# TV-BC vs BC on Procgen Maze — Generalisation Experiment
#
# Same structure as run_generalisation.py but for Procgen Maze:
#   - Each round: one new maze level (new map layout)
#   - Demonstrations collected with stored teacher loss
#   - TV-BC filters noisy/wrong-direction steps via TV score
#   - Evaluated on 20 held-out maze levels never seen during training
#
# Key differences from GridWorld version:
#   - Obs = 64x64x3 RGB image (not 128-dim vector)
#   - Teacher = trained PPO expert (not Value Iteration)
#   - teacher_loss = -log pi_ppo(action|obs) stored at collection time
#   - Policy = CNNPolicy (not MLP)
#   - state_idx not needed (TV uses full image obs directly)

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

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

# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════
N_ROUNDS        = 10       # training rounds = unique maze levels
TRAJS_PER_ROUND = 5        # demonstrations per round
MAX_STEPS       = 100      # max steps per trajectory
NOISE_EPS       = 0.10     # 10% random actions
TRAIN_STEPS     = 50      # gradient updates per round
BATCH_SIZE      = 16       # smaller batch for image obs
BETA_TV         = 1.0      # TV softmax temperature
ETA             = 1e-4     # learning rate (smaller for CNN)
N_EVAL_EPS      = 5       # evaluation episodes per test level
N_ACTIONS       = 15       # Procgen action space

# Training maze levels: 0-49
TRAIN_LEVELS = list(range(N_ROUNDS))

# Test maze levels: 1000-1019 — NEVER used for training
TEST_LEVELS = list(range(1000, 1020))

EXPERT_PATH = "results/data/maze_expert_best.pt"
SEEDS       = [0]   # multiple seeds for robustness

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Load expert
# ══════════════════════════════════════════════════════════════════════
def load_expert(path=EXPERT_PATH):
    policy = CNNPolicy(n_actions=N_ACTIONS).to(DEVICE)
    policy.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True))
    policy.eval()
    return policy

# ══════════════════════════════════════════════════════════════════════
# Trajectory collection — stores (obs, action, teacher_loss)
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(level_seed, expert, eps=NOISE_EPS, seed=0):
    """
    Collect one trajectory on maze level = level_seed.
    Each step stores (obs_64x64x3, action, teacher_loss).
    teacher_loss stored immediately using correct expert reference.
    """
    rng = np.random.default_rng(seed)
    env = procgen.ProcgenEnv(
        num_envs=1, env_name='maze',
        num_levels=1, start_level=level_seed,
        distribution_mode='easy')

    obs   = env.reset()['rgb'][0]   # (64, 64, 3)
    traj  = []
    done  = False
    steps = 0

    while not done and steps < MAX_STEPS:
        if rng.random() < eps:
            action = int(rng.integers(0, N_ACTIONS))
        else:
            action = expert.act_greedy(obs)

        teacher_loss = expert.loss_at(obs, action)
        traj.append((obs.copy(), action, teacher_loss))

        obs_dict, reward_arr, done_arr, _ = env.step(np.array([action]))
        obs    = obs_dict["rgb"][0]
        done   = bool(done_arr[0])
        steps += 1

    env.close()
    return traj

# ══════════════════════════════════════════════════════════════════════
# TV computation — uses stored teacher loss
# ══════════════════════════════════════════════════════════════════════
def compute_tv_batch(trajs, policy, eta=ETA):
    """
    Compute TV per trajectory using current policy theta.
    Uses stored teacher_loss — no expert needed here.
    TV = -eta^2 ||grad||^2 + 2*eta*(loss_learner - stored_teacher_loss)
    """
    scores = []
    for traj in trajs:
        if len(traj) == 0:
            scores.append(0.0)
            continue
        step_tvs = []
        # Subsample steps for speed — TV on every 10th step
        traj_sample = traj[::max(1,len(traj)//5)][:5]
        for (obs, action, teacher_loss) in traj_sample:
            policy.zero_grad()
            loss = policy.compute_loss(obs, action)
            loss.backward()
            grads = [p.grad.detach().view(-1) if p.grad is not None
                     else torch.zeros(p.numel(), device=next(policy.parameters()).device)
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
# BC update step
# ══════════════════════════════════════════════════════════════════════
def bc_step(policy, optimiser, batch):
    """Plain SGD step for BC. Batch = list of (obs, action, tl) tuples."""
    policy.train()
    optimiser.zero_grad()
    total_loss = torch.tensor(0.0).to(DEVICE)
    for (obs, action, _) in batch:
        loss       = policy.compute_loss(obs, action)
        total_loss = total_loss + loss
    avg_loss = total_loss / len(batch)
    avg_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimiser.step()
    return avg_loss.item()

# ══════════════════════════════════════════════════════════════════════
# TV-BC update step (ITAL)
# ══════════════════════════════════════════════════════════════════════
def tvbc_step(policy, batch, beta=BETA_TV, eta=ETA, rng=None):
    """
    Full 6-step ITAL update on one mini-batch.
    Uses stored teacher_loss for correct TV computation.
    Correction clipped to norm <= 1.0.
    """
    if rng is None:
        rng = np.random.default_rng()

    n      = len(batch)
    parsed = batch   # already (obs, action, teacher_loss) tuples

    # Step 1: TV for all examples
    tv_scores   = []
    loss_values = []
    for (obs, action, teacher_loss) in parsed:
        policy.zero_grad()
        loss = policy.compute_loss(obs, action)
        loss.backward()
        grads = [p.grad.detach().view(-1) if p.grad is not None
                 else torch.zeros(p.numel(), device=next(policy.parameters()).device)
                 for p in policy.parameters()]
        flat      = torch.cat(grads)
        gnorm_sq  = float(flat.dot(flat).item())
        term1     = -(eta ** 2) * gnorm_sq
        term2     = 2.0 * eta * (loss.item() - teacher_loss)
        tv_scores.append(term1 + term2)
        loss_values.append(loss.item())

    avg_loss = float(np.mean(loss_values))

    # Step 2: Sample teacher-chosen example
    tv_arr   = np.array(tv_scores, dtype=np.float64)
    tv_arr  -= tv_arr.max()
    weights  = np.exp(beta * tv_arr)
    probs    = weights / weights.sum()
    chosen_t = int(rng.choice(n, p=probs))
    obs_t, a_t, tl_t = parsed[chosen_t]

    # Step 3: Naive BC step on chosen example → theta_hat
    policy.zero_grad()
    loss_t = policy.compute_loss(obs_t, a_t)
    loss_t.backward()
    grads_t = [p.grad.detach().view(-1) if p.grad is not None
               else torch.zeros(p.numel(), device=next(policy.parameters()).device)
               for p in policy.parameters()]
    g_t = torch.cat(grads_t)

    theta_hat = policy.clone()
    with torch.no_grad():
        offset = 0
        for (p_orig, p_hat) in zip(policy.parameters(),
                                    theta_hat.parameters()):
            numel      = p_orig.numel()
            grad_chunk = g_t[offset:offset+numel].view(p_orig.shape)
            p_hat.data.copy_(p_orig.data - eta * grad_chunk)
            offset    += numel

    # Step 4: Recompute TV at theta_hat
    tv_hat = []
    for (obs, action, teacher_loss) in parsed:
        theta_hat.zero_grad()
        loss = theta_hat.compute_loss(obs, action)
        loss.backward()
        grads = [p.grad.detach().view(-1) if p.grad is not None
                 else torch.zeros(p.numel(), device=next(policy.parameters()).device)
                 for p in theta_hat.parameters()]
        flat     = torch.cat(grads)
        gnorm_sq = float(flat.dot(flat).item())
        term1    = -(eta ** 2) * gnorm_sq
        term2    = 2.0 * eta * (loss.item() - teacher_loss)
        tv_hat.append(term1 + term2)

    tv_hat_arr  = np.array(tv_hat, dtype=np.float64)
    tv_hat_arr -= tv_hat_arr.max()
    w_hat       = np.exp(beta * tv_hat_arr)
    q_hat       = w_hat / w_hat.sum()

    # Step 5: Expected gradient g_q
    n_params = sum(p.numel() for p in theta_hat.parameters())
    g_q      = torch.zeros(n_params).to(DEVICE)
    for i, (obs, action, _) in enumerate(parsed):
        theta_hat.zero_grad()
        loss_i = theta_hat.compute_loss(obs, action)
        loss_i.backward()
        grads_i = [p.grad.detach().view(-1) if p.grad is not None
                   else torch.zeros(p.numel(), device=next(policy.parameters()).device)
                   for p in theta_hat.parameters()]
        g_i  = torch.cat(grads_i)
        g_q += float(q_hat[i]) * g_i

    # Step 6: ITAL correction with clipping
    theta_hat.zero_grad()
    loss_t_hat = theta_hat.compute_loss(obs_t, a_t)
    loss_t_hat.backward()
    grads_th = [p.grad.detach().view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=next(policy.parameters()).device)
                for p in theta_hat.parameters()]
    g_t_hat    = torch.cat(grads_th)
    correction = 2.0 * beta * (eta ** 2) * (g_t_hat - g_q)

    corr_norm = float(correction.norm().item())
    if corr_norm > 1.0:
        correction = correction / corr_norm

    with torch.no_grad():
        offset = 0
        for (p_main, p_hat) in zip(policy.parameters(),
                                    theta_hat.parameters()):
            numel      = p_main.numel()
            corr_chunk = correction[offset:offset+numel].view(p_main.shape)
            p_main.data.copy_(p_hat.data - corr_chunk)
            offset    += numel

    return avg_loss

# ══════════════════════════════════════════════════════════════════════
# Evaluation on held-out maze levels
# ══════════════════════════════════════════════════════════════════════
def evaluate_on_levels(policy, test_levels, n_eps=N_EVAL_EPS):
    """
    Evaluate policy on list of maze levels.
    Returns avg return and solve rate (reached goal = reward > 0).
    """
    policy.eval()
    all_returns = []
    all_solved  = []

    for level in test_levels:
        env = procgen.ProcgenEnv(
            num_envs=1, env_name='maze',
            num_levels=1, start_level=level,
            distribution_mode='easy')

        ep_returns = []
        for _ in range(n_eps):
            obs      = env.reset()['rgb'][0]
            done     = False
            ep_r     = 0.0
            steps    = 0
            while not done and steps < MAX_STEPS:
                with torch.no_grad():
                    probs = policy.get_action_probs(obs).numpy()
                action   = int(np.argmax(probs))
                obs_dict, reward_arr, done_arr, _ = env.step(np.array([action]))
                obs = obs_dict['rgb'][0]
                ep_r += float(reward_arr[0])
                done = bool(done_arr[0])
                steps   += 1
            ep_returns.append(ep_r)

        env.close()
        all_returns.append(float(np.mean(ep_returns)))
        all_solved.append(float(np.mean([r > 0 for r in ep_returns])))

    return float(np.mean(all_returns)), float(np.mean(all_solved))

def eval_expert_on_levels(expert, test_levels, n_eps=10):
    """Get expert reference return on test levels."""
    ret, solved = evaluate_on_levels(expert, test_levels, n_eps)
    return ret, solved

# ══════════════════════════════════════════════════════════════════════
# Single seed run
# ══════════════════════════════════════════════════════════════════════
def run_one_seed(seed_init, expert, expert_ret, expert_solved):

    print(f"\n  {'─'*55}")
    print(f"  Seed {seed_init}")
    print(f"  {'─'*55}")

    # Fresh policies with this seed
    bc_policy   = CNNPolicy(n_actions=N_ACTIONS).to(DEVICE)
    tvbc_policy = CNNPolicy(n_actions=N_ACTIONS).to(DEVICE)

    # Same init weights
    torch.manual_seed(seed_init)
    for p in bc_policy.parameters():
        if p.dim() > 1:
            nn.init.orthogonal_(p)
    tvbc_policy.load_state_dict(bc_policy.state_dict())

    bc_optimiser = optim.Adam(bc_policy.parameters(), lr=ETA)
    tvbc_rng     = np.random.default_rng(seed_init)

    bc_pool   = []; bc_tv   = []
    tvbc_pool = []; tvbc_tv = []

    bc_returns    = []; tvbc_returns  = []
    bc_solved     = []; tvbc_solved   = []
    traj_seed     = seed_init * 10000

    for rnd in range(N_ROUNDS):

        level = TRAIN_LEVELS[rnd]

        # Collect 5 demonstrations on this maze level
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            t = collect_trajectory(
                level, expert, eps=NOISE_EPS, seed=traj_seed)
            new_trajs.append(t)
            traj_seed += 1

        # Compute TV using current theta — freeze immediately
        bc_new_tv   = compute_tv_batch(new_trajs, bc_policy)
        tvbc_new_tv = compute_tv_batch(new_trajs, tvbc_policy)

        bc_pool.extend(new_trajs);   bc_tv.extend(bc_new_tv)
        tvbc_pool.extend(new_trajs); tvbc_tv.extend(tvbc_new_tv)

        # Select — equal weightage to all history
        rng_s = np.random.default_rng(42 + rnd + seed_init * 1000)
        tvbc_sel, _, _ = select_tvbc(
            tvbc_pool, tvbc_tv, beta=BETA_TV, rng=rng_s)
        bc_sel, _ = select_bc(
            bc_pool, n_select=len(tvbc_sel),
            rng=np.random.default_rng(
                42 + rnd + seed_init * 1000 + 500))

        bc_flat   = flatten(bc_sel)
        tvbc_flat = flatten(tvbc_sel)

        # Train TRAIN_STEPS steps
        rng_t = np.random.default_rng(
            42 + rnd + seed_init * 1000 + 2000)
        for _ in range(TRAIN_STEPS):
            bs = min(BATCH_SIZE, len(bc_flat))
            ts = min(BATCH_SIZE, len(tvbc_flat))
            if bs == 0 or ts == 0:
                break
            bi = rng_t.choice(len(bc_flat),   size=bs, replace=False)
            ti = rng_t.choice(len(tvbc_flat), size=ts, replace=False)
            bc_batch   = [bc_flat[i]   for i in bi]
            tvbc_batch = [tvbc_flat[i] for i in ti]
            bc_step(bc_policy, bc_optimiser, bc_batch)
            tvbc_step(tvbc_policy, tvbc_batch,
                      beta=BETA_TV, eta=ETA, rng=tvbc_rng)

        # Evaluate on held-out test levels
        bc_ret,   bc_sol   = evaluate_on_levels(
            bc_policy,   TEST_LEVELS)
        tvbc_ret, tvbc_sol = evaluate_on_levels(
            tvbc_policy, TEST_LEVELS)

        bc_returns.append(bc_ret);   tvbc_returns.append(tvbc_ret)
        bc_solved.append(bc_sol);    tvbc_solved.append(tvbc_sol)

        if (rnd + 1) % 10 == 0:
            print(f"    Round {rnd+1:>3} | "
                  f"BC={bc_ret:.3f} ({bc_sol*100:.1f}%) | "
                  f"TV={tvbc_ret:.3f} ({tvbc_sol*100:.1f}%)")

    # Save final policies from last seed
    torch.save(bc_policy.state_dict(),
               'results/data/bc_policy_maze.pt')
    torch.save(tvbc_policy.state_dict(),
               'results/data/tvbc_policy_maze.pt')
    print(f'  Policies saved for seed {seed_init}')
    return bc_returns, tvbc_returns, bc_solved, tvbc_solved

# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def run():
    print(f"{'='*60}")
    print(f"  Procgen Maze — TV-BC vs BC Generalisation")
    print(f"{'='*60}")
    print(f"  Training levels: {TRAIN_LEVELS[0]}–{TRAIN_LEVELS[-1]}")
    print(f"  Test levels:     {TEST_LEVELS[0]}–{TEST_LEVELS[-1]}")
    print(f"  N_ROUNDS:        {N_ROUNDS}")
    print(f"  Seeds:           {SEEDS}")
    print(f"  Noise:           {NOISE_EPS}")
    print(f"  Beta:            {BETA_TV}")
    print(f"  Device:          {DEVICE}")

    # Load expert
    print(f"\n  Loading expert from {EXPERT_PATH}...")
    expert = load_expert(EXPERT_PATH)
    print(f"  Expert params: {expert.count_parameters():,}")

    # Expert reference on test levels
    print(f"  Evaluating expert on test levels...")
    expert_ret, expert_solved = eval_expert_on_levels(
        expert, TEST_LEVELS, n_eps=20)
    print(f"  Expert test return: {expert_ret:.3f}  "
          f"solve rate: {expert_solved*100:.1f}%")

    # Run multiple seeds
    all_bc_ret    = []
    all_tvbc_ret  = []
    all_bc_sol    = []
    all_tvbc_sol  = []

    for seed in SEEDS:
        bc_r, tvbc_r, bc_s, tvbc_s = run_one_seed(
            seed, expert, expert_ret, expert_solved)
        all_bc_ret.append(bc_r)
        all_tvbc_ret.append(tvbc_r)
        all_bc_sol.append(bc_s)
        all_tvbc_sol.append(tvbc_s)

    # Average across seeds
    bc_mean    = np.mean(all_bc_ret,   axis=0)
    tvbc_mean  = np.mean(all_tvbc_ret, axis=0)
    bc_std     = np.std(all_bc_ret,    axis=0)
    tvbc_std   = np.std(all_tvbc_ret,  axis=0)
    bc_s_mean  = np.mean(all_bc_sol,   axis=0)
    tvbc_s_mean= np.mean(all_tvbc_sol, axis=0)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS (avg {len(SEEDS)} seeds x {len(TEST_LEVELS)} test levels)")
    print(f"{'='*60}")
    print(f"  Expert return:  {expert_ret:.3f}  "
          f"solve={expert_solved*100:.1f}%")
    print(f"  BC   return:    {bc_mean[-1]:.3f} "
          f"(+/-{bc_std[-1]:.3f})  "
          f"solve={bc_s_mean[-1]*100:.1f}%")
    print(f"  TV-BC return:   {tvbc_mean[-1]:.3f} "
          f"(+/-{tvbc_std[-1]:.3f})  "
          f"solve={tvbc_s_mean[-1]*100:.1f}%")
    print(f"  Diff (TV-BC-BC):{tvbc_mean[-1]-bc_mean[-1]:+.3f}")
    winner = "TV-BC" if tvbc_mean[-1] > bc_mean[-1] else "BC"
    print(f"  Winner:         {winner}")

    # Save
    results = {
        "bc_mean": bc_mean.tolist(), "tvbc_mean": tvbc_mean.tolist(),
        "bc_std": bc_std.tolist(),   "tvbc_std": tvbc_std.tolist(),
        "bc_s_mean": bc_s_mean.tolist(),
        "tvbc_s_mean": tvbc_s_mean.tolist(),
        "expert_ret": expert_ret, "expert_solved": expert_solved,
        "SEEDS": SEEDS, "N_ROUNDS": N_ROUNDS,
        "TRAIN_LEVELS": TRAIN_LEVELS, "TEST_LEVELS": TEST_LEVELS,
        "BETA_TV": BETA_TV, "ETA": ETA,
    }
    with open("results/data/maze_generalisation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/maze_generalisation_results.pkl")

    # Plot
    rounds = np.arange(1, N_ROUNDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(rounds, bc_mean,   color="#e07b39", lw=2,
                 label="BC",    marker="o", ms=3)
    axes[0].plot(rounds, tvbc_mean, color="#3a7ebf", lw=2,
                 label="TV-BC", marker="s", ms=3)
    axes[0].fill_between(rounds,
                          bc_mean - bc_std,
                          bc_mean + bc_std,
                          color="#e07b39", alpha=0.2)
    axes[0].fill_between(rounds,
                          tvbc_mean - tvbc_std,
                          tvbc_mean + tvbc_std,
                          color="#3a7ebf", alpha=0.2)
    axes[0].axhline(y=expert_ret, color="green",
                    linestyle="--", lw=1.5,
                    label=f"Expert ({expert_ret:.2f})")
    axes[0].set_xlabel("Training Round", fontsize=11)
    axes[0].set_ylabel("Avg Return — 20 unseen mazes", fontsize=11)
    axes[0].set_title("Procgen Maze Generalisation\n"
                      f"(mean ± std, {len(SEEDS)} seeds)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, [r*100 for r in bc_s_mean],
                 color="#e07b39", lw=2, label="BC",    marker="o", ms=3)
    axes[1].plot(rounds, [r*100 for r in tvbc_s_mean],
                 color="#3a7ebf", lw=2, label="TV-BC", marker="s", ms=3)
    axes[1].axhline(y=expert_solved*100, color="green",
                    linestyle="--", lw=1.5,
                    label=f"Expert ({expert_solved*100:.1f}%)")
    axes[1].set_xlabel("Training Round", fontsize=11)
    axes[1].set_ylabel("Maze Solve Rate (%) — 20 unseen mazes",
                       fontsize=11)
    axes[1].set_title("Maze Solve Rate — Unseen Levels", fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Procgen Maze | {N_ROUNDS} train levels | "
        f"20 test levels | noise={NOISE_EPS} | "
        f"beta={BETA_TV} | {len(SEEDS)} seeds",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("results/plots/maze_generalisation_curves.png", dpi=150)
    plt.close()



    print(f'  Plot: results/plots/maze_generalisation_curves.png')
    print(f"  Plot: results/plots/maze_generalisation_curves.png")


if __name__ == "__main__":
    run()
