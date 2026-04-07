# experiments/run_growing_mapobs.py
#
# Growing dataset experiment with:
#   1. Full map observation (128-dim: map encoding + position)
#   2. Map changes every MAP_CHANGE_EVERY rounds
#   3. Old trajectories kept with frozen TV scores (professor's rule)
#
# This simulates a robot learning in a changing environment.
# The map changes represent the environment being reorganised
# while the robot is still learning — goals move, traps appear
# in new locations.
#
# Why this is harder and more realistic than previous experiments:
#   - Robot must adapt to new map layouts mid-training
#   - With map observation, robot CAN adapt because it sees
#     where goals and traps are on the current map
#   - Without map observation (old setup), robot would be blind
#     to the map change and continue using stale memorised actions
#
# Key design decisions:
#   - Map changes every 4 rounds (5 maps over 20 rounds total)
#   - When map changes: teacher is retrained on new map
#   - Old trajectories: kept in pool with frozen 128-dim observations
#     (these observations encode the OLD map — correctly so,
#      because that is what the robot saw when collecting them)
#   - TV scores for new trajectories: computed using current theta
#   - Evaluation: always on the CURRENT map
#
# Comparison:
#   TV-BC: selects trajectories via softmax over TV scores
#          TV scoring uses 128-dim obs so it is map-aware
#   BC:    selects trajectories randomly (same size as TV-BC)

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.rl_teacher              import RLTeacher
from agents.policy_net_mapobs       import PolicyNetworkMapObs
from gridworld_env.gridworld_mapobs import GridWorldMapObs
from learners.standard_bc           import StandardBC
from learners.tv_bc                 import TeacherAwareBC
from teaching.teaching_volume       import compute_tv_single

# ══════════════════════════════════════════════════════════════════════
# Configurable parameters
# ══════════════════════════════════════════════════════════════════════
N_ROUNDS          = 20      # total teaching rounds
TRAJS_PER_ROUND   = 5       # new trajectories added each round
TRAJ_LENGTH       = 50      # max steps per trajectory
NOISE_EPS         = 0.30    # teacher demonstration noise
TRAIN_STEPS       = 200     # SGD steps per round
BATCH_SIZE        = 20      # mini-batch size
K_SELECT          = 0       # 0 = use all accumulated trajectories
BETA_TV           = 2.0     # softmax temperature (lower = more stable)
ETA               = 0.01    # learning rate
GAMMA             = 0.99    # discount for Value Iteration
SOFTMAX_TEMP      = 0.1     # teacher policy temperature
N_EVAL_EPS        = 50      # evaluation episodes per checkpoint
TERMINAL          = False   # True = episodes end at goal
STEP_PENALTY      = 0.0     # reward per empty cell step
MAP_CHANGE_EVERY  = 4       # change map every this many rounds
N_GOALS           = 3       # goals per map (including best goal)
N_TRAPS           = 8       # traps per map
SEED_BASE         = 42      # base seed (maps use 42, 142, 242, ...)
SEED_INIT         = 0       # network init seed (same for BC and TV-BC)

# Observation and network dimensions
OBS_DIM    = 128   # 64 map encoding + 64 position
ACTION_DIM = 4

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Helper: collect one trajectory using full map observation
# ══════════════════════════════════════════════════════════════════════
def collect_trajectory(env, teacher, noisy=True,
                        eps=NOISE_EPS, seed=0):
    """
    Roll out teacher for one episode.
    Each (obs, action) pair uses the 128-dim full map observation.

    The obs includes the CURRENT map encoding at collection time.
    If map changes later, these obs correctly encode the OLD map —
    which is exactly what the robot saw when learning from them.

    Returns
    -------
    list of (obs_128, action_int) tuples
    """
    rng_noise = np.random.default_rng(seed)
    state     = env.reset(start_pos=env.START_CELL)
    traj      = []
    done      = False

    while not done:
        # Full 128-dim observation: map encoding + position
        obs = env.get_full_obs(state)

        if noisy and rng_noise.random() < eps:
            action = int(rng_noise.integers(0, env.n_actions))
        else:
            action = teacher.act_greedy(state)

        traj.append((obs, action))
        state, _, done, _ = env.step(action)

    return traj

# ══════════════════════════════════════════════════════════════════════
# Helper: compute TV scores for new trajectories
# ══════════════════════════════════════════════════════════════════════
def compute_tv_for_trajectories(trajs, policy, teacher, eta=ETA):
    """
    Compute one TV score per trajectory using CURRENT policy weights.

    TV formula:
      TV(obs, a | theta) = -eta^2 * ||grad_theta loss||^2
                         + 2*eta * [loss(theta; obs, a)
                                    - loss(omega*; obs, a)]

    Note: teacher.loss_at() uses state_idx (cell index), not full obs.
    We extract the cell index from the position part of obs (last 64).

    Parameters
    ----------
    trajs   : list of trajectories, each = list of (obs_128, action)
    policy  : PolicyNetworkMapObs — current learner weights
    teacher : RLTeacher           — provides reference loss
    eta     : float

    Returns
    -------
    tv_scores : list of float, one per trajectory
    """
    tv_scores = []

    for traj in trajs:
        if len(traj) == 0:
            tv_scores.append(0.0)
            continue

        step_tvs = []
        for (obs_128, action) in traj:
            # Extract cell index from position part of observation
            # obs_128[:64] = map encoding
            # obs_128[64:] = position one-hot
            pos_vec    = obs_128[64:]
            state_idx  = int(np.argmax(pos_vec))

            # Term 1: gradient penalty using full 128-dim obs
            policy.zero_grad()
            loss_t = policy.compute_loss(obs_128, action)
            loss_t.backward()

            grads = []
            for p in policy.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().view(-1))
                else:
                    grads.append(torch.zeros(p.numel()))

            import torch
            flat_grad    = torch.cat(grads)
            grad_norm_sq = float(flat_grad.dot(flat_grad).item())
            term1        = -(eta ** 2) * grad_norm_sq

            # Term 2: loss gap
            learner_loss = loss_t.item()
            teacher_loss = teacher.loss_at(state_idx, action)
            term2        = 2.0 * eta * (learner_loss - teacher_loss)

            step_tvs.append(term1 + term2)

        tv_scores.append(float(np.mean(step_tvs)))

    return tv_scores

# ══════════════════════════════════════════════════════════════════════
# Helper: softmax selection
# ══════════════════════════════════════════════════════════════════════
def softmax_select(all_trajs, all_tv_scores, K, beta, rng):
    """
    Select K trajectories weighted by softmax over TV scores.
    K=0 means use all trajectories.
    """
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
# Helper: flatten trajectories to (obs list, action list)
# ══════════════════════════════════════════════════════════════════════
def flatten(trajs):
    obs_list = []
    act_list = []
    for traj in trajs:
        for (obs, a) in traj:
            obs_list.append(obs)
            act_list.append(a)
    return obs_list, act_list

# ══════════════════════════════════════════════════════════════════════
# Helper: evaluate policy on current map
# ══════════════════════════════════════════════════════════════════════
def evaluate_policy(policy, env, n_episodes=N_EVAL_EPS, seed=99):
    """
    Evaluate policy on the CURRENT map using full 128-dim observation.
    Returns average return and goal rate.
    """
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    goals_reached = 0
    reached_this_ep = False
    policy.eval()

    for ep in range(n_episodes):
        start = env.START_CELL if ep < n_episodes // 2 \
                else int(rng.integers(0, env.n_states))
        state     = env.reset(start_pos=start)
        done      = False
        ep_reward = 0.0

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

    avg_return = total_reward  / n_episodes
    goal_rate  = goals_reached / n_episodes
    return avg_return, goal_rate

# ══════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════
def run_experiment():

    print(f"{'═'*65}")
    print(f"  Growing Dataset — Map Observation — Changing Maps")
    print(f"{'═'*65}")
    print(f"  Rounds:          {N_ROUNDS}")
    print(f"  Trajs/round:     {TRAJS_PER_ROUND}")
    print(f"  Map changes:     every {MAP_CHANGE_EVERY} rounds")
    print(f"  Obs dim:         {OBS_DIM} (map {OBS_DIM//2} + pos {OBS_DIM//2})")
    print(f"  Train steps:     {TRAIN_STEPS}")
    print(f"  Beta TV:         {BETA_TV}")
    print(f"  ETA:             {ETA}")
    print(f"  Terminal:        {TERMINAL}")
    print(f"  Noise:           {NOISE_EPS}")

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

    # ── Evaluate teacher on initial map ───────────────────────────────
    teacher_ret = _eval_teacher(teacher, env)
    print(f"  Teacher return (map 1): {teacher_ret:.3f}\n")

    # ── Initialise both learners with IDENTICAL weights ────────────────
    # Both use 128-dim input
    # Same seed = same initial weights = fair comparison
    bc_policy   = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)
    tvbc_policy = PolicyNetworkMapObs(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, seed=SEED_INIT)

    print(f"  Policy network parameters: "
          f"{bc_policy.count_parameters():,}")

    bc_learner   = StandardBC(
        policy=bc_policy,   eta=ETA, seed=SEED_BASE)
    tvbc_learner = TeacherAwareBC(
        policy=tvbc_policy, teacher=teacher,
        eta=ETA, beta=BETA_TV, seed=SEED_BASE)

    # ── Accumulators ───────────────────────────────────────────────────
    # TV scores frozen at collection time — never recomputed
    # Observations in old trajectories encode the map at collection time
    bc_all_trajs   = []
    bc_all_tv      = []
    tvbc_all_trajs = []
    tvbc_all_tv    = []

    # Logging
    bc_returns    = []
    tvbc_returns  = []
    bc_goal_rates = []
    tvbc_goal_rates = []
    map_seeds_log = []   # which map was active each round
    teacher_rets  = []   # teacher return on current map each round
    round_sizes   = []

    traj_seed_ctr  = 0
    current_map_id = 1

    # Print header
    print(f"  {'Rnd':>3} | {'Map':>3} | {'Dataset':>7} | "
          f"{'BC ret':>8} | {'BC goal%':>8} | "
          f"{'TVBC ret':>8} | {'TVBC goal%':>10} | "
          f"{'Teacher':>7}")
    print(f"  {'─'*75}")

    # ══════════════════════════════════════════════════════════════════
    # Round loop
    # ══════════════════════════════════════════════════════════════════
    for round_idx in range(N_ROUNDS):

        # ── Check if map should change ─────────────────────────────────
        # Maps change at rounds 5, 9, 13, 17 (every MAP_CHANGE_EVERY)
        # round_idx is 0-indexed so change when round_idx % 4 == 0
        # and round_idx > 0
        if round_idx > 0 and round_idx % MAP_CHANGE_EVERY == 0:
            current_map_id += 1
            new_seed = SEED_BASE + (current_map_id - 1) * 100

            print(f"\n  ── MAP CHANGE: Map {current_map_id-1} "
                  f"→ Map {current_map_id} "
                  f"(seed={new_seed}) ──")

            # Update environment map
            env.change_map(new_seed)
            env.print_map()

            # Retrain teacher on new map
            teacher = RLTeacher(
                omega_star   = env.omega_star,
                grid_size    = env.grid_size,
                gamma        = GAMMA,
                softmax_temp = SOFTMAX_TEMP
            )
            teacher.train(verbose=True)

            # Update TV-BC learner's teacher reference
            tvbc_learner.teacher = teacher

            teacher_ret = _eval_teacher(teacher, env)
            print(f"  Teacher return (map {current_map_id}): "
                  f"{teacher_ret:.3f}\n")

        map_seeds_log.append(SEED_BASE + (current_map_id - 1) * 100)

        # ── Step 1: Collect new trajectories on CURRENT map ───────────
        # Observations encode the CURRENT map layout
        new_trajs = []
        for _ in range(TRAJS_PER_ROUND):
            traj = collect_trajectory(
                env, teacher,
                noisy = True,
                eps   = NOISE_EPS,
                seed  = traj_seed_ctr
            )
            new_trajs.append(traj)
            traj_seed_ctr += 1

        # ── Step 2: Compute TV for new trajs using CURRENT theta ───────
        # Old trajectories keep their frozen TV scores
        new_bc_tv   = compute_tv_for_trajectories(
            new_trajs, bc_policy,   teacher, eta=ETA)
        new_tvbc_tv = compute_tv_for_trajectories(
            new_trajs, tvbc_policy, teacher, eta=ETA)

        # ── Step 3: Accumulate ─────────────────────────────────────────
        bc_all_trajs.extend(new_trajs)
        bc_all_tv.extend(new_bc_tv)
        tvbc_all_trajs.extend(new_trajs)
        tvbc_all_tv.extend(new_tvbc_tv)

        dataset_size = sum(len(t) for t in bc_all_trajs)
        round_sizes.append(dataset_size)

        # ── Step 4: Select trajectories ────────────────────────────────
        rng_sel = np.random.default_rng(SEED_BASE + round_idx)

        # TV-BC: softmax selection over all accumulated TV scores
        tvbc_selected, _ = softmax_select(
            tvbc_all_trajs, tvbc_all_tv,
            K=K_SELECT, beta=BETA_TV, rng=rng_sel)

        # BC: random selection of same size
        n_select   = len(tvbc_selected)
        rng_bc_sel = np.random.default_rng(
            SEED_BASE + round_idx + 1000)
        bc_idx     = list(rng_bc_sel.choice(
            len(bc_all_trajs),
            size=min(n_select, len(bc_all_trajs)),
            replace=False))
        bc_selected = [bc_all_trajs[i] for i in bc_idx]

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

            batch_bc   = [bc_pool[i]   for i in idx_bc]
            batch_tvbc = [tvbc_pool[i] for i in idx_tvbc]

            bc_learner.step(batch_bc)
            tvbc_learner.step(batch_tvbc)

        # ── Step 6: Evaluate on CURRENT map ────────────────────────────
        bc_ret,   bc_gr   = evaluate_policy(bc_policy,   env)
        tvbc_ret, tvbc_gr = evaluate_policy(tvbc_policy, env)

        # Teacher return on current map
        t_ret = _eval_teacher(teacher, env, n_episodes=30)

        bc_returns.append(bc_ret)
        tvbc_returns.append(tvbc_ret)
        bc_goal_rates.append(bc_gr)
        tvbc_goal_rates.append(tvbc_gr)
        teacher_rets.append(t_ret)

        print(f"  {round_idx+1:>3} | "
              f"{current_map_id:>3} | "
              f"{dataset_size:>7} | "
              f"{bc_ret:>8.3f} | "
              f"{bc_gr*100:>7.1f}% | "
              f"{tvbc_ret:>8.3f} | "
              f"{tvbc_gr*100:>9.1f}% | "
              f"{t_ret:>7.3f}")

    # ══════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print(f"  FINAL RESULTS")
    print(f"{'═'*65}")
    print(f"  BC   final return:  {bc_returns[-1]:.3f}  "
          f"goal_rate={bc_goal_rates[-1]*100:.1f}%")
    print(f"  TV-BC final return: {tvbc_returns[-1]:.3f}  "
          f"goal_rate={tvbc_goal_rates[-1]*100:.1f}%")
    print(f"  Difference (TV-BC - BC): "
          f"{tvbc_returns[-1]-bc_returns[-1]:+.3f}")
    print(f"  Teacher final return: {teacher_rets[-1]:.3f}")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "bc_returns"    : bc_returns,
        "tvbc_returns"  : tvbc_returns,
        "bc_goal_rates" : bc_goal_rates,
        "tvbc_goal_rates": tvbc_goal_rates,
        "teacher_rets"  : teacher_rets,
        "map_seeds_log" : map_seeds_log,
        "round_sizes"   : round_sizes,
        "N_ROUNDS"      : N_ROUNDS,
        "MAP_CHANGE_EVERY": MAP_CHANGE_EVERY,
        "BETA_TV"       : BETA_TV,
        "ETA"           : ETA,
        "OBS_DIM"       : OBS_DIM,
        "TERMINAL"      : TERMINAL,
    }
    with open("results/data/growing_mapobs_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Saved: results/data/growing_mapobs_results.pkl")

    # ── Plot ───────────────────────────────────────────────────────────
    rounds = np.arange(1, N_ROUNDS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Shade map regions
    map_colors = ['#f0f4ff', '#fff0f0', '#f0fff0',
                  '#fffff0', '#fff0ff']
    for map_i in range(5):
        start = map_i * MAP_CHANGE_EVERY + 1
        end   = min((map_i + 1) * MAP_CHANGE_EVERY, N_ROUNDS)
        for ax in axes:
            ax.axvspan(start - 0.5, end + 0.5,
                       alpha=0.3,
                       color=map_colors[map_i % len(map_colors)],
                       label=f'Map {map_i+1}' if map_i < 5 else '')

    # Return curves
    axes[0].plot(rounds, bc_returns,   color="#e07b39",
                 lw=2, label="Standard BC",  marker="o", ms=4)
    axes[0].plot(rounds, tvbc_returns, color="#3a7ebf",
                 lw=2, label="TV-BC (ITAL)", marker="s", ms=4)
    axes[0].plot(rounds, teacher_rets, color="green",
                 lw=1.5, linestyle="--", label="Teacher")
    axes[0].set_xlabel("Teaching Round", fontsize=12)
    axes[0].set_ylabel("Average Return", fontsize=12)
    axes[0].set_title("Policy Return — Changing Maps\n"
                      "(128-dim Map Observation)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Add map change markers
    for change_round in range(MAP_CHANGE_EVERY,
                               N_ROUNDS, MAP_CHANGE_EVERY):
        axes[0].axvline(x=change_round + 0.5,
                        color='red', linestyle=':', lw=1.5,
                        alpha=0.7)
        axes[0].text(change_round + 0.6,
                     axes[0].get_ylim()[0],
                     'map\nchange', fontsize=7,
                     color='red', va='bottom')

    # Goal rate curves
    axes[1].plot(rounds,
                 [r*100 for r in bc_goal_rates],
                 color="#e07b39", lw=2,
                 label="Standard BC",  marker="o", ms=4)
    axes[1].plot(rounds,
                 [r*100 for r in tvbc_goal_rates],
                 color="#3a7ebf", lw=2,
                 label="TV-BC (ITAL)", marker="s", ms=4)
    for change_round in range(MAP_CHANGE_EVERY,
                               N_ROUNDS, MAP_CHANGE_EVERY):
        axes[1].axvline(x=change_round + 0.5,
                        color='red', linestyle=':', lw=1.5,
                        alpha=0.7)
    axes[1].set_xlabel("Teaching Round", fontsize=12)
    axes[1].set_ylabel("Goal Reached (%)", fontsize=12)
    axes[1].set_title("Goal Rate — Changing Maps\n"
                      "(red dotted = map change)", fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Map Observation Experiment  |  "
        f"Map changes every {MAP_CHANGE_EVERY} rounds  |  "
        f"beta={BETA_TV}  eta={ETA}  obs={OBS_DIM}d",
        fontsize=10)
    plt.tight_layout()
    plt.savefig("results/plots/growing_mapobs_curves.png", dpi=150)
    plt.close()
    print(f"  Plot: results/plots/growing_mapobs_curves.png")

# ══════════════════════════════════════════════════════════════════════
# Helper: evaluate teacher (used multiple times above)
# ══════════════════════════════════════════════════════════════════════
def _eval_teacher(teacher, env, n_episodes=N_EVAL_EPS):
    """Roll out teacher greedily and return average return."""
    rng   = np.random.default_rng(99)
    total = 0.0
    for ep in range(n_episodes):
        start = env.START_CELL if ep < n_episodes // 2 \
                else int(rng.integers(0, env.n_states))
        state = env.reset(start_pos=start)
        done  = False
        ep_r  = 0.0
        while not done:
            action = teacher.act_greedy(state)
            state, r, done, _ = env.step(action)
            ep_r += r
        total += ep_r
    return total / n_episodes


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import torch
    run_experiment()
