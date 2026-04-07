# experiments/run_iterative.py
#
# True ITAL experiment — iterative training loop.
#
# Unlike run_experiment.py (which trained over a fixed dataset),
# this script resamples a fresh mini-batch at every iteration and
# recomputes TV using the CURRENT learner parameters each time.
# This is the correct implementation of Algorithm 1 from the paper.
#
# What this script does:
# ──────────────────────
# 1. Load teacher and all 300 trajectories (clean + noisy pool)
# 2. Build BC learner and TV-BC learner with identical initial weights
# 3. Run T=2000 iterations for each learner:
#       - At every iteration: sample fresh batch of 20 (s,a) pairs
#       - BC:    pick random pair from batch, plain SGD step
#       - TV-BC: compute TV with current θ, ITAL update
# 4. Every EVAL_EVERY=5 iterations: evaluate policy return
# 5. Save learning curves and plot them

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

# ──────────────────────────────────────────────────────────────────────
# Configurable hyperparameters — change these to experiment
# ──────────────────────────────────────────────────────────────────────
ETA          = 0.01    # learning rate (same for both learners)
BETA         = 2.0     # Boltzmann temperature for TV-BC teacher dist.
BATCH_SIZE   = 20      # mini-batch size per iteration
T_ITERATIONS = 16000    # total number of update iterations
EVAL_EVERY   = 100       # evaluate policy return every this many iterations
N_EVAL_EPS   = 30      # episodes per evaluation rollout
SEED_INIT    = 0       # network initialisation seed (same for both)
SEED_BC      = 1       # random seed for BC sampling
SEED_TVBC    = 2       # random seed for TV-BC sampling
TEACHER_RETURN = 47.9  # reference (from Value Iteration teacher)

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/data",  exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Step 1: Load saved data
# ──────────────────────────────────────────────────────────────────────
print("Loading data...")

with open("data/rollouts/teacher.pkl",    "rb") as f:
    teacher = pickle.load(f)
with open("data/rollouts/omega_star.pkl", "rb") as f:
    omega_star = pickle.load(f)

# Load ALL trajectories — both clean and noisy
# In the iterative setting the full pool is the sampling source.
# TV scoring at each iteration decides which examples are useful NOW.
with open("data/rollouts/all_trajs.pkl",  "rb") as f:
    all_trajs = pickle.load(f)

env = GridWorld(grid_size=8, omega_star=omega_star,
                max_steps=50, seed=42)

# Flatten all trajectories into a single pool of (state, action) pairs
pool_states  = []
pool_actions = []
for traj in all_trajs:
    for (state_np, action) in traj:
        pool_states.append(state_np)
        pool_actions.append(action)

pool_states  = np.array(pool_states,  dtype=np.float32)  # (N, 64)
pool_actions = np.array(pool_actions, dtype=np.int64)     # (N,)
N_POOL       = len(pool_states)

print(f"  Pool size: {N_POOL} (s,a) pairs from {len(all_trajs)} trajectories")
print(f"  Iterations: {T_ITERATIONS}  |  Eval every: {EVAL_EVERY} steps")
print(f"  Batch size: {BATCH_SIZE}  |  eta={ETA}  beta={BETA}")

# ──────────────────────────────────────────────────────────────────────
# Step 2: Evaluation function
# ──────────────────────────────────────────────────────────────────────

def evaluate_policy(policy, n_episodes=N_EVAL_EPS, seed=99):
    """
    Roll out the policy greedily for n_episodes episodes.
    Returns the average cumulative reward.

    'Greedy' means: always pick the highest-probability action.
    No exploration during evaluation.
    """
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    policy.eval()

    for _ in range(n_episodes):
        state     = env.reset(start_pos=int(rng.integers(0, env.n_states)))
        done      = False
        ep_reward = 0.0

        while not done:
            probs  = policy.get_action_probs(state).numpy()
            action = int(np.argmax(probs))
            state, reward, done, _ = env.step(action)
            ep_reward += reward

        total_reward += ep_reward

    return total_reward / n_episodes

# ──────────────────────────────────────────────────────────────────────
# Step 3: Batch sampling helper
# ──────────────────────────────────────────────────────────────────────

def sample_batch(rng, batch_size=BATCH_SIZE):
    """
    Sample a fresh random mini-batch from the full pool.

    Returns a list of (state_np, action_int) tuples.
    Called at every single iteration for both learners.

    Using the same rng for both BC and TV-BC within one iteration
    means they see identical batches — fair comparison.
    """
    idx   = rng.choice(N_POOL, size=batch_size, replace=False)
    batch = [(pool_states[i], int(pool_actions[i])) for i in idx]
    return batch

# ──────────────────────────────────────────────────────────────────────
# Step 4: Initialise both learners with identical weights
# ──────────────────────────────────────────────────────────────────────

# Both policies start from EXACTLY the same random initialisation.
# This is critical — any performance difference must come from the
# update rule, not from lucky/unlucky weight initialisation.
bc_policy   = PolicyNetwork(state_dim=64, action_dim=4,
                             hidden_dim=64, seed=SEED_INIT)
tvbc_policy = PolicyNetwork(state_dim=64, action_dim=4,
                             hidden_dim=64, seed=SEED_INIT)

bc_learner   = StandardBC(policy=bc_policy,   eta=ETA, seed=SEED_BC)
tvbc_learner = TeacherAwareBC(policy=tvbc_policy, teacher=teacher,
                               eta=ETA, beta=BETA, seed=SEED_TVBC)

# Shared batch rng — both learners see the same mini-batch each iteration
batch_rng = np.random.default_rng(42)

print(f"\n  Initial BC   return: {evaluate_policy(bc_policy):.3f}")
print(f"  Initial TV-BC return: {evaluate_policy(tvbc_policy):.3f}")
print(f"  Teacher return (ref): {TEACHER_RETURN}")

# ──────────────────────────────────────────────────────────────────────
# Step 5: Iterative training loop
# ──────────────────────────────────────────────────────────────────────
#
# At EVERY iteration:
#   1. Sample one fresh batch (shared between BC and TV-BC)
#   2. BC step:    random pair from batch → plain SGD
#   3. TV-BC step: compute TV with current θ → ITAL update
#   4. Every EVAL_EVERY steps → evaluate both policies
#
# This is the key difference from the batch experiment:
# TV is recomputed with the CURRENT learner parameters at step 3,
# so the teacher's selection adapts as the learner improves.
# ──────────────────────────────────────────────────────────────────────

bc_returns   = []   # eval return at each checkpoint
tvbc_returns = []
eval_iters   = []   # iteration index of each checkpoint

bc_losses   = []    # loss at every iteration
tvbc_losses = []

print(f"\n{'='*60}")
print(f"  Running iterative experiment ({T_ITERATIONS} iterations)")
print(f"{'='*60}")

for t in range(1, T_ITERATIONS + 1):

    # ── Sample fresh batch (same for both learners this iteration) ──
    batch = sample_batch(batch_rng)

    # ── BC step ────────────────────────────────────────────────────
    bc_loss = bc_learner.step(batch)
    bc_losses.append(bc_loss)

    # ── TV-BC step (TV recomputed with current tvbc_policy.θ) ──────
    tvbc_loss = tvbc_learner.step(batch)
    tvbc_losses.append(tvbc_loss)

    # ── Evaluate every EVAL_EVERY iterations ───────────────────────
    if t % EVAL_EVERY == 0 or t == 1:
        bc_ret   = evaluate_policy(bc_policy)
        tvbc_ret = evaluate_policy(tvbc_policy)

        bc_returns.append(bc_ret)
        tvbc_returns.append(tvbc_ret)
        eval_iters.append(t)

        print(f"  iter {t:5d}/{T_ITERATIONS}  "
              f"BC loss={bc_loss:.4f}  ret={bc_ret:.3f}  |  "
              f"TV-BC loss={tvbc_loss:.4f}  ret={tvbc_ret:.3f}")

print(f"\n{'='*60}")
print(f"  FINAL RESULTS")
print(f"{'='*60}")
print(f"  Teacher return (reference): {TEACHER_RETURN:.3f}")
print(f"  Standard BC final return:   {bc_returns[-1]:.3f}  "
      f"({100*bc_returns[-1]/TEACHER_RETURN:.1f}% of teacher)")
print(f"  TV-BC final return:         {tvbc_returns[-1]:.3f}  "
      f"({100*tvbc_returns[-1]/TEACHER_RETURN:.1f}% of teacher)")
print(f"  Difference (TV-BC - BC):    {tvbc_returns[-1]-bc_returns[-1]:+.3f}")

# ──────────────────────────────────────────────────────────────────────
# Step 6: Action probs at key cell
# ──────────────────────────────────────────────────────────────────────
s5 = np.zeros(64, dtype=np.float32); s5[5] = 1.0
bc_probs   = bc_policy.get_action_probs(s5).detach().numpy()
tvbc_probs = tvbc_policy.get_action_probs(s5).detach().numpy()
action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

print(f"\n  Action probs at cell 5 (best cell, optimal=UP):")
print(f"  {'':8s}  {'UP':>6}  {'DOWN':>6}  {'LEFT':>6}  {'RIGHT':>6}")
for label, probs in [("BC", bc_probs), ("TV-BC", tvbc_probs)]:
    print(f"  {label:8s}  "
          + "  ".join(f"{p:6.4f}" for p in probs))

# ──────────────────────────────────────────────────────────────────────
# Step 7: Save results
# ──────────────────────────────────────────────────────────────────────
results = {
    "eval_iters"  : eval_iters,
    "bc_returns"  : bc_returns,
    "tvbc_returns": tvbc_returns,
    "bc_losses"   : bc_losses,
    "tvbc_losses" : tvbc_losses,
    "T_ITERATIONS": T_ITERATIONS,
    "EVAL_EVERY"  : EVAL_EVERY,
    "ETA"         : ETA,
    "BETA"        : BETA,
    "BATCH_SIZE"  : BATCH_SIZE,
}
with open("results/data/iterative_results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"\n  Results saved to results/data/iterative_results.pkl")

# ──────────────────────────────────────────────────────────────────────
# Step 8: Plots
# ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Plot 1: Policy return over iterations ──────────────────────────
axes[0].plot(eval_iters, bc_returns,
             label="Standard BC",  color="#e07b39", lw=2)
axes[0].plot(eval_iters, tvbc_returns,
             label="TV-BC (ITAL)", color="#3a7ebf", lw=2)
axes[0].axhline(y=TEACHER_RETURN, color="green",
                linestyle="--", lw=1.5, label=f"Teacher ({TEACHER_RETURN})")
axes[0].set_xlabel("Iteration", fontsize=12)
axes[0].set_ylabel("Average Policy Return", fontsize=12)
axes[0].set_title("Policy Return: BC vs TV-BC (Iterative)", fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# ── Plot 2: Training loss over iterations ──────────────────────────
# Smooth the loss with a rolling window for readability
window = 50
def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')

iters_smooth = np.arange(window, T_ITERATIONS + 1)
axes[1].plot(iters_smooth, smooth(bc_losses,   window),
             label="Standard BC",  color="#e07b39", lw=2)
axes[1].plot(iters_smooth, smooth(tvbc_losses, window),
             label="TV-BC (ITAL)", color="#3a7ebf", lw=2)
axes[1].set_xlabel("Iteration", fontsize=12)
axes[1].set_ylabel("Training Loss (smoothed)", fontsize=12)
axes[1].set_title(f"Training Loss (rolling avg, window={window})", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    f"Iterative ITAL  |  eta={ETA}  beta={BETA}  "
    f"batch={BATCH_SIZE}  iters={T_ITERATIONS}",
    fontsize=11)
plt.tight_layout()
plt.savefig("results/plots/iterative_curves.png", dpi=150)
plt.close()
print(f"  Plot saved to results/plots/iterative_curves.png")

