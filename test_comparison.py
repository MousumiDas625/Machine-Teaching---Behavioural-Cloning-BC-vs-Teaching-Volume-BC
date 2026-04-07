# test_comparison.py
# Fair comparison: BC vs TV-BC using identical settings
# Both use SGD with eta=0.01, same initial weights, same training data

import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import torch
from agents.policy_net       import PolicyNetwork
from agents.rl_teacher       import RLTeacher
from gridworld_env.gridworld import GridWorld
from learners.standard_bc    import StandardBC
from learners.tv_bc          import TeacherAwareBC

# ----------------------------------------------------------
# Load saved data
# ----------------------------------------------------------
with open("data/rollouts/teacher.pkl",       "rb") as f:
    teacher = pickle.load(f)
with open("data/rollouts/omega_star.pkl",    "rb") as f:
    omega_star = pickle.load(f)
with open("data/selected/train_states.pkl",  "rb") as f:
    states = pickle.load(f)
with open("data/selected/train_actions.pkl", "rb") as f:
    actions = pickle.load(f)

env = GridWorld(grid_size=8, omega_star=omega_star,
                max_steps=50, seed=42)

# ----------------------------------------------------------
# Shared evaluation function
# ----------------------------------------------------------
def evaluate_policy(policy, env, n_episodes=30, seed=99):
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

# ----------------------------------------------------------
# Shared hyperparameters
# Both learners MUST use the same eta and same initial weights
# ----------------------------------------------------------
ETA        = 0.01    # learning rate — same for both
BETA       = 2.0     # TV temperature — only used by TV-BC
N_EPOCHS   = 80      # enough to see convergence
BATCH_SIZE = 20
SEED_INIT  = 0       # same random init for both networks
SEED_TRAIN = 7       # same mini-batch shuffling for both

print("=" * 60)
print("  Fair Comparison: Standard BC vs TV-BC (ITAL)")
print("=" * 60)
print(f"  eta={ETA}, beta={BETA}, epochs={N_EPOCHS}, batch={BATCH_SIZE}")
print(f"  Training samples: {len(states)}")
print(f"  Teacher return (reference): ~47.9")

# Check initial performance
init_policy = PolicyNetwork(state_dim=64, action_dim=4,
                             hidden_dim=64, seed=SEED_INIT)
init_return = evaluate_policy(init_policy, env)
print(f"  Initial return (random weights): {init_return:.3f}")
print()

# ----------------------------------------------------------
# Train Standard BC
# ----------------------------------------------------------
print("-" * 60)
print("Training Standard BC...")
print("-" * 60)

bc_policy = PolicyNetwork(state_dim=64, action_dim=4,
                           hidden_dim=64, seed=SEED_INIT)
bc_eval_fn = lambda p: evaluate_policy(p, env, n_episodes=30)

bc_learner = StandardBC(policy=bc_policy, eta=ETA, seed=SEED_TRAIN)
bc_learner.train(
    states     = states,
    actions    = actions,
    n_epochs   = N_EPOCHS,
    batch_size = BATCH_SIZE,
    eval_fn    = bc_eval_fn,
    eval_every = 10,
    verbose    = True
)

# ----------------------------------------------------------
# Train TV-BC — SAME initial weights, SAME data
# ----------------------------------------------------------
print()
print("-" * 60)
print("Training TV-BC (ITAL)...")
print("-" * 60)

tvbc_policy = PolicyNetwork(state_dim=64, action_dim=4,
                             hidden_dim=64, seed=SEED_INIT)
tvbc_eval_fn = lambda p: evaluate_policy(p, env, n_episodes=30)

tvbc_learner = TeacherAwareBC(
    policy  = tvbc_policy,
    teacher = teacher,
    eta     = ETA,
    beta    = BETA,
    seed    = SEED_TRAIN
)
tvbc_learner.train(
    states     = states,
    actions    = actions,
    n_epochs   = N_EPOCHS,
    batch_size = BATCH_SIZE,
    eval_fn    = tvbc_eval_fn,
    eval_every = 10,
    verbose    = True
)

# ----------------------------------------------------------
# Final evaluation — more episodes for reliable estimate
# ----------------------------------------------------------
print()
print("=" * 60)
print("  FINAL RESULTS")
print("=" * 60)

bc_final   = evaluate_policy(bc_policy,   env, n_episodes=100, seed=0)
tvbc_final = evaluate_policy(tvbc_policy, env, n_episodes=100, seed=0)
teacher_return = 47.9

print(f"  Teacher (reference):  {teacher_return:.3f}")
print(f"  Standard BC:          {bc_final:.3f}  "
      f"({100*bc_final/teacher_return:.1f}% of teacher)")
print(f"  TV-BC (ITAL):         {tvbc_final:.3f}  "
      f"({100*tvbc_final/teacher_return:.1f}% of teacher)")
print(f"  Difference (TV-BC - BC): {tvbc_final - bc_final:+.3f}")

winner = "TV-BC" if tvbc_final > bc_final else "Standard BC"
print(f"\n  Winner: {winner}")

# ----------------------------------------------------------
# Policy behaviour at key cells
# ----------------------------------------------------------
print()
print("  Action probs at cell 5 (best cell, optimal=UP):")
print(f"  {'':8s}  {'UP':>6}  {'DOWN':>6}  {'LEFT':>6}  {'RIGHT':>6}")
for label, policy in [("BC", bc_policy), ("TV-BC", tvbc_policy)]:
    s = np.zeros(64, dtype=np.float32); s[5] = 1.0
    p = policy.get_action_probs(s).numpy()
    print(f"  {label:8s}  {p[0]:6.4f}  {p[1]:6.4f}  {p[2]:6.4f}  {p[3]:6.4f}")

# ----------------------------------------------------------
# Loss curves summary
# ----------------------------------------------------------
print()
print("  Loss history (every 10 epochs):")
print(f"  {'Epoch':>6}  {'BC loss':>10}  {'TV-BC loss':>10}")
for i in range(0, N_EPOCHS, 10):
    bc_l   = bc_learner.loss_history[i]
    tvbc_l = tvbc_learner.loss_history[i]
    print(f"  {i+1:>6}  {bc_l:>10.4f}  {tvbc_l:>10.4f}")
