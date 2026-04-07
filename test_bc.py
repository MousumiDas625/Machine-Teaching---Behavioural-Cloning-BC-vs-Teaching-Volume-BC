# test_bc.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pickle
import torch
from agents.policy_net    import PolicyNetwork
from agents.rl_teacher    import RLTeacher
from gridworld_env.gridworld import GridWorld
from learners.standard_bc import StandardBC


# ----------------------------------------------------------
# Load saved data
# ----------------------------------------------------------
with open("data/rollouts/teacher.pkl",      "rb") as f:
    teacher = pickle.load(f)
with open("data/rollouts/omega_star.pkl",   "rb") as f:
    omega_star = pickle.load(f)
with open("data/selected/train_states.pkl", "rb") as f:
    states = pickle.load(f)
with open("data/selected/train_actions.pkl","rb") as f:
    actions = pickle.load(f)

print(f"Training data: {len(states)} (state, action) pairs")

# Rebuild environment for evaluation rollouts
env = GridWorld(grid_size=8, omega_star=omega_star,
                max_steps=50, seed=42)

# ----------------------------------------------------------
# Evaluation function: roll out policy, measure total reward
# ----------------------------------------------------------
def evaluate_policy(policy, env, n_episodes=20, seed=99):
    """
    Roll out the policy for n_episodes and return average reward.
    This tells us how well the learner can actually navigate the grid.
    """
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    policy.eval()

    for _ in range(n_episodes):
        state = env.reset(start_pos=int(rng.integers(0, env.n_states)))
        done  = False
        ep_reward = 0.0

        while not done:
            # Get action probabilities
            probs  = policy.get_action_probs(state).numpy()
            # Greedy action for evaluation
            action = int(np.argmax(probs))
            state, reward, done, _ = env.step(action)
            ep_reward += reward

        total_reward += ep_reward

    return total_reward / n_episodes


# ----------------------------------------------------------
# Build learner and train
# ----------------------------------------------------------
print("\nBuilding fresh PolicyNetwork (seed=0)...")
policy = PolicyNetwork(state_dim=64, action_dim=4,
                       hidden_dim=64, seed=0)

# Check initial policy return (before any training)
init_return = evaluate_policy(policy, env, n_episodes=20)
print(f"Initial policy return (random weights): {init_return:.3f}")
print(f"Teacher return (for reference): ~47.9")

# Create eval function closure
eval_fn = lambda p: evaluate_policy(p, env, n_episodes=20)

# Train
bc_learner = StandardBC(policy=policy, eta=0.001, seed=7)

print("\nStarting BC training (50 epochs for quick test)...")
bc_learner.train(
    states     = states,
    actions    = actions,
    n_epochs   = 50,
    batch_size = 20,
    eval_fn    = eval_fn,
    eval_every = 10,
    verbose    = True
)

# ----------------------------------------------------------
# Final evaluation
# ----------------------------------------------------------
final_return = evaluate_policy(policy, env, n_episodes=50)
print(f"\nFinal policy return after 50 epochs: {final_return:.3f}")
print(f"Initial return was: {init_return:.3f}")
print(f"Improvement: {final_return - init_return:.3f}")
print(f"\nLoss went from {bc_learner.loss_history[0]:.4f} "
      f"to {bc_learner.loss_history[-1]:.4f}")

# Check action distribution of learned policy
print("\nLearned policy action probs at cell 5 (best cell):")
state_5 = np.zeros(64, dtype=np.float32)
state_5[5] = 1.0
probs = policy.get_action_probs(state_5).numpy()
for i, (name, p) in enumerate(zip(["UP","DOWN","LEFT","RIGHT"], probs)):
    print(f"  {name:5s}: {p:.4f}")
print("  (UP should be highest — optimal action at cell 5)")
