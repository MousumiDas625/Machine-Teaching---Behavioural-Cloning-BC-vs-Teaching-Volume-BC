# test_teacher.py
import sys
sys.path.insert(0, ".")

import numpy as np
from gridworld_env.gridworld import GridWorld
from agents.rl_teacher import RLTeacher

# Create environment (same seed = same omega_star)
env = GridWorld(grid_size=8, seed=42)

# Create and train teacher
teacher = RLTeacher(
    omega_star   = env.omega_star,
    grid_size    = 8,
    gamma        = 0.99,
    conv_thresh  = 1e-6,
    softmax_temp = 0.1,
    seed         = 42
)
teacher.train(verbose=True)

# Check Q-table shape
print("\nQ_table shape:", teacher.Q_table.shape)   # (64, 4)
print("V* shape:", teacher.V.shape)                 # (64,)

# Check the best cell — should match highest omega_star
best_cell = np.argmax(env.omega_star)
print(f"\nBest cell (highest reward): {best_cell}")
print(f"omega_star at best cell: {env.omega_star[best_cell]:.3f}")
print(f"V* at best cell: {teacher.V[best_cell]:.3f}")
print(f"V* should be higher than omega_star (includes future)")

# Check action probabilities at a state
state_idx = 0   # top-left corner
probs = teacher.get_action_probs(state_idx)
print(f"\nAction probs at cell 0 (top-left):")
print(f"  UP:    {probs[0]:.4f}  (should be near 0 — can't go up)")
print(f"  DOWN:  {probs[1]:.4f}")
print(f"  LEFT:  {probs[2]:.4f}  (should be near 0 — can't go left)")
print(f"  RIGHT: {probs[3]:.4f}")
print(f"  Sum:   {probs.sum():.4f}  (must be 1.0)")

# Check teacher loss
loss_best_action = teacher.loss_at(state_idx, np.argmax(probs))
loss_worst_action = teacher.loss_at(state_idx, np.argmin(probs))
print(f"\nTeacher loss on best action:  {loss_best_action:.4f}  (should be low)")
print(f"Teacher loss on worst action: {loss_worst_action:.4f}  (should be high)")

# Roll out the teacher greedily and measure return
print("\nRolling out greedy teacher for 10 episodes...")
total_reward = 0.0
for ep in range(10):
    state = env.reset(start_pos=int(np.random.randint(0, 64)))
    done  = False
    ep_reward = 0.0
    while not done:
        state_idx = int(np.argmax(state))
        action    = teacher.act_greedy(state_idx)
        state, reward, done, _ = env.step(action)
        ep_reward += reward
    total_reward += ep_reward
print(f"Average return (greedy): {total_reward/10:.3f}")
print("(Should be clearly positive — teacher navigates to good cells)")
