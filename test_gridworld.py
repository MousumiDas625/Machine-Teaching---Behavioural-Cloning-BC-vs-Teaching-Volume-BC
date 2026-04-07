# test_gridworld.py  (run this from project root)
import sys
sys.path.insert(0, ".")   # make sure imports work from root

from gridworld_env.gridworld import GridWorld
import numpy as np

# Create environment
env = GridWorld(grid_size=8, seed=42)

print("n_states :", env.n_states)          # should be 64
print("n_actions:", env.n_actions)         # should be 4
print("omega_star shape:", env.omega_star.shape)   # should be (64,)
print("omega_star range: [{:.3f}, {:.3f}]".format(
    env.omega_star.min(), env.omega_star.max()))    # should be [-1, 1]

# Test reset
state = env.reset(start_pos=0)
print("\nAfter reset at pos 0:")
print("State shape:", state.shape)     # (64,)
print("State sum:", state.sum())       # 1.0  (it's one-hot)
print("Hot index:", np.argmax(state))  # 0

# Test step — move RIGHT from cell 0 should go to cell 1
state, reward, done, info = env.step(1)  # DOWN
print("\nAfter stepping DOWN from cell 0:")
print("New position:", info["pos"])    # should be 8 (one row down)
print("Reward:", reward)               # omega_star[8]
print("Done:", done)                   # False

# Test transition table
print("\nTransition table spot checks:")
print("From cell 0, UP  → stays at:", env.T[0, 0])   # 0  (can't go above row 0)
print("From cell 0, RIGHT →", env.T[0, 3])             # 1
print("From cell 7, RIGHT → stays at:", env.T[7, 3])  # 7  (can't go past col 7)

# Test render
env.reset(start_pos=9)
print("\nGrid render (agent at cell 9 = row 1, col 1):")
print(env.render())

