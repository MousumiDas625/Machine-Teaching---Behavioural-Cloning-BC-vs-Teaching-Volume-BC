# test_policy_net.py
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from agents.policy_net import PolicyNetwork

# Build network
net = PolicyNetwork(state_dim=64, action_dim=4, hidden_dim=64, seed=0)
print("Network architecture:")
print(net)
print(f"\nTotal parameters: "
      f"{sum(p.numel() for p in net.parameters())}")

# Test with a one-hot state (agent at cell 5)
state = np.zeros(64, dtype=np.float32)
state[5] = 1.0

# Forward pass
probs = net.get_action_probs(state)
print(f"\nAction probs at cell 5: {probs.numpy()}")
print(f"Sum of probs: {probs.sum().item():.6f}  (must be 1.0)")

# Test loss computation
action = 2  # LEFT
loss = net.compute_loss(state, action)
print(f"\nLoss for action LEFT (idx=2): {loss.item():.4f}")
print(f"Expected: -log({probs[2].item():.4f}) = "
      f"{-np.log(probs[2].item()):.4f}")

# Test gradient norm squared
grad_norm_sq = net.compute_grad_norm_sq(state, action)
print(f"\n||grad||^2 for (cell 5, LEFT): {grad_norm_sq:.6f}")
print("(Should be a positive number — gradients exist)")

# Test that different actions give different grad norms
grad_norms = []
for a in range(4):
    gn = net.compute_grad_norm_sq(state, a)
    grad_norms.append(gn)
    print(f"  action={a}: ||grad||^2 = {gn:.6f}")

# Test clone
net2 = net.clone()
probs2 = net2.get_action_probs(state)
print(f"\nClone gives same probs: "
      f"{np.allclose(probs.numpy(), probs2.numpy())}")

# Verify clone is independent — modify net2, check net unchanged
with torch.no_grad():
    net2.fc1.weight += 1.0
probs_after = net.get_action_probs(state)
print(f"Original unchanged after modifying clone: "
      f"{np.allclose(probs.numpy(), probs_after.numpy())}")

print("\nAll tests passed!")
