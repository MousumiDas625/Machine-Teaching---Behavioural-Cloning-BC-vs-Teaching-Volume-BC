# agents/policy_net_mapobs.py
# Policy network for map observation input.
# Supports both 128-dim (8x8 grid) and 512-dim (16x16 grid).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PolicyNetworkMapObs(nn.Module):
    """
    MLP policy network for full map observation input.

    8x8  grid: obs_dim=128  (64 map  + 64  position)
    16x16 grid: obs_dim=512  (256 map + 256 position)

    Architecture (auto-scaled):
      obs_dim=128: 128 -> 128 -> 64 -> 4   (25,028 params)
      obs_dim=512: 512 -> 512 -> 256 -> 4  (394,500 params)
    """

    def __init__(self,
                 obs_dim   : int = 128,
                 action_dim: int = 4,
                 hidden1   : int = None,
                 hidden2   : int = None,
                 seed      : int = 0):
        super(PolicyNetworkMapObs, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Auto-scale hidden layers based on obs_dim if not specified
        if hidden1 is None:
            hidden1 = obs_dim          # 128 or 512
        if hidden2 is None:
            hidden2 = obs_dim // 2     # 64  or 256

        self.fc1 = nn.Linear(obs_dim,  hidden1)
        self.fc2 = nn.Linear(hidden1,  hidden2)
        self.fc3 = nn.Linear(hidden2,  action_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

    def get_action_probs(self, obs_np: np.ndarray) -> torch.Tensor:
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        with torch.no_grad():
            probs = self.forward(obs_t)
        return probs.squeeze(0)

    def compute_loss(self, obs_np: np.ndarray,
                     action: int) -> torch.Tensor:
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        probs = self.forward(obs_t).squeeze(0)
        return -torch.log(probs[action] + 1e-12)

    def compute_grad_norm_sq(self, obs_np: np.ndarray,
                              action: int) -> float:
        self.zero_grad()
        loss = self.compute_loss(obs_np, action)
        loss.backward()
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(p.numel()))
        flat = torch.cat(grads)
        return float(flat.dot(flat).item())

    def act(self, obs_np: np.ndarray,
            rng=None, greedy: bool = False) -> int:
        probs = self.get_action_probs(obs_np).numpy()
        if greedy:
            return int(np.argmax(probs))
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(self.action_dim, p=probs))

    def clone(self) -> 'PolicyNetworkMapObs':
        cloned = PolicyNetworkMapObs(
            obs_dim    = self.obs_dim,
            action_dim = self.action_dim,
            hidden1    = self.fc1.out_features,
            hidden2    = self.fc2.out_features,
            seed       = 0
        )
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
