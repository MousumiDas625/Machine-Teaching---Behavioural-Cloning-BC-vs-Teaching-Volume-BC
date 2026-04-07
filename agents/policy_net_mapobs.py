# agents/policy_net_mapobs.py
#
# Policy network for 128-dimensional map observation input.
#
# Key difference from policy_net.py:
#   - Input dimension is 128 (map encoding 64 + position 64)
#     instead of 64 (position only)
#   - Larger hidden layers to handle richer input
#   - Architecture: 128 -> 128 -> 64 -> 4
#   - All methods identical to policy_net.py so existing
#     learners (standard_bc.py, tv_bc.py) work unchanged
#
# Why larger network:
#   With 128-dim input the network needs more capacity to learn
#   the relationship between map layout and optimal actions.
#   The first layer must learn to separately process the map
#   encoding and the position encoding before combining them.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PolicyNetworkMapObs(nn.Module):
    """
    3-layer MLP policy network for 128-dim map observation input.

    Input:  128-dim vector = [map_encoding(64) | position(64)]
    Output: probability distribution over 4 actions

    Architecture:
      Linear(128 -> 128) + ReLU
      Linear(128 ->  64) + ReLU
      Linear( 64 ->   4) + Softmax

    Total parameters:
      fc1: 128*128 + 128 = 16,512
      fc2: 128*64  +  64 =  8,256
      fc3:  64*4   +   4 =    260
      Total: 25,028
    """

    def __init__(self,
                 obs_dim   : int = 128,
                 action_dim: int = 4,
                 hidden1   : int = 128,
                 hidden2   : int = 64,
                 seed      : int = 0):
        """
        Parameters
        ----------
        obs_dim    : input dimension (128 for map + position)
        action_dim : number of actions (4)
        hidden1    : first hidden layer size (128)
        hidden2    : second hidden layer size (64)
        seed       : random seed for weight initialisation
                     must be same for BC and TV-BC for fair comparison
        """
        super(PolicyNetworkMapObs, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Network layers
        self.fc1 = nn.Linear(obs_dim,  hidden1)
        self.fc2 = nn.Linear(hidden1,  hidden2)
        self.fc3 = nn.Linear(hidden2,  action_dim)

        # Xavier uniform initialisation — better than default
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    # ------------------------------------------------------------------ #
    #  Forward pass                                                        #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor shape (..., 128)

        Returns
        -------
        probs : torch.Tensor shape (..., 4) — action probabilities
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)   # add batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

    # ------------------------------------------------------------------ #
    #  Action probability interface                                        #
    # ------------------------------------------------------------------ #

    def get_action_probs(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Get action probabilities for a single observation.

        Parameters
        ----------
        obs_np : np.ndarray shape (128,) — full map observation
                 from env.get_full_obs(cell)

        Returns
        -------
        probs : torch.Tensor shape (4,) — action probabilities
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        with torch.no_grad():
            probs = self.forward(obs_t)
        return probs.squeeze(0)

    # ------------------------------------------------------------------ #
    #  Loss computation (with gradient tracking)                          #
    # ------------------------------------------------------------------ #

    def compute_loss(self, obs_np: np.ndarray,
                     action: int) -> torch.Tensor:
        """
        Compute cross-entropy loss for one (observation, action) pair.

        loss = -log pi_theta(action | obs)

        Gradient tracking is ENABLED so this can be used in:
          - BC gradient update
          - TV gradient norm computation
          - ITAL correction computation

        Parameters
        ----------
        obs_np : np.ndarray shape (128,)
        action : int in {0, 1, 2, 3}

        Returns
        -------
        loss : torch.Tensor scalar (with grad)
        """
        obs_t  = torch.tensor(obs_np, dtype=torch.float32)
        probs  = self.forward(obs_t).squeeze(0)
        loss   = -torch.log(probs[action] + 1e-12)
        return loss

    # ------------------------------------------------------------------ #
    #  Gradient norm squared (for TV formula Term 1)                      #
    # ------------------------------------------------------------------ #

    def compute_grad_norm_sq(self, obs_np: np.ndarray,
                              action: int) -> float:
        """
        Compute ||nabla_theta loss(theta; obs, action)||^2

        This is Term 1 of the Teaching Volume formula.
        Calls loss.backward() and collects all parameter gradients
        into a flat vector, then returns the squared L2 norm.

        Parameters
        ----------
        obs_np : np.ndarray shape (128,)
        action : int

        Returns
        -------
        grad_norm_sq : float
        """
        self.zero_grad()
        loss = self.compute_loss(obs_np, action)
        loss.backward()

        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(p.numel()))

        flat_grad    = torch.cat(grads)
        grad_norm_sq = float(flat_grad.dot(flat_grad).item())
        return grad_norm_sq

    # ------------------------------------------------------------------ #
    #  Action sampling                                                     #
    # ------------------------------------------------------------------ #

    def act(self, obs_np: np.ndarray,
            rng: np.random.Generator = None,
            greedy: bool = False) -> int:
        """
        Sample or select an action given a full map observation.

        Parameters
        ----------
        obs_np  : np.ndarray shape (128,)
        rng     : numpy random generator (for stochastic sampling)
        greedy  : if True, return argmax action (no sampling)

        Returns
        -------
        action : int in {0, 1, 2, 3}
        """
        probs = self.get_action_probs(obs_np).numpy()
        if greedy:
            return int(np.argmax(probs))
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(4, p=probs))

    # ------------------------------------------------------------------ #
    #  Deep copy (needed for ITAL theta_hat computation)                  #
    # ------------------------------------------------------------------ #

    def clone(self) -> 'PolicyNetworkMapObs':
        """
        Return a deep copy of this network with identical weights.

        Used in tv_bc.py to compute theta_hat without modifying
        the original network's parameters.

        Returns
        -------
        cloned_net : PolicyNetworkMapObs with same weights
        """
        cloned = PolicyNetworkMapObs(
            obs_dim    = self.obs_dim,
            action_dim = self.action_dim,
            hidden1    = self.fc1.out_features,
            hidden2    = self.fc2.out_features,
            seed       = 0
        )
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
