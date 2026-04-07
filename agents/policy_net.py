# agents/policy_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    """
    Small MLP policy for the learner.

    Maps a one-hot state vector to a probability distribution
    over 4 actions:  pi_theta(a | s)

    Architecture:
        Linear(state_dim → hidden_dim) + ReLU
        Linear(hidden_dim → hidden_dim) + ReLU
        Linear(hidden_dim → action_dim)
        Softmax

    Parameters
    ----------
    state_dim  : int, input size (64 for 8x8 grid)
    action_dim : int, output size (4 actions)
    hidden_dim : int, size of hidden layers (64)
    seed       : int, for reproducibility
    """

    def __init__(self,
                 state_dim  : int = 64,
                 action_dim : int = 4,
                 hidden_dim : int = 64,
                 seed       : int = 0):

        # Must call parent __init__ first in any nn.Module
        super(PolicyNetwork, self).__init__()

        # Set seed for reproducible weight initialisation
        torch.manual_seed(seed)

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------------
        # Define the three linear layers
        # nn.Linear(in, out) creates a weight matrix W (out x in)
        # and bias vector b (out,)
        # Output = W @ input + b
        # ----------------------------------------------------------
        self.fc1 = nn.Linear(state_dim,  hidden_dim)   # 64 → 64
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # 64 → 64
        self.fc3 = nn.Linear(hidden_dim, action_dim)   # 64 → 4

        # ----------------------------------------------------------
        # Weight initialisation — Xavier uniform
        # This keeps gradient magnitudes stable at the start of training
        # Without this, gradients can vanish or explode early on
        # ----------------------------------------------------------
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor shape (..., state_dim)
            Can be a single state (64,) or a batch (B, 64)

        Returns
        -------
        probs : torch.Tensor shape (..., action_dim)
                Probability distribution over actions.
                Each row sums to 1.0.
        """
        # Layer 1: Linear + ReLU
        x = F.relu(self.fc1(x))

        # Layer 2: Linear + ReLU
        x = F.relu(self.fc2(x))

        # Layer 3: Linear (no activation yet)
        x = self.fc3(x)

        # Softmax over the last dimension (action dimension)
        # dim=-1 means "apply softmax over the last axis"
        # This converts raw scores (logits) to probabilities
        return F.softmax(x, dim=-1)

    def get_action_probs(self, state_np: np.ndarray) -> torch.Tensor:
        """
        Convenience method: takes a numpy state, returns action probs.

        Parameters
        ----------
        state_np : np.ndarray shape (state_dim,) — one-hot state vector

        Returns
        -------
        probs : torch.Tensor shape (action_dim,)
        """
        # Convert numpy → torch tensor
        # float32 to match network weight dtype
        state_t = torch.tensor(state_np, dtype=torch.float32)

        # No gradient tracking needed just for getting probs
        with torch.no_grad():
            probs = self.forward(state_t)

        return probs

    def compute_loss(self,
                     state_np : np.ndarray,
                     action   : int) -> torch.Tensor:
        """
        Compute BC loss for a single (state, action) pair.

        l(theta; s, a) = -log pi_theta(a | s)

        This is the negative log-likelihood of the expert action.
        The gradient of this loss w.r.t. theta is what we need for TV.

        IMPORTANT: This returns a torch.Tensor WITH gradient tracking.
        Do NOT use torch.no_grad() here — we need .backward() to work.

        Parameters
        ----------
        state_np : np.ndarray shape (64,)
        action   : int in {0,1,2,3}

        Returns
        -------
        loss : torch.Tensor scalar — the negative log-likelihood
        """
        state_t  = torch.tensor(state_np, dtype=torch.float32)
        probs    = self.forward(state_t)

        # F.cross_entropy expects logits, not probs — so we use nll_loss
        # nll_loss = -log(prob[action])
        # We take log first, then nll_loss picks the action index
        log_probs = torch.log(probs + 1e-12)   # +1e-12 prevents log(0)
        action_t  = torch.tensor(action, dtype=torch.long)
        loss      = F.nll_loss(log_probs.unsqueeze(0),   # (1, 4)
                               action_t.unsqueeze(0))    # (1,)
        return loss

    def compute_grad_norm_sq(self,
                              state_np: np.ndarray,
                              action  : int) -> float:
        """
        Compute ||∇_theta l(theta; s, a)||²

        This is Term 1 of the Teaching Volume formula.
        It measures how "hard" this example is for the current policy.

        Steps:
          1. Zero out any existing gradients
          2. Compute loss and call .backward()
          3. Collect all parameter gradients into one flat vector
          4. Return the squared L2 norm of that vector

        Parameters
        ----------
        state_np : np.ndarray shape (64,)
        action   : int

        Returns
        -------
        grad_norm_sq : float — ||grad||^2
        """
        # Step 1: clear any gradients from previous calls
        self.zero_grad()

        # Step 2: compute loss WITH gradient tracking
        loss = self.compute_loss(state_np, action)

        # Step 3: backpropagate — fills .grad for every parameter
        loss.backward()

        # Step 4: collect all gradients into one flat vector
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                # .detach() removes from computation graph
                # .cpu() ensures it's on CPU even if model is on GPU
                # .numpy().ravel() flattens to 1D array
                grads.append(param.grad.detach().cpu().numpy().ravel())

        # Concatenate all gradients: [grad_fc1_W, grad_fc1_b,
        #                             grad_fc2_W, grad_fc2_b,
        #                             grad_fc3_W, grad_fc3_b]
        flat_grad = np.concatenate(grads)

        # Squared L2 norm = dot product with itself = sum of squares
        return float(np.dot(flat_grad, flat_grad))

    def act(self, state_np: np.ndarray,
            rng=None, greedy: bool = False) -> int:
        """
        Sample an action from the current policy.

        Parameters
        ----------
        state_np : np.ndarray shape (64,)
        rng      : np.random.Generator or None
        greedy   : if True, return argmax action (for evaluation)

        Returns
        -------
        action : int
        """
        probs = self.get_action_probs(state_np).numpy()

        if greedy:
            return int(np.argmax(probs))

        if rng is not None:
            return int(rng.choice(4, p=probs))
        else:
            return int(np.random.choice(4, p=probs))

    def clone(self) -> "PolicyNetwork":
        """
        Return a deep copy of this network with the same weights.
        Used in the ITAL update to create theta_hat without
        modifying the original network's parameters.
        """
        new_net = PolicyNetwork(
            state_dim  = self.state_dim,
            action_dim = self.action_dim,
            hidden_dim = self.hidden_dim,
        )
        # Copy all parameters from self into new_net
        new_net.load_state_dict(self.state_dict())
        return new_net
