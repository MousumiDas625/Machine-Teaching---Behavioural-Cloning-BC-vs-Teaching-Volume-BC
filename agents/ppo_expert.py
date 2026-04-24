# agents/ppo_expert.py
#
# CNN policy network + PPO trainer for Procgen Maze.
#
# Architecture:
#   Input: 64x64x3 RGB image
#   CNN: 3 conv layers → flatten → 256-dim feature vector
#   MLP: 256 → 128 → n_actions
#
# Device: automatically uses GPU if available, CPU otherwise.
# When you get GPU access, no code changes needed.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import copy

# Auto-detect device — works on both CPU and GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNPolicy(nn.Module):
    """
    CNN policy for processing 64x64x3 Procgen observations.

    Architecture:
      Conv1: 3  → 32 filters, 8x8, stride 4  → 15x15x32
      Conv2: 32 → 64 filters, 4x4, stride 2  → 6x6x64
      Conv3: 64 → 64 filters, 3x3, stride 1  → 4x4x64
      Flatten → 1024
      FC1:  1024 → 256  + ReLU
      FC2:  256  → 128  + ReLU
      Actor head:  128 → n_actions (policy logits)
      Critic head: 128 → 1         (state value)

    The actor head gives action probabilities.
    The critic head gives state value estimate (used in PPO).

    Parameters
    ----------
    n_actions : int — number of discrete actions (15 for Procgen)
    """

    def __init__(self, n_actions: int = 15):
        super(CNNPolicy, self).__init__()

        self.n_actions = n_actions

        # CNN feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute CNN output size
        # Input: 64x64x3
        # After conv1(8x8,s4): floor((64-8)/4)+1 = 15 → 15x15x32
        # After conv2(4x4,s2): floor((15-4)/2)+1 = 6  → 6x6x64
        # After conv3(3x3,s1): floor((6-3)/1)+1  = 4  → 4x4x64
        self.cnn_output_size = 4 * 4 * 64   # = 1024

        # Fully connected layers
        self.fc1 = nn.Linear(self.cnn_output_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Actor head: outputs action logits
        self.actor  = nn.Linear(128, n_actions)

        # Critic head: outputs state value
        self.critic = nn.Linear(128, 1)

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation — standard for PPO."""
        for layer in [self.conv1, self.conv2, self.conv3,
                      self.fc1, self.fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Pass observation through CNN + FC layers.

        Parameters
        ----------
        obs : torch.Tensor shape (B, 64, 64, 3) uint8 or float32

        Returns
        -------
        features : torch.Tensor shape (B, 128)
        """
        # Procgen gives uint8 [0,255] images, normalise to [0,1]
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0

        # Procgen format: (B, H, W, C) → PyTorch needs (B, C, H, W)
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        if obs.shape[-1] == 3:
            obs = obs.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)   # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, obs: torch.Tensor):
        """
        Returns action probabilities and state value.

        Returns
        -------
        probs  : torch.Tensor shape (B, n_actions)
        value  : torch.Tensor shape (B, 1)
        """
        features = self._encode(obs)
        logits   = self.actor(features)
        probs    = F.softmax(logits, dim=-1)
        value    = self.critic(features)
        return probs, value

    def get_action_probs(self, obs_np: np.ndarray) -> torch.Tensor:
        """
        Get action probabilities for a single observation.
        Used during demo collection and TV computation.

        Parameters
        ----------
        obs_np : np.ndarray shape (64, 64, 3) uint8

        Returns
        -------
        probs : torch.Tensor shape (15,)
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.forward(obs_t)
        return probs.squeeze(0).cpu()

    def compute_loss(self, obs_np: np.ndarray,
                     action: int) -> torch.Tensor:
        """
        Compute cross-entropy loss: -log π(action|obs)
        WITH gradient tracking. Used in TV computation and BC update.

        Parameters
        ----------
        obs_np : np.ndarray shape (64, 64, 3)
        action : int

        Returns
        -------
        loss : torch.Tensor scalar (with grad)
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        probs, _ = self.forward(obs_t)
        probs    = probs.squeeze(0)
        return -torch.log(probs[action] + 1e-12)

    def act_greedy(self, obs_np: np.ndarray) -> int:
        """Return highest probability action (for demo collection)."""
        probs = self.get_action_probs(obs_np).numpy()
        return int(np.argmax(probs))

    def act_sample(self, obs_np: np.ndarray) -> int:
        """Sample action from policy distribution."""
        probs = self.get_action_probs(obs_np).numpy()
        return int(np.random.choice(self.n_actions, p=probs))

    def loss_at(self, obs_np: np.ndarray, action: int) -> float:
        """
        Compute teacher reference loss: -log π_teacher(action|obs)
        Used in TV formula Term 2.
        Stored at demo collection time for correct multi-level reference.

        Parameters
        ----------
        obs_np : np.ndarray shape (64, 64, 3)
        action : int

        Returns
        -------
        loss : float
        """
        obs_t = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.forward(obs_t)
        probs = probs.squeeze(0).cpu().numpy()
        return -float(np.log(probs[action] + 1e-12))

    def clone(self) -> 'CNNPolicy':
        """Deep copy for ITAL theta_hat computation."""
        cloned = CNNPolicy(n_actions=self.n_actions).to(DEVICE)
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PPOTrainer:
    """
    PPO trainer for CNNPolicy on Procgen environments.

    Uses clipped surrogate objective with entropy bonus.
    Runs on whatever device is available (CPU or GPU).

    Parameters
    ----------
    policy      : CNNPolicy
    lr          : float — learning rate
    clip_eps    : float — PPO clip parameter
    entropy_coef: float — entropy bonus coefficient
    value_coef  : float — value loss coefficient
    max_grad_norm: float — gradient clipping
    """

    def __init__(self,
                 policy        : CNNPolicy,
                 lr            : float = 2.5e-4,
                 clip_eps      : float = 0.2,
                 entropy_coef  : float = 0.01,
                 value_coef    : float = 0.5,
                 max_grad_norm : float = 0.5):

        self.policy        = policy.to(DEVICE)
        self.optimiser     = optim.Adam(
            policy.parameters(), lr=lr, eps=1e-5)
        self.clip_eps      = clip_eps
        self.entropy_coef  = entropy_coef
        self.value_coef    = value_coef
        self.max_grad_norm = max_grad_norm

    def update(self, rollout: dict, n_epochs: int = 4,
               minibatch_size: int = 256) -> dict:
        """
        Run PPO update on collected rollout data.

        Parameters
        ----------
        rollout : dict with keys:
            obs      : np.ndarray (N, 64, 64, 3)
            actions  : np.ndarray (N,)
            returns  : np.ndarray (N,) — discounted returns
            advantages: np.ndarray (N,) — GAE advantages
            log_probs_old: np.ndarray (N,) — old log probs

        Returns
        -------
        metrics : dict with loss values for logging
        """
        obs_t       = torch.tensor(
            rollout['obs'],       dtype=torch.float32).to(DEVICE)
        actions_t   = torch.tensor(
            rollout['actions'],   dtype=torch.long).to(DEVICE)
        returns_t   = torch.tensor(
            rollout['returns'],   dtype=torch.float32).to(DEVICE)
        advantages_t = torch.tensor(
            rollout['advantages'], dtype=torch.float32).to(DEVICE)
        log_probs_old_t = torch.tensor(
            rollout['log_probs_old'], dtype=torch.float32).to(DEVICE)

        # Normalise advantages
        advantages_t = (advantages_t - advantages_t.mean()) / \
                       (advantages_t.std() + 1e-8)

        N = len(obs_t)
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        for _ in range(n_epochs):
            # Shuffle data
            idx = torch.randperm(N)
            for start in range(0, N, minibatch_size):
                mb_idx     = idx[start:start + minibatch_size]
                mb_obs     = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_adv     = advantages_t[mb_idx]
                mb_lp_old  = log_probs_old_t[mb_idx]

                probs, values = self.policy.forward(mb_obs)
                dist          = Categorical(probs)
                log_probs_new = dist.log_prob(mb_actions)
                entropy       = dist.entropy().mean()

                # PPO clipped objective
                ratio      = torch.exp(log_probs_new - mb_lp_old)
                surr1      = ratio * mb_adv
                surr2      = torch.clamp(
                    ratio, 1 - self.clip_eps,
                           1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss  = F.mse_loss(
                    values.squeeze(-1), mb_returns)

                # Total loss
                loss = (policy_loss
                        + self.value_coef  * value_loss
                        - self.entropy_coef * entropy)

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                self.optimiser.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()

        n_updates = n_epochs * max(1, N // minibatch_size)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss" : total_value_loss  / n_updates,
            "entropy"    : total_entropy     / n_updates,
        }


def collect_rollout(env, policy: CNNPolicy,
                    n_steps: int = 2048,
                    gamma: float = 0.99,
                    gae_lambda: float = 0.95) -> dict:
    """
    Collect rollout data from environment for PPO update.
    Works with Procgen's vectorised environment interface.

    Parameters
    ----------
    env     : Procgen ProcgenEnv (vectorised, num_envs=1)
    policy  : CNNPolicy
    n_steps : int — steps to collect per rollout
    gamma   : float — discount factor
    gae_lambda: float — GAE lambda

    Returns
    -------
    rollout : dict ready for PPOTrainer.update()
    """
    obs_list      = []
    action_list   = []
    reward_list   = []
    done_list     = []
    value_list    = []
    log_prob_list = []

    obs = env.reset()
    obs = obs['rgb']   # shape (num_envs, 64, 64, 3)

    for _ in range(n_steps):
        obs_t  = torch.tensor(
            obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs, values = policy.forward(obs_t)
        dist    = Categorical(probs)
        actions = dist.sample()
        log_p   = dist.log_prob(actions)

        actions_np = actions.cpu().numpy()
        next_data  = env.step(actions_np)
        next_obs   = next_data['rgb']
        rewards    = next_data['reward']
        dones      = next_data['done']

        obs_list.append(obs.copy())
        action_list.append(actions_np)
        reward_list.append(rewards)
        done_list.append(dones)
        value_list.append(values.squeeze(-1).cpu().detach().numpy())
        log_prob_list.append(log_p.cpu().detach().numpy())

        obs = next_obs

    # Compute final value for bootstrapping
    obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, last_values = policy.forward(obs_t)
    last_values = last_values.squeeze(-1).cpu().numpy()

    # Compute GAE advantages and returns
    obs_arr      = np.concatenate(obs_list,      axis=0)
    action_arr   = np.concatenate(action_list,   axis=0)
    reward_arr   = np.concatenate(reward_list,   axis=0)
    done_arr     = np.concatenate(done_list,     axis=0)
    value_arr    = np.concatenate(value_list,    axis=0)
    log_prob_arr = np.concatenate(log_prob_list, axis=0)

    advantages   = np.zeros_like(reward_arr)
    last_gae     = 0.0

    for t in reversed(range(len(reward_arr))):
        if t == len(reward_arr) - 1:
            next_val = last_values[0]
            next_done = 0.0
        else:
            next_val  = value_arr[t + 1]
            next_done = done_arr[t + 1]
        delta    = (reward_arr[t]
                    + gamma * next_val * (1 - next_done)
                    - value_arr[t])
        last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
        advantages[t] = last_gae

    returns = advantages + value_arr

    return {
        'obs'          : obs_arr,
        'actions'      : action_arr,
        'returns'      : returns.astype(np.float32),
        'advantages'   : advantages.astype(np.float32),
        'log_probs_old': log_prob_arr.astype(np.float32),
    }
