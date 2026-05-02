# agents/pavel_policy.py
#
# IMPALA-style ResNet policy matching Pavel's checkpoint exactly.
# Architecture reverse-engineered from model_200015872.pth keys.
#
# embedder:
#   block1: conv(3->16) + 2 residual blocks (16->16)
#   block2: conv(16->32) + 2 residual blocks (32->32)
#   block3: conv(32->32) + 2 residual blocks (32->32)
#   fc: 2048 -> 256
# fc_policy: 256 -> 15
# fc_value:  256 -> 1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.res1  = ResBlock(out_channels)
        self.res2  = ResBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ImpalaBlock(3,  16)
        self.block2 = ImpalaBlock(16, 32)
        self.block3 = ImpalaBlock(32, 32)
        self.fc     = nn.Linear(2048, 256)

    def forward(self, x):
        # x: (B, 64, 64, 3) uint8 or float32
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)  # (B,H,W,C) -> (B,C,H,W)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class PavelPolicy(nn.Module):
    """
    IMPALA-style policy matching Pavel's checkpoint.
    Provides same interface as CNNPolicy for drop-in replacement.
    """

    def __init__(self, n_actions=15):
        super().__init__()
        self.n_actions = n_actions
        self.embedder  = ImpalaEmbedder()
        self.fc_policy = nn.Linear(256, n_actions)
        self.fc_value  = nn.Linear(256, 1)

    def forward(self, obs):
        features = self.embedder(obs)
        logits   = self.fc_policy(features)
        probs    = F.softmax(logits, dim=-1)
        value    = self.fc_value(features)
        return probs, value

    def get_action_probs(self, obs_np):
        obs_t = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.forward(obs_t)
        return probs.squeeze(0).cpu()

    def compute_loss(self, obs_np, action):
        obs_t = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        probs, _ = self.forward(obs_t)
        probs    = probs.squeeze(0)
        return -torch.log(probs[action] + 1e-12)

    def act_greedy(self, obs_np):
        probs = self.get_action_probs(obs_np).numpy()
        return int(np.argmax(probs))

    def loss_at(self, obs_np, action):
        obs_t = torch.tensor(obs_np, dtype=torch.float32).to(DEVICE)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            probs, _ = self.forward(obs_t)
        probs = probs.squeeze(0).cpu().numpy()
        return -float(np.log(probs[action] + 1e-12))

    def clone(self):
        cloned = PavelPolicy(n_actions=self.n_actions).to(DEVICE)
        cloned.load_state_dict(copy.deepcopy(self.state_dict()))
        return cloned

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def load_pavel_expert(ckpt_path, device=DEVICE):
    """
    Load Pavel's trained checkpoint into PavelPolicy.
    """
    policy = PavelPolicy(n_actions=15).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device,
                        weights_only=False)
    policy.load_state_dict(ckpt['model_state_dict'])
    policy.eval()
    print(f"  Loaded Pavel expert from {ckpt_path}")
    print(f"  Parameters: {policy.count_parameters():,}")
    return policy
