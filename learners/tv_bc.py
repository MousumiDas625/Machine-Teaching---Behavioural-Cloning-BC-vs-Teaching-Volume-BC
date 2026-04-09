# learners/tv_bc.py
# ITAL learner — Teacher-Aware Behavioral Cloning
# Yuan et al., NeurIPS 2021 — Algorithm 1
#
# KEY FIX: batch items can be either:
#   (obs, action)                  — old format, uses self.teacher
#   (obs, action, teacher_loss)    — new format, uses stored teacher loss
#
# The stored teacher_loss ensures the ITAL correction uses the correct
# teacher reference for each example, regardless of which training map
# it came from. This fixes the oscillation caused by using the wrong
# teacher for old-map examples.

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


class TeacherAwareBC:

    def __init__(self, policy, teacher,
                 eta: float = 0.01,
                 beta: float = 2.0,
                 seed: int = 0):
        self.policy  = policy
        self.teacher = teacher   # used only if teacher_loss not in batch
        self.eta     = eta
        self.beta    = beta
        self.rng     = np.random.default_rng(seed)
        self.loss_history = []
        self.eval_history = []
        self.eval_steps   = []

    def _state_idx_from_obs(self, obs_np):
        if len(obs_np) == 128:
            return int(np.argmax(obs_np[64:]))
        return int(np.argmax(obs_np))

    def _get_flat_grad(self, policy):
        grads = []
        for p in policy.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(p.numel()))
        return torch.cat(grads)

    def _get_teacher_loss(self, obs_np, action, stored_tl):
        """
        Get teacher reference loss.
        If stored_tl is provided (new format), use it directly.
        Otherwise fall back to self.teacher.loss_at() (old format).
        """
        if stored_tl is not None:
            return stored_tl
        state_idx = self._state_idx_from_obs(obs_np)
        return self.teacher.loss_at(state_idx, action)

    def _compute_tv(self, policy, obs_np, action, stored_tl=None):
        """
        TV = -eta² ||grad||² + 2eta [loss_learner - teacher_loss]
        Uses stored teacher loss if available (correct for multi-map setting).
        """
        policy.zero_grad()
        learner_loss = policy.compute_loss(obs_np, action)
        learner_loss.backward()
        flat_grad    = self._get_flat_grad(policy)
        grad_norm_sq = float(flat_grad.dot(flat_grad).item())
        term1        = -(self.eta ** 2) * grad_norm_sq
        learner_l    = learner_loss.item()
        teacher_l    = self._get_teacher_loss(obs_np, action, stored_tl)
        term2        = 2.0 * self.eta * (learner_l - teacher_l)
        return term1 + term2

    def _parse_batch(self, batch):
        """
        Parse batch items. Each item is either:
          (obs, action)                 → stored_tl = None
          (obs, action, teacher_loss)   → stored_tl = float
        Returns list of (obs, action, stored_tl_or_None).
        """
        parsed = []
        for item in batch:
            if len(item) == 3:
                parsed.append((item[0], item[1], item[2]))
            else:
                parsed.append((item[0], item[1], None))
        return parsed

    def _ital_step(self, batch):
        parsed = self._parse_batch(batch)
        n      = len(parsed)

        # Step 1: TV for all examples at current theta
        tv_scores   = []
        loss_values = []
        for (obs_np, action, stored_tl) in parsed:
            tv = self._compute_tv(self.policy, obs_np, action, stored_tl)
            tv_scores.append(tv)
            with torch.no_grad():
                lv = self.policy.compute_loss(obs_np, action).item()
            loss_values.append(lv)

        avg_loss = float(np.mean(loss_values))

        # Step 2: Sample teacher-chosen example ~ softmax(beta * TV)
        tv_arr   = np.array(tv_scores, dtype=np.float64)
        tv_arr  -= tv_arr.max()
        weights  = np.exp(self.beta * tv_arr)
        probs    = weights / weights.sum()
        chosen_t = int(self.rng.choice(n, p=probs))
        obs_t, a_t, tl_t = parsed[chosen_t]

        # Step 3: Naive BC step on chosen example → theta_hat on clone
        self.policy.zero_grad()
        loss_t = self.policy.compute_loss(obs_t, a_t)
        loss_t.backward()
        g_t = self._get_flat_grad(self.policy)

        theta_hat = self.policy.clone()
        with torch.no_grad():
            offset = 0
            for (p_orig, p_hat) in zip(self.policy.parameters(),
                                       theta_hat.parameters()):
                numel      = p_orig.numel()
                grad_chunk = g_t[offset:offset+numel].view(p_orig.shape)
                p_hat.data.copy_(p_orig.data - self.eta * grad_chunk)
                offset    += numel

        # Step 4: Recompute TV at theta_hat
        tv_hat = []
        for (obs_np, action, stored_tl) in parsed:
            tv = self._compute_tv(theta_hat, obs_np, action, stored_tl)
            tv_hat.append(tv)
        tv_hat_arr  = np.array(tv_hat, dtype=np.float64)
        tv_hat_arr -= tv_hat_arr.max()
        w_hat       = np.exp(self.beta * tv_hat_arr)
        q_hat       = w_hat / w_hat.sum()

        # Step 5: Expected gradient g_q
        n_params = sum(p.numel() for p in theta_hat.parameters())
        g_q      = torch.zeros(n_params)
        for i, (obs_np, action, _) in enumerate(parsed):
            theta_hat.zero_grad()
            loss_i = theta_hat.compute_loss(obs_np, action)
            loss_i.backward()
            g_i  = self._get_flat_grad(theta_hat)
            g_q += float(q_hat[i]) * g_i

        # Step 6: ITAL correction with clipping
        theta_hat.zero_grad()
        loss_t_hat = theta_hat.compute_loss(obs_t, a_t)
        loss_t_hat.backward()
        g_t_hat    = self._get_flat_grad(theta_hat)
        correction = 2.0 * self.beta * (self.eta ** 2) * (g_t_hat - g_q)

        # Clip correction norm to 1.0
        corr_norm = float(correction.norm().item())
        if corr_norm > 1.0:
            correction = correction / corr_norm

        with torch.no_grad():
            offset = 0
            for (p_main, p_hat) in zip(self.policy.parameters(),
                                       theta_hat.parameters()):
                numel      = p_main.numel()
                corr_chunk = correction[offset:offset+numel].view(p_main.shape)
                p_main.data.copy_(p_hat.data - corr_chunk)
                offset    += numel

        return avg_loss

    def step(self, batch):
        self.policy.train()
        loss = self._ital_step(batch)
        self.loss_history.append(loss)
        return loss

    def _train_one_epoch(self, states, actions, batch_size):
        self.policy.train()
        N       = len(states)
        indices = np.random.default_rng().permutation(N)
        losses  = []
        for start in range(0, N, batch_size):
            batch_idx = indices[start:start+batch_size]
            batch     = [(states[i], actions[i]) for i in batch_idx]
            losses.append(self._ital_step(batch))
        return float(np.mean(losses))

    def train(self, states, actions, n_epochs=80,
              batch_size=20, eval_fn=None,
              eval_every=10, verbose=True):
        N = len(states)
        if verbose:
            print(f"  [TV-BC] {n_epochs} epochs, {N} samples, "
                  f"beta={self.beta}")
        for epoch in range(1, n_epochs + 1):
            loss = self._train_one_epoch(states, actions, batch_size)
            self.loss_history.append(loss)
            if eval_fn and (epoch % eval_every == 0
                            or epoch == 1 or epoch == n_epochs):
                ret = eval_fn(self.policy)
                self.eval_history.append(ret)
                self.eval_steps.append(epoch)
                if verbose:
                    print(f"  [TV-BC] Epoch {epoch:3d}  "
                          f"loss={loss:.4f}  eval={ret:.3f}")
        if verbose:
            print(f"  [TV-BC] Done. Final loss: "
                  f"{self.loss_history[-1]:.4f}")
