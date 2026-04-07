# learners/tv_bc.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from agents.policy_net import PolicyNetwork
from agents.rl_teacher import RLTeacher


class TeacherAwareBC:
    """
    Teacher-Aware Behavioral Cloning using the ITAL update rule.
    (Yuan et al., NeurIPS 2021 — Algorithm 1)

    Two modes of use:
    ─────────────────
    1. BATCH mode (original):
       Call train(states, actions, n_epochs, ...) to train over a fixed
       dataset. TV is recomputed each mini-batch but the dataset is fixed.

    2. ITERATIVE mode (new — what the professor asked for):
       Call step(batch) once per iteration with a freshly sampled batch.
       TV is recomputed at CURRENT θ every single call, so the teacher's
       example selection adapts as the learner improves. This is true ITAL.

    ITAL update (one mini-batch):
    ──────────────────────────────
    Given batch D_t = {(s_i, a_i)}:

    1. Compute TV_i = TV(s_i, a_i | θ)  for all i in D_t
    2. Sample teacher-chosen index t ~ softmax(β · TV)
    3. Naive BC step:   g_t = ∇_θ ℓ(θ; s_t, a_t)
                        θ_hat = θ - η · g_t
    4. Recompute TV at θ_hat, build q_hat distribution
    5. Expected gradient: g_q = Σ_i q_hat_i · ∇_{θ_hat} ℓ(θ_hat; s_i, a_i)
    6. Correction:  g_t_hat = ∇_{θ_hat} ℓ(θ_hat; s_t, a_t)
                    θ_new = θ_hat - 2βη²(g_t_hat - g_q)

    When β=0, correction vanishes → reduces to standard BC.
    """

    def __init__(self, policy: PolicyNetwork, teacher: RLTeacher,
                 eta: float = 0.01, beta: float = 2.0, seed: int = 0):
        """
        Parameters
        ----------
        policy  : PolicyNetwork  — learner's policy, modified in-place
        teacher : RLTeacher      — provides reference loss ℓ(ω*; s, a)
        eta     : float          — SGD learning rate (same η in TV formula)
        beta    : float          — Boltzmann temperature for teacher dist.
                                   β=0 → uniform (reduces to BC)
                                   β=2 → moderately selective
                                   β→∞ → always picks highest TV example
        seed    : int            — for stochastic teacher sampling
        """
        self.policy  = policy
        self.teacher = teacher
        self.eta     = eta
        self.beta    = beta
        self.rng     = np.random.default_rng(seed)

        # Logging
        self.loss_history = []   # avg loss per step or epoch
        self.eval_history = []   # policy return at checkpoints
        self.eval_steps   = []   # iteration index of each checkpoint

    # ------------------------------------------------------------------ #
    #  Helper: flat gradient vector                                        #
    # ------------------------------------------------------------------ #

    def _get_flat_grad(self, policy: PolicyNetwork) -> torch.Tensor:
        """
        Collect all parameter gradients into one flat vector.
        Called after loss.backward() to get ∇_θ ℓ as a single tensor.
        """
        grads = []
        for p in policy.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(p.numel()))
        return torch.cat(grads)

    def _apply_flat_grad(self, policy: PolicyNetwork,
                         flat_grad: torch.Tensor) -> None:
        """
        Write a flat gradient vector back into policy.parameters().grad.
        Used when we need to apply a manually computed gradient.
        """
        offset = 0
        for p in policy.parameters():
            numel = p.numel()
            p.grad = flat_grad[offset: offset + numel].view(p.shape).clone()
            offset += numel

    # ------------------------------------------------------------------ #
    #  Helper: Teaching Volume for one (s, a) pair                        #
    # ------------------------------------------------------------------ #

    def _compute_tv(self, policy: PolicyNetwork,
                    state_np: np.ndarray, action: int) -> float:
        """
        TV(s, a | θ) = -η² · ‖∇_θ ℓ(θ; s, a)‖² + 2η · [ℓ(θ;s,a) - ℓ(ω*;s,a)]

        Parameters
        ----------
        policy   : the policy whose parameters define θ
        state_np : np.ndarray shape (64,) — one-hot state vector
        action   : int — action index

        Returns
        -------
        tv : float
        """
        # --- Term 2 part A: learner's loss ---
        # Needs grad so we can compute grad norm for Term 1
        policy.zero_grad()
        learner_loss = policy.compute_loss(state_np, action)
        learner_loss.backward()
        flat_grad = self._get_flat_grad(policy)
        grad_norm_sq = float(flat_grad.dot(flat_grad).item())

        # --- Term 1: gradient penalty ---
        term1 = -(self.eta ** 2) * grad_norm_sq

        # --- Term 2: loss gap ---
        learner_loss_val = learner_loss.item()
        state_idx        = int(np.argmax(state_np))
        teacher_loss_val = self.teacher.loss_at(state_idx, action)
        term2            = 2.0 * self.eta * (learner_loss_val - teacher_loss_val)

        return term1 + term2

    # ------------------------------------------------------------------ #
    #  Core ITAL update — one mini-batch                                  #
    # ------------------------------------------------------------------ #

    def _ital_step(self, batch: list) -> float:
        """
        Full 6-step ITAL update on one mini-batch.

        Parameters
        ----------
        batch : list of (state_np, action_int) tuples

        Returns
        -------
        avg_loss : float — average learner loss over the batch (for logging)
        """
        n = len(batch)

        # ── Step 1: Compute TV for every example at current θ ──────────
        tv_scores   = []
        loss_values = []

        for (state_np, action) in batch:
            tv  = self._compute_tv(self.policy, state_np, action)
            tv_scores.append(tv)
            # Re-read loss without grad (TV already computed it)
            with torch.no_grad():
                lv = self.policy.compute_loss(state_np, action).item()
            loss_values.append(lv)

        avg_loss = float(np.mean(loss_values))

        # ── Step 2: Sample teacher-chosen index ~ softmax(β · TV) ───────
        tv_arr   = np.array(tv_scores, dtype=np.float64)
        tv_arr  -= tv_arr.max()                          # numerical stability
        weights  = np.exp(self.beta * tv_arr)
        probs    = weights / weights.sum()
        chosen_t = int(self.rng.choice(n, p=probs))

        s_t, a_t = batch[chosen_t]

        # ── Step 3: Naive BC step on chosen example ──────────────────────
        #    g_t = ∇_θ ℓ(θ; s_t, a_t)
        #    θ_hat = θ - η · g_t
        self.policy.zero_grad()
        loss_t = self.policy.compute_loss(s_t, a_t)
        loss_t.backward()
        g_t = self._get_flat_grad(self.policy)     # shape: (n_params,)

        # Apply naive SGD step to a CLONE so θ is not yet modified
        theta_hat = self.policy.clone()
        with torch.no_grad():
            offset = 0
            for (p_orig, p_hat) in zip(self.policy.parameters(),
                                       theta_hat.parameters()):
                numel   = p_orig.numel()
                grad_chunk = g_t[offset: offset + numel].view(p_orig.shape)
                p_hat.data.copy_(p_orig.data - self.eta * grad_chunk)
                offset += numel

        # ── Step 4: Recompute TV at θ_hat, build q_hat distribution ─────
        tv_hat = []
        for (state_np, action) in batch:
            tv = self._compute_tv(theta_hat, state_np, action)
            tv_hat.append(tv)

        tv_hat_arr  = np.array(tv_hat, dtype=np.float64)
        tv_hat_arr -= tv_hat_arr.max()
        w_hat       = np.exp(self.beta * tv_hat_arr)
        q_hat       = w_hat / w_hat.sum()            # shape: (n,)

        # ── Step 5: Expected gradient g_q = Σ_i q_hat_i · ∇_{θ_hat} ℓ_i ─
        #    We accumulate a weighted sum of flat gradients
        n_params = sum(p.numel() for p in theta_hat.parameters())
        g_q      = torch.zeros(n_params)

        for i, (state_np, action) in enumerate(batch):
            theta_hat.zero_grad()
            loss_i = theta_hat.compute_loss(state_np, action)
            loss_i.backward()
            g_i  = self._get_flat_grad(theta_hat)
            g_q += float(q_hat[i]) * g_i

        # ── Step 6: Recompute g_t at θ_hat, apply ITAL correction ────────
        #    g_t_hat = ∇_{θ_hat} ℓ(θ_hat; s_t, a_t)
        #    θ_new = θ_hat - 2βη²(g_t_hat - g_q)
        theta_hat.zero_grad()
        loss_t_hat = theta_hat.compute_loss(s_t, a_t)
        loss_t_hat.backward()
        g_t_hat = self._get_flat_grad(theta_hat)

        correction = 2.0 * self.beta * (self.eta ** 2) * (g_t_hat - g_q)

        # Apply θ_new = θ_hat - correction  →  write into self.policy
        with torch.no_grad():
            offset = 0
            for (p_main, p_hat) in zip(self.policy.parameters(),
                                       theta_hat.parameters()):
                numel      = p_main.numel()
                corr_chunk = correction[offset: offset + numel].view(p_main.shape)
                p_main.data.copy_(p_hat.data - corr_chunk)
                offset += numel

        return avg_loss

    # ------------------------------------------------------------------ #
    #  ITERATIVE MODE — single update step (called by run_iterative.py)  #
    # ------------------------------------------------------------------ #

    def step(self, batch: list) -> float:
        """
        One complete ITAL update on a freshly sampled mini-batch.

        This is what run_iterative.py calls at every iteration.
        TV is recomputed using the CURRENT θ every time this is called,
        so the teacher's selection adapts as the learner improves.

        Parameters
        ----------
        batch : list of (state_np, action_int) tuples
                Freshly sampled at each iteration from the full pool.

        Returns
        -------
        loss : float — average cross-entropy loss over the batch
        """
        self.policy.train()
        loss = self._ital_step(batch)
        self.loss_history.append(loss)
        return loss

    # ------------------------------------------------------------------ #
    #  BATCH MODE — epoch-based training over fixed dataset               #
    # ------------------------------------------------------------------ #

    def _train_one_epoch(self, states: list, actions: list,
                         batch_size: int) -> float:
        """One full pass over the fixed dataset. Returns mean loss."""
        self.policy.train()
        N       = len(states)
        indices = np.random.default_rng().permutation(N)
        losses  = []

        for start in range(0, N, batch_size):
            batch_idx = indices[start: start + batch_size]
            batch     = [(states[i], actions[i]) for i in batch_idx]
            loss      = self._ital_step(batch)
            losses.append(loss)

        return float(np.mean(losses))

    def train(self, states: list, actions: list,
              n_epochs: int   = 80,
              batch_size: int = 20,
              eval_fn         = None,
              eval_every: int = 10,
              verbose: bool   = True) -> None:
        """
        Batch-mode training over a fixed dataset for n_epochs.

        Parameters
        ----------
        states     : list of np.ndarray each shape (64,)
        actions    : list of int
        n_epochs   : number of full passes over the dataset
        batch_size : mini-batch size
        eval_fn    : callable(policy) -> float, or None
        eval_every : how often in epochs to evaluate
        verbose    : print progress
        """
        N = len(states)
        if verbose:
            print(f"  [TV-BC] Training for {n_epochs} epochs on {N} samples "
                  f"(batch_size={batch_size}, beta={self.beta})")

        for epoch in range(1, n_epochs + 1):
            loss = self._train_one_epoch(states, actions, batch_size)
            self.loss_history.append(loss)

            if eval_fn is not None and (epoch % eval_every == 0
                                        or epoch == 1
                                        or epoch == n_epochs):
                ret = eval_fn(self.policy)
                self.eval_history.append(ret)
                self.eval_steps.append(epoch)
                if verbose:
                    print(f"  [TV-BC] Epoch {epoch:3d}/{n_epochs}  "
                          f"loss={loss:.4f}  |  eval={ret:.3f}")
            elif verbose and (epoch % eval_every == 0
                              or epoch == 1
                              or epoch == n_epochs):
                print(f"  [TV-BC] Epoch {epoch:3d}/{n_epochs}  "
                      f"loss={loss:.4f}")

        if verbose:
            print(f"  [TV-BC] Training complete. "
                  f"Final loss: {self.loss_history[-1]:.4f}")
