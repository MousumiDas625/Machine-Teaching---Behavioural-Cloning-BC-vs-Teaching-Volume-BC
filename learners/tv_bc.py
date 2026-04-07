# learners/tv_bc.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from agents.rl_teacher import RLTeacher


class TeacherAwareBC:
    """
    Teacher-Aware Behavioral Cloning using the ITAL update rule.
    (Yuan et al., NeurIPS 2021 — Algorithm 1)

    Works with both:
      - 64-dim position-only observations (PolicyNetwork)
      - 128-dim map observation inputs (PolicyNetworkMapObs)

    The policy object must have these methods:
      .compute_loss(obs_np, action) -> torch.Tensor
      .clone()                      -> copy of network
      .parameters()                 -> iterator
      .zero_grad()                  -> clears gradients
      .train()                      -> sets training mode

    Two modes of use:
    ─────────────────
    1. BATCH mode:
       Call train(states, actions, n_epochs, ...)
       TV is recomputed each mini-batch, dataset is fixed.

    2. ITERATIVE mode:
       Call step(batch) once per iteration with a fresh batch.
       TV is recomputed at CURRENT θ every call.
       This is the correct implementation of Algorithm 1.

    ITAL update (one mini-batch):
    ──────────────────────────────
    Given batch D_t = {(obs_i, a_i)}:

    1. Compute TV_i = TV(obs_i, a_i | θ)  for all i in D_t
    2. Sample teacher-chosen index t ~ softmax(β · TV)
    3. Naive BC step:   g_t = ∇_θ ℓ(θ; obs_t, a_t)
                        θ_hat = θ - η · g_t
    4. Recompute TV at θ_hat, build q_hat distribution
    5. Expected gradient: g_q = Σ_i q_hat_i · ∇_{θ_hat} ℓ(θ_hat; obs_i, a_i)
    6. Correction:  g_t_hat = ∇_{θ_hat} ℓ(θ_hat; obs_t, a_t)
                    θ_new = θ_hat - 2βη²(g_t_hat - g_q)
                    (correction clipped to norm ≤ 1.0)

    When β=0, correction vanishes → reduces to standard BC.

    Changes from original version:
    ────────────────────────────────
    FIX 1: state_idx extraction handles both 64-dim and 128-dim obs.
            For 128-dim map obs, argmax is taken over obs[64:] (position part)
            not the full obs (which would pick the goal cell via map encoding).

    FIX 2: ITAL correction is clipped to norm ≤ 1.0 before being applied.
            Prevents single-round policy crashes caused by large corrections
            when batch TV scores are highly variable.

    FIX 3: Removed strict PolicyNetwork type hints so PolicyNetworkMapObs
            works without import changes.
    """

    def __init__(self, policy, teacher,
                 eta: float = 0.01,
                 beta: float = 2.0,
                 seed: int = 0):
        """
        Parameters
        ----------
        policy  : policy network object
                  Must have compute_loss(), clone(), parameters(),
                  zero_grad(), train() methods.
                  Works with both PolicyNetwork (64-dim) and
                  PolicyNetworkMapObs (128-dim).
        teacher : RLTeacher or None
                  Provides reference loss ℓ(ω*; s, a).
                  Can be set to None at init and assigned later
                  via self.teacher = new_teacher (used in generalisation
                  experiment where teacher changes every round).
        eta     : float — SGD learning rate, same η used in TV formula
        beta    : float — Boltzmann temperature
                  β=0   → uniform selection (reduces to BC)
                  β=2.0 → moderately selective
                  β→∞   → always picks highest TV example
        seed    : int — for stochastic teacher sampling
        """
        self.policy  = policy
        self.teacher = teacher
        self.eta     = eta
        self.beta    = beta
        self.rng     = np.random.default_rng(seed)

        # Logging
        self.loss_history = []
        self.eval_history = []
        self.eval_steps   = []

    # ------------------------------------------------------------------ #
    #  Helper: extract cell index from observation                        #
    # ------------------------------------------------------------------ #

    def _state_idx_from_obs(self, obs_np: np.ndarray) -> int:
        """
        Extract the agent's cell index from an observation vector.

        FIX 1: Handles both 64-dim and 128-dim observations correctly.

        For 64-dim (position only):
            obs is a one-hot vector → argmax gives cell index directly

        For 128-dim (map observation):
            obs[:64]  = map encoding (omega_star values)
            obs[64:]  = position one-hot
            argmax of FULL obs would return the goal cell index
            (because the goal cell has value +1.0 in the map encoding)
            We must take argmax of obs[64:] ONLY to get the position.

        Parameters
        ----------
        obs_np : np.ndarray, shape (64,) or (128,)

        Returns
        -------
        cell_index : int in {0, ..., 63}
        """
        if len(obs_np) == 128:
            return int(np.argmax(obs_np[64:]))   # position part only
        else:
            return int(np.argmax(obs_np))        # full obs is position

    # ------------------------------------------------------------------ #
    #  Helper: flat gradient vector                                       #
    # ------------------------------------------------------------------ #

    def _get_flat_grad(self, policy) -> torch.Tensor:
        """
        Collect all parameter gradients into one flat vector.
        Called after loss.backward() to get ∇_θ ℓ as a single tensor.
        Returns zero vector for parameters with no gradient.
        """
        grads = []
        for p in policy.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(p.numel()))
        return torch.cat(grads)

    # ------------------------------------------------------------------ #
    #  Helper: Teaching Volume for one (obs, action) pair                #
    # ------------------------------------------------------------------ #

    def _compute_tv(self, policy, obs_np: np.ndarray,
                    action: int) -> float:
        """
        TV(obs, a | θ) = -η² · ‖∇_θ ℓ(θ; obs, a)‖²
                       + 2η  · [ℓ(θ; obs, a) - ℓ(ω*; s, a)]

        Term 1: penalises examples with large gradients (destabilising)
        Term 2: rewards examples where learner is much worse than teacher

        The teacher reference loss ℓ(ω*; s, a) uses the cell index s,
        not the full observation. The cell index is extracted from obs
        using _state_idx_from_obs() which handles both 64 and 128-dim.

        Parameters
        ----------
        policy  : policy network at current θ
        obs_np  : np.ndarray shape (64,) or (128,)
        action  : int

        Returns
        -------
        tv : float
        """
        # --- Compute learner loss with gradient tracking ---
        policy.zero_grad()
        learner_loss = policy.compute_loss(obs_np, action)
        learner_loss.backward()

        # --- Term 1: gradient penalty ---
        flat_grad    = self._get_flat_grad(policy)
        grad_norm_sq = float(flat_grad.dot(flat_grad).item())
        term1        = -(self.eta ** 2) * grad_norm_sq

        # --- Term 2: loss gap ---
        learner_loss_val = learner_loss.item()

        # FIX 1: use correct state extraction for 64 or 128-dim obs
        state_idx        = self._state_idx_from_obs(obs_np)
        teacher_loss_val = self.teacher.loss_at(state_idx, action)
        term2            = 2.0 * self.eta * (learner_loss_val
                                              - teacher_loss_val)

        return term1 + term2

    # ------------------------------------------------------------------ #
    #  Core ITAL update — one mini-batch                                  #
    # ------------------------------------------------------------------ #

    def _ital_step(self, batch: list) -> float:
        """
        Full 6-step ITAL update on one mini-batch.

        Parameters
        ----------
        batch : list of (obs_np, action_int) tuples
                obs_np can be shape (64,) or (128,)

        Returns
        -------
        avg_loss : float — average cross-entropy loss for logging
        """
        n = len(batch)

        # ── Step 1: TV for every example at current θ ──────────────────
        tv_scores   = []
        loss_values = []

        for (obs_np, action) in batch:
            tv = self._compute_tv(self.policy, obs_np, action)
            tv_scores.append(tv)
            with torch.no_grad():
                lv = self.policy.compute_loss(obs_np, action).item()
            loss_values.append(lv)

        avg_loss = float(np.mean(loss_values))

        # ── Step 2: Sample teacher-chosen index ~ softmax(β · TV) ───────
        tv_arr   = np.array(tv_scores, dtype=np.float64)
        tv_arr  -= tv_arr.max()                   # numerical stability
        weights  = np.exp(self.beta * tv_arr)
        probs    = weights / weights.sum()
        chosen_t = int(self.rng.choice(n, p=probs))

        obs_t, a_t = batch[chosen_t]

        # ── Step 3: Naive BC step on chosen example ─────────────────────
        #    g_t = ∇_θ ℓ(θ; obs_t, a_t)
        #    θ_hat = θ - η · g_t
        self.policy.zero_grad()
        loss_t = self.policy.compute_loss(obs_t, a_t)
        loss_t.backward()
        g_t = self._get_flat_grad(self.policy)

        # Apply to a CLONE — θ itself is not yet modified
        theta_hat = self.policy.clone()
        with torch.no_grad():
            offset = 0
            for (p_orig, p_hat) in zip(self.policy.parameters(),
                                       theta_hat.parameters()):
                numel      = p_orig.numel()
                grad_chunk = g_t[offset: offset + numel].view(p_orig.shape)
                p_hat.data.copy_(p_orig.data - self.eta * grad_chunk)
                offset += numel

        # ── Step 4: Recompute TV at θ_hat, build q_hat ──────────────────
        tv_hat = []
        for (obs_np, action) in batch:
            tv = self._compute_tv(theta_hat, obs_np, action)
            tv_hat.append(tv)

        tv_hat_arr  = np.array(tv_hat, dtype=np.float64)
        tv_hat_arr -= tv_hat_arr.max()
        w_hat       = np.exp(self.beta * tv_hat_arr)
        q_hat       = w_hat / w_hat.sum()

        # ── Step 5: Expected gradient g_q ───────────────────────────────
        #    g_q = Σ_i q_hat_i · ∇_{θ_hat} ℓ(θ_hat; obs_i, a_i)
        n_params = sum(p.numel() for p in theta_hat.parameters())
        g_q      = torch.zeros(n_params)

        for i, (obs_np, action) in enumerate(batch):
            theta_hat.zero_grad()
            loss_i = theta_hat.compute_loss(obs_np, action)
            loss_i.backward()
            g_i  = self._get_flat_grad(theta_hat)
            g_q += float(q_hat[i]) * g_i

        # ── Step 6: ITAL correction with clipping ───────────────────────
        #    g_t_hat = ∇_{θ_hat} ℓ(θ_hat; obs_t, a_t)
        #    correction = 2βη²(g_t_hat - g_q)
        #    θ_new = θ_hat - correction
        theta_hat.zero_grad()
        loss_t_hat = theta_hat.compute_loss(obs_t, a_t)
        loss_t_hat.backward()
        g_t_hat = self._get_flat_grad(theta_hat)

        correction = 2.0 * self.beta * (self.eta ** 2) * (g_t_hat - g_q)

        # FIX 2: clip correction norm to 1.0
        # Prevents single-round crashes when batch TV scores are
        # highly variable and g_t_hat - g_q is very large.
        # Same clipping philosophy as gradient clipping in standard_bc.py.
        corr_norm = float(correction.norm().item())
        if corr_norm > 1.0:
            correction = correction / corr_norm

        # Write θ_new into self.policy
        with torch.no_grad():
            offset = 0
            for (p_main, p_hat) in zip(self.policy.parameters(),
                                       theta_hat.parameters()):
                numel      = p_main.numel()
                corr_chunk = correction[offset: offset + numel].view(
                    p_main.shape)
                p_main.data.copy_(p_hat.data - corr_chunk)
                offset += numel

        return avg_loss

    # ------------------------------------------------------------------ #
    #  ITERATIVE MODE — single update step                                #
    # ------------------------------------------------------------------ #

    def step(self, batch: list) -> float:
        """
        One complete ITAL update on a freshly sampled mini-batch.

        TV is recomputed using CURRENT θ every call.
        This is the correct iterative implementation of Algorithm 1.

        Parameters
        ----------
        batch : list of (obs_np, action_int) tuples
                obs_np can be shape (64,) or (128,)

        Returns
        -------
        loss : float — average cross-entropy loss
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
        states     : list of np.ndarray each shape (64,) or (128,)
        actions    : list of int
        n_epochs   : number of full passes over the dataset
        batch_size : mini-batch size
        eval_fn    : callable(policy) -> float, or None
        eval_every : how often in epochs to evaluate
        verbose    : print progress
        """
        N = len(states)
        if verbose:
            print(f"  [TV-BC] Training for {n_epochs} epochs on {N} "
                  f"samples (batch_size={batch_size}, beta={self.beta})")

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
