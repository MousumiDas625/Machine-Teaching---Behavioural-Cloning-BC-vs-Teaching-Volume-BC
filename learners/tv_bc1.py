# learners/tv_bc.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.optim as optim
from agents.policy_net          import PolicyNetwork
from agents.rl_teacher          import RLTeacher
from teaching.teaching_volume   import compute_tv_batch
from teaching.subset_selection  import make_minibatches, softmax_with_beta


class TeacherAwareBC:
    """
    Teacher-Aware Behavioral Cloning (TV-BC / ITAL).

    Implements the full ITAL update from:
      - Machine Teaching Report 1 (Section 2, Update Rule)
      - ITAL NeurIPS 2021 paper (Eq. 6, Algorithm 1)

    The learner models the teacher as selecting demonstrations
    proportionally to their Teaching Volume (Boltzmann rational teacher).
    It modifies the gradient update to also incorporate the implicit
    information in WHICH example the teacher chose — not just the
    content of that example.

    Parameters
    ----------
    policy  : PolicyNetwork — the learner's MLP (theta)
    teacher : RLTeacher     — trained teacher (omega*)
    eta     : float         — learning rate
    beta    : float         — Boltzmann temperature for teacher model
    seed    : int
    """

    def __init__(self,
                 policy  : PolicyNetwork,
                 teacher : RLTeacher,
                 eta     : float = 0.01,
                 beta    : float = 2.0,
                 seed    : int   = 7):

        self.policy  = policy
        self.teacher = teacher
        self.eta     = eta
        self.beta    = beta
        self.rng     = np.random.default_rng(seed)
        self.n_params = sum(p.numel() for p in policy.parameters())

        # Training history
        self.loss_history = []
        self.eval_history = []
        self.eval_epochs  = []

    def _get_flat_grad(self,
                       policy : PolicyNetwork,
                       state  : np.ndarray,
                       action : int) -> np.ndarray:
        """
        Compute gradient of loss for (state, action) and return
        as a single flat numpy vector.

        This is the per-example gradient ∇_θ ℓ(θ; s, a) needed for:
          - The ITAL correction term g_{t,θ̂}
          - The expected gradient g_q

        Parameters
        ----------
        policy : PolicyNetwork (could be original or cloned θ̂)
        state  : np.ndarray shape (64,)
        action : int

        Returns
        -------
        flat_grad : np.ndarray — all gradients concatenated into 1D vector
        """
        policy.zero_grad()
        loss = policy.compute_loss(state, action)
        loss.backward()

        grads = []
        for param in policy.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().cpu().numpy().ravel())

        return np.concatenate(grads)

    def _apply_flat_grad(self,
                         policy    : PolicyNetwork,
                         flat_grad : np.ndarray,
                         lr        : float):
        """
        Manually apply a gradient update using a flat gradient vector.

        This is used to apply the ITAL correction which is computed
        as a numpy vector (not a PyTorch autograd gradient).

        theta ← theta - lr * flat_grad

        Parameters
        ----------
        policy    : PolicyNetwork to update in-place
        flat_grad : np.ndarray — gradient vector, same length as all params
        lr        : float — step size
        """
        idx = 0
        with torch.no_grad():
            for param in policy.parameters():
                # How many elements does this parameter have?
                n = param.numel()

                # Slice the corresponding chunk from flat_grad
                grad_chunk = flat_grad[idx : idx + n]

                # Reshape to match the parameter's shape
                grad_tensor = torch.tensor(
                    grad_chunk.reshape(param.shape),
                    dtype=torch.float32
                )

                # Apply update: theta ← theta - lr * grad
                param.data -= lr * grad_tensor

                idx += n

    def _ital_step(self,
                   s_batch : list,
                   a_batch : list) -> float:
        """
        Perform one full ITAL update on a single mini-batch.

        This implements Algorithm 1 from the ITAL paper exactly.

        Parameters
        ----------
        s_batch : list of np.ndarray — states in mini-batch
        a_batch : list of int        — actions in mini-batch

        Returns
        -------
        batch_loss : float — average BC loss for logging
        """
        n = len(s_batch)

        # --------------------------------------------------------
        # Step 1: Compute TV for every (s,a) in batch
        #         using CURRENT parameters theta
        # --------------------------------------------------------
        tv_scores = compute_tv_batch(
            s_batch, a_batch, self.policy, self.teacher, self.eta
        )
        # tv_scores shape: (n,) — one score per example

        # --------------------------------------------------------
        # Step 2: Sample teacher-chosen example from TV softmax
        #         Stochastic Boltzmann sampling (not argmax)
        # --------------------------------------------------------
        q_probs    = softmax_with_beta(tv_scores, beta=self.beta)
        chosen_idx = int(self.rng.choice(n, p=q_probs))
        s_t        = s_batch[chosen_idx]
        a_t        = a_batch[chosen_idx]

        # --------------------------------------------------------
        # Step 3: Naive BC step on the chosen example
        #         theta_hat = theta - eta * grad_t
        # --------------------------------------------------------
        # Compute gradient of chosen example at current theta
        g_t = self._get_flat_grad(self.policy, s_t, a_t)

        # Clone the network to get theta_hat
        # We clone so we can compute things at theta_hat without
        # permanently changing theta yet
        policy_hat = self.policy.clone()

        # Apply: theta_hat = theta - eta * g_t
        self._apply_flat_grad(policy_hat, g_t, lr=self.eta)

        # --------------------------------------------------------
        # Step 4: Recompute TV at theta_hat
        #         Build teacher distribution q_hat
        # --------------------------------------------------------
        tv_hat = compute_tv_batch(
            s_batch, a_batch, policy_hat, self.teacher, self.eta
        )
        q_hat = softmax_with_beta(tv_hat, beta=self.beta)
        # q_hat shape: (n,) — probabilities at theta_hat

        # --------------------------------------------------------
        # Step 5: Expected gradient under q_hat
        #         g_q = sum_i q_hat_i * grad_i(theta_hat)
        # --------------------------------------------------------
        n_params = len(g_t)
        g_q      = np.zeros(n_params, dtype=np.float64)

        for i, (s, a) in enumerate(zip(s_batch, a_batch)):
            # Gradient of example i evaluated at theta_hat
            g_i  = self._get_flat_grad(policy_hat, s, a)
            # Weighted by its probability under teacher distribution
            g_q += q_hat[i] * g_i

        # --------------------------------------------------------
        # Step 6: ITAL correction
        #         Gradient of chosen example AT theta_hat
        # --------------------------------------------------------
        g_t_hat = self._get_flat_grad(policy_hat, s_t, a_t)

        # Correction term: 2 * beta * eta^2 * (g_{t,theta_hat} - g_q)
        correction = 2.0 * self.beta * (self.eta ** 2) * (g_t_hat - g_q)

        # Full update: theta_new = theta_hat - correction
        # First get theta_hat values, then subtract correction
        # We do this by applying correction directly to policy_hat
        # (which already has theta_hat weights)
        self._apply_flat_grad(policy_hat, correction, lr=1.0)

        # --------------------------------------------------------
        # Copy final theta_new back into self.policy
        # --------------------------------------------------------
        self.policy.load_state_dict(policy_hat.state_dict())

        # Compute average batch loss at new theta for logging
        self.policy.eval()
        with torch.no_grad():
            batch_loss = np.mean([
                self.policy.compute_loss(s, a).item()
                for s, a in zip(s_batch, a_batch)
            ])
        self.policy.train()

        return float(batch_loss)

    def train(self,
              states     : list,
              actions    : list,
              n_epochs   : int,
              batch_size : int  = 20,
              eval_fn          = None,
              eval_every : int  = 10,
              verbose    : bool = True) -> "TeacherAwareBC":
        """
        Full training loop — same interface as StandardBC.train().

        Parameters
        ----------
        states     : list of np.ndarray
        actions    : list of int
        n_epochs   : int
        batch_size : int
        eval_fn    : callable(policy) → float, optional
        eval_every : int
        verbose    : bool

        Returns
        -------
        self
        """
        print(f"  [TV-BC] Training for {n_epochs} epochs on "
              f"{len(states)} samples (batch_size={batch_size}, "
              f"beta={self.beta})")

        for epoch in range(n_epochs):
            self.policy.train()
            batches    = make_minibatches(states, actions,
                                          batch_size, self.rng)
            epoch_loss = 0.0
            n_batches  = 0

            for s_batch, a_batch in batches:
                batch_loss  = self._ital_step(s_batch, a_batch)
                epoch_loss += batch_loss
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)

            # Periodic evaluation
            if eval_fn is not None and (epoch % eval_every == 0
                                         or epoch == n_epochs - 1):
                self.policy.eval()
                score = eval_fn(self.policy)
                self.eval_history.append(score)
                self.eval_epochs.append(epoch)
                self.policy.train()

            if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
                eval_str = ""
                if self.eval_history:
                    eval_str = f"  |  eval={self.eval_history[-1]:.3f}"
                print(f"  [TV-BC] Epoch {epoch+1:3d}/{n_epochs}  "
                      f"loss={avg_loss:.4f}{eval_str}")

        print(f"  [TV-BC] Training complete. "
              f"Final loss: {self.loss_history[-1]:.4f}")
        return self
