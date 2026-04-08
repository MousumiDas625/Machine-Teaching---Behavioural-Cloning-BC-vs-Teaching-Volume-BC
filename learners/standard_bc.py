# learners/standard_bc.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.optim as optim

from agents.policy_net_mapobs import PolicyNetworkMapObs


class StandardBC:
    """
    Standard Behavioral Cloning using SGD.

    Two modes of use:
    ─────────────────
    1. BATCH mode (original):
       Call train(states, actions, n_epochs, ...) to train over a fixed
       dataset for multiple epochs. Used in the batch comparison experiment.

    2. ITERATIVE mode (new):
       Call step(batch) once per iteration, where batch is a freshly sampled
       mini-batch of (state, action) pairs. Used in run_iterative.py.
       In iterative mode the caller controls the training loop.
    """

    def __init__(self, policy: PolicyNetworkMapObs, eta: float = 0.01,
                 seed: int = 0):
        """
        Parameters
        ----------
        policy : PolicyNetwork
            The learner's policy network. Weights are modified in-place.
        eta : float
            SGD learning rate. Must be the same value used in TV-BC
            for a fair comparison.
        seed : int
            Random seed for mini-batch shuffling in batch mode.
        """
        self.policy    = policy
        self.eta       = eta
        self.rng       = np.random.default_rng(seed)
        self.optimiser = optim.SGD(self.policy.parameters(), lr=eta)

        # Logging (populated during train() or step())
        self.loss_history  = []   # one entry per epoch (batch mode)
                                  # or one entry per step (iterative mode)
        self.eval_history  = []   # policy return at eval checkpoints
        self.eval_steps    = []   # iteration index of each eval checkpoint

    # ------------------------------------------------------------------ #
    #  ITERATIVE MODE — single update step                                #
    # ------------------------------------------------------------------ #

    def step(self, batch: list) -> float:
        """
        Perform one SGD update on a single mini-batch.

        This is the method called by run_iterative.py at every iteration.
        The caller is responsible for sampling the batch, running the loop,
        and calling evaluate() at the right intervals.

        Parameters
        ----------
        batch : list of (state_np, action_int) tuples
            A mini-batch of (s, a) pairs. Typically 20 pairs, freshly
            sampled from the full trajectory pool at each iteration.

        Returns
        -------
        loss : float
            Average cross-entropy loss over the mini-batch for logging.
        """
        self.policy.train()
        self.optimiser.zero_grad()

        total_loss = torch.tensor(0.0)

        for (state_np, action) in batch:
            loss       = self.policy.compute_loss(state_np, action)
            total_loss = total_loss + loss

        avg_loss = total_loss / len(batch)
        avg_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        self.optimiser.step()

        loss_val = avg_loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------ #
    #  BATCH MODE — epoch-based training over fixed dataset               #
    # ------------------------------------------------------------------ #

    def _train_one_epoch(self, states: list, actions: list,
                         batch_size: int) -> float:
        """One full pass over the dataset. Returns mean loss."""
        self.policy.train()
        N       = len(states)
        indices = self.rng.permutation(N)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            batch_idx = indices[start: start + batch_size]
            if len(batch_idx) == 0:
                continue

            self.optimiser.zero_grad()
            batch_loss = torch.tensor(0.0)

            for idx in batch_idx:
                loss       = self.policy.compute_loss(states[idx], actions[idx])
                batch_loss = batch_loss + loss

            avg = batch_loss / len(batch_idx)
            avg.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=1.0)
            self.optimiser.step()

            total_loss += avg.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

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
        states     : list of np.ndarray, each shape (64,)
        actions    : list of int
        n_epochs   : number of full passes over the dataset
        batch_size : mini-batch size
        eval_fn    : callable(policy) -> float, or None
                     If provided, called every eval_every epochs to get
                     the policy return for logging.
        eval_every : how often (in epochs) to call eval_fn
        verbose    : print progress
        """
        N = len(states)
        if verbose:
            print(f"  [BC] Training for {n_epochs} epochs on {N} samples "
                  f"(batch_size={batch_size})")

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
                    print(f"  [BC] Epoch {epoch:3d}/{n_epochs}  "
                          f"loss={loss:.4f}  |  eval={ret:.3f}")
            elif verbose and (epoch % eval_every == 0
                              or epoch == 1
                              or epoch == n_epochs):
                print(f"  [BC] Epoch {epoch:3d}/{n_epochs}  loss={loss:.4f}")

        if verbose:
            print(f"  [BC] Training complete. Final loss: "
                  f"{self.loss_history[-1]:.4f}")
