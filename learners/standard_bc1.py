# learners/standard_bc.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.optim as optim
from agents.policy_net         import PolicyNetwork
from teaching.subset_selection import make_minibatches


class StandardBC:
    """
    Standard Behavioral Cloning learner — the baseline.

    Treats demonstrations as i.i.d. (state, action) pairs and
    minimises the cross-entropy loss:

        l(theta; s, a) = -log pi_theta(a | s)

    Has NO awareness that the teacher chose these demonstrations
    intentionally. Just learns from the literal actions shown.

    When beta=0 in TV-BC, the ITAL update reduces exactly to this.

    Parameters
    ----------
    policy     : PolicyNetwork — the learner's MLP (theta)
    eta        : float         — learning rate
    seed       : int
    """

    def __init__(self,
                 policy : PolicyNetwork,
                 eta    : float = 0.01,
                 seed   : int   = 7):

        self.policy = policy
        self.eta    = eta
        self.rng    = np.random.default_rng(seed)

        # Use Adam optimiser — more stable than plain SGD
        # Adam adapts the learning rate per parameter using
        # moving averages of gradients and squared gradients
        self.optimiser = optim.SGD(self.policy.parameters(), lr=eta)

        # Training history — filled during train()
        self.loss_history   = []   # average loss per epoch
        self.eval_history   = []   # evaluation metric per checkpoint
        self.eval_epochs    = []   # which epochs were evaluated

    def _train_one_epoch(self,
                         states    : list,
                         actions   : list,
                         batch_size: int) -> float:
        """
        Run one full pass over the training data.

        Shuffles data into mini-batches, computes loss and gradient
        for each batch, updates parameters.

        Returns
        -------
        avg_loss : float — average loss across all mini-batches this epoch
        """
        # Put network in training mode
        # (enables dropout/batchnorm if present — not used here
        #  but good practice to always set this)
        self.policy.train()

        batches    = make_minibatches(states, actions, batch_size, self.rng)
        total_loss = 0.0
        n_batches  = 0

        for s_batch, a_batch in batches:
            # Zero gradients before each batch
            # (PyTorch accumulates gradients by default)
            self.optimiser.zero_grad()

            batch_loss = torch.tensor(0.0, requires_grad=True)

            for s, a in zip(s_batch, a_batch):
                # Compute loss for this single (s, a) pair
                loss = self.policy.compute_loss(s, a)

                # Accumulate loss across the batch
                # We'll average at the end
                batch_loss = batch_loss + loss

            # Average over batch size
            batch_loss = batch_loss / len(s_batch)

            # Backpropagate through the averaged batch loss
            batch_loss.backward()

            # Gradient clipping — prevents exploding gradients
            # Clips gradient norm to at most 1.0
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

            # Update parameters:  theta <- theta - eta * grad
            self.optimiser.step()

            total_loss += batch_loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def train(self,
              states     : list,
              actions    : list,
              n_epochs   : int,
              batch_size : int  = 20,
              eval_fn          = None,
              eval_every : int  = 10,
              verbose    : bool = True) -> "StandardBC":
        """
        Full training loop.

        Parameters
        ----------
        states     : list of np.ndarray shape (64,) — training states
        actions    : list of int — training actions
        n_epochs   : int — number of passes over the full dataset
        batch_size : int — mini-batch size
        eval_fn    : callable(policy) → float, optional
                     Called every eval_every epochs.
                     Should return a scalar metric (e.g. policy return).
        eval_every : int — how often to call eval_fn
        verbose    : bool — print progress

        Returns
        -------
        self (for chaining)
        """
        print(f"  [BC] Training for {n_epochs} epochs on "
              f"{len(states)} samples (batch_size={batch_size})")

        for epoch in range(n_epochs):

            # One full pass over the data
            avg_loss = self._train_one_epoch(states, actions, batch_size)
            self.loss_history.append(avg_loss)

            # Periodic evaluation
            if eval_fn is not None and (epoch % eval_every == 0
                                         or epoch == n_epochs - 1):
                # Switch to eval mode for consistent behaviour
                self.policy.eval()
                score = eval_fn(self.policy)
                self.eval_history.append(score)
                self.eval_epochs.append(epoch)
                self.policy.train()

            # Print progress
            if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
                eval_str = ""
                if self.eval_history:
                    eval_str = f"  |  eval={self.eval_history[-1]:.3f}"
                print(f"  [BC] Epoch {epoch+1:3d}/{n_epochs}  "
                      f"loss={avg_loss:.4f}{eval_str}")

        print(f"  [BC] Training complete. "
              f"Final loss: {self.loss_history[-1]:.4f}")
        return self
