# teaching/teaching_volume.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from agents.policy_net import PolicyNetwork
from agents.rl_teacher import RLTeacher


def compute_tv_single(state_np  : np.ndarray,
                      action    : int,
                      learner   : PolicyNetwork,
                      teacher   : RLTeacher,
                      eta       : float) -> float:
    """
    Compute Teaching Volume for ONE (state, action) pair.

    TV(s, a | θ) = -η² · ‖∇_θ ℓ(θ; s, a)‖²
                 + 2η  · [ℓ(θ; s, a) − ℓ(ω*; s, a)]

    Parameters
    ----------
    state_np  : np.ndarray shape (64,) — one-hot state vector
    action    : int in {0,1,2,3}
    learner   : PolicyNetwork — current learner policy (theta)
    teacher   : RLTeacher     — trained teacher (omega*)
    eta       : float         — learning rate used in TV formula

    Returns
    -------
    tv : float — teaching volume score for this (s,a) pair
    """

    # ----------------------------------------------------------
    # Term 1: -η² · ‖∇_θ ℓ(θ; s, a)‖²
    # ----------------------------------------------------------
    # compute_grad_norm_sq does:
    #   1. forward pass
    #   2. compute -log pi_theta(a|s)
    #   3. backward pass
    #   4. return sum of squared gradients
    grad_norm_sq = learner.compute_grad_norm_sq(state_np, action)
    term1 = -(eta ** 2) * grad_norm_sq

    # ----------------------------------------------------------
    # Term 2: 2η · [ℓ(θ; s, a) − ℓ(ω*; s, a)]
    # ----------------------------------------------------------
    # Learner loss: -log pi_theta(a|s)
    # We use torch.no_grad() here because we only need the VALUE
    # of the loss, not the gradient (we already computed that above)
    with torch.no_grad():
        learner_loss = learner.compute_loss(state_np, action).item()

    # Teacher loss: -log pi_teacher(a|s)
    # state_idx = index of the hot cell in the one-hot vector
    state_idx    = int(np.argmax(state_np))
    teacher_loss = teacher.loss_at(state_idx, action)

    term2 = 2.0 * eta * (learner_loss - teacher_loss)

    tv = term1 + term2
    return tv


def compute_tv_batch(states  : list,
                     actions : list,
                     learner : PolicyNetwork,
                     teacher : RLTeacher,
                     eta     : float) -> np.ndarray:
    """
    Compute TV for every (state, action) pair in a mini-batch.

    Parameters
    ----------
    states  : list of np.ndarray, each shape (64,)
    actions : list of int
    learner : PolicyNetwork
    teacher : RLTeacher
    eta     : float

    Returns
    -------
    tv_scores : np.ndarray shape (batch_size,)
                One TV score per (s,a) pair
    """
    tv_scores = np.array([
        compute_tv_single(s, a, learner, teacher, eta)
        for s, a in zip(states, actions)
    ], dtype=np.float64)

    return tv_scores


def compute_trajectory_tv(trajectory  : list,
                          learner     : PolicyNetwork,
                          teacher     : RLTeacher,
                          eta         : float,
                          aggregation : str = "mean") -> float:
    """
    Aggregate TV over all (s,a) pairs in a single trajectory.

    This gives ONE score per trajectory — used for subset selection.

    Parameters
    ----------
    trajectory  : list of (state_np, action) tuples — one full episode
    learner     : PolicyNetwork
    teacher     : RLTeacher
    eta         : float
    aggregation : "mean" → average TV across all steps
                  "sum"  → total TV across all steps

    Returns
    -------
    traj_tv : float — trajectory-level teaching volume
    """
    if len(trajectory) == 0:
        return 0.0

    states  = [pair[0] for pair in trajectory]
    actions = [pair[1] for pair in trajectory]

    # Get TV for every step in the trajectory
    step_tvs = compute_tv_batch(states, actions, learner, teacher, eta)

    if aggregation == "sum":
        return float(step_tvs.sum())
    else:
        return float(step_tvs.mean())


def compute_all_trajectory_tvs(trajectories : list,
                                learner      : PolicyNetwork,
                                teacher      : RLTeacher,
                                eta          : float,
                                aggregation  : str = "mean") -> np.ndarray:
    """
    Compute one TV score for every trajectory in the pool.

    This is called ONCE before subset selection to score all 300 trajectories.

    Parameters
    ----------
    trajectories : list of trajectories (each = list of (s,a) tuples)
    learner      : PolicyNetwork — the INITIAL learner (random weights)
    teacher      : RLTeacher
    eta          : float
    aggregation  : "mean" or "sum"

    Returns
    -------
    tv_scores : np.ndarray shape (n_trajectories,)
                One score per trajectory. Higher = more informative.
    """
    n = len(trajectories)
    print(f"  Computing TV for {n} trajectories...")

    tv_scores = np.zeros(n, dtype=np.float64)

    for i, traj in enumerate(trajectories):
        tv_scores[i] = compute_trajectory_tv(
            traj, learner, teacher, eta, aggregation
        )
        # Progress every 50 trajectories
        if (i + 1) % 50 == 0:
            print(f"    Scored {i+1}/{n} trajectories... "
                  f"(running mean TV: {tv_scores[:i+1].mean():.5f})")

    print(f"  Done. TV range: [{tv_scores.min():.5f}, "
          f"{tv_scores.max():.5f}], mean: {tv_scores.mean():.5f}")

    return tv_scores
