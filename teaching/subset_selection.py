# teaching/subset_selection.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pickle


def softmax_with_beta(scores: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute softmax over scores with temperature parameter beta.

    p(i) = exp(beta * score_i) / sum_j exp(beta * score_j)

    Numerically stable: subtract max before exponentiating.

    Parameters
    ----------
    scores : np.ndarray shape (N,) — TV score per trajectory
    beta   : float — temperature
             High beta → distribution concentrates on highest scores
             Low beta  → distribution becomes more uniform
             beta=0    → perfectly uniform (random selection)

    Returns
    -------
    probs : np.ndarray shape (N,) — selection probabilities, sums to 1
    """
    # Scale by beta first
    z = beta * scores

    # Subtract max for numerical stability (prevents exp overflow)
    z = z - z.max()

    exp_z = np.exp(z)

    # Normalise to get probabilities
    probs = exp_z / exp_z.sum()

    return probs


def select_subset(trajectories : list,
                  tv_scores    : np.ndarray,
                  K            : int,
                  beta         : float = 2.0,
                  seed         : int   = 0) -> tuple:
    """
    Select K trajectories from the pool using softmax over TV scores.

    Both BC and TV-BC learners will train on this SAME selected subset.
    The only difference between them is the update rule — not the data.

    Parameters
    ----------
    trajectories : list of trajectories (each = list of (s,a) tuples)
    tv_scores    : np.ndarray shape (N,) — one TV score per trajectory
    K            : int — how many trajectories to select
    beta         : float — softmax temperature for selection
    seed         : int — for reproducibility

    Returns
    -------
    selected_trajs   : list of K selected trajectories
    selected_indices : np.ndarray shape (K,) — which indices were chosen
    probs            : np.ndarray shape (N,) — the softmax probabilities
                       (useful for analysis and plotting)
    """
    rng = np.random.default_rng(seed)
    N   = len(trajectories)
    K   = min(K, N)    # can't select more than we have

    # Compute softmax probabilities over TV scores
    probs = softmax_with_beta(tv_scores, beta)

    # Sample K indices WITHOUT replacement
    # This means each trajectory can only be selected once
    # p=probs makes higher-TV trajectories more likely to be chosen
    selected_indices = rng.choice(N, size=K, replace=False, p=probs)

    # Gather the actual trajectories
    selected_trajs = [trajectories[i] for i in selected_indices]

    # Print selection statistics
    selected_tvs = tv_scores[selected_indices]
    all_tvs      = tv_scores

    print(f"  Selected {K} out of {N} trajectories")
    print(f"  All trajectories  — mean TV: {all_tvs.mean():.5f}, "
          f"std: {all_tvs.std():.5f}")
    print(f"  Selected subset   — mean TV: {selected_tvs.mean():.5f}, "
          f"std: {selected_tvs.std():.5f}")
    print(f"  (Selected mean TV should be >= all mean TV)")

    return selected_trajs, selected_indices, probs


def flatten_trajectories(trajectories: list) -> tuple:
    """
    Convert a list of trajectories into two flat lists.

    A trajectory is a list of (state, action) tuples.
    We flatten all trajectories into one big list of (state, action) pairs
    that the learner can iterate over during training.

    Parameters
    ----------
    trajectories : list of trajectories

    Returns
    -------
    all_states  : list of np.ndarray, each shape (64,)
    all_actions : list of int
    """
    all_states  = []
    all_actions = []

    for traj in trajectories:
        for state, action in traj:
            all_states.append(state)
            all_actions.append(action)

    return all_states, all_actions


def make_minibatches(states    : list,
                     actions   : list,
                     batch_size: int,
                     rng       = None) -> list:
    """
    Shuffle and split the dataset into mini-batches.

    Called at the start of each training epoch to ensure the learner
    sees data in a different order every epoch.

    Parameters
    ----------
    states     : list of np.ndarray
    actions    : list of int
    batch_size : int — number of (s,a) pairs per batch
    rng        : np.random.Generator or None

    Returns
    -------
    batches : list of (states_batch, actions_batch) tuples
              each states_batch  is a list of np.ndarray
              each actions_batch is a list of int
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N   = len(states)
    # Shuffle indices so batches are different each epoch
    idx = rng.permutation(N)

    batches = []
    for start in range(0, N, batch_size):
        # Slice out batch_size indices
        batch_idx = idx[start : start + batch_size]

        s_batch = [states[i]  for i in batch_idx]
        a_batch = [actions[i] for i in batch_idx]

        batches.append((s_batch, a_batch))

    return batches
