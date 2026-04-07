# evaluation/metrics.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from agents.policy_net       import PolicyNetwork
from agents.rl_teacher       import RLTeacher
from gridworld_env.gridworld import GridWorld


def policy_return(policy    : PolicyNetwork,
                  env       : GridWorld,
                  n_episodes: int = 100,
                  seed      : int = 99) -> float:
    """
    Average cumulative reward over n_episodes rollouts.

    The most direct measure of how well the learner navigates
    the grid. Higher = better.
    """
    rng          = np.random.default_rng(seed)
    total_reward = 0.0
    policy.eval()

    with torch.no_grad():
        for _ in range(n_episodes):
            state     = env.reset(
                start_pos=int(rng.integers(0, env.n_states)))
            done      = False
            ep_reward = 0.0

            while not done:
                probs  = policy.get_action_probs(state).numpy()
                action = int(np.argmax(probs))
                state, reward, done, _ = env.step(action)
                ep_reward += reward

            total_reward += ep_reward

    return total_reward / n_episodes


def policy_agreement(policy  : PolicyNetwork,
                     teacher : RLTeacher,
                     env     : GridWorld) -> float:
    """
    Fraction of states where the learner's greedy action
    matches the teacher's greedy action.

    Measures how well the learner has cloned the teacher's
    decision-making, independent of reward scale.
    Higher = better. Perfect clone = 1.0.
    """
    policy.eval()
    n_agree = 0

    with torch.no_grad():
        for s in range(env.n_states):
            state_vec      = env.get_state_vector(s)
            probs          = policy.get_action_probs(state_vec).numpy()
            learner_action = int(np.argmax(probs))
            teacher_action = teacher.act_greedy(s)

            if learner_action == teacher_action:
                n_agree += 1

    return n_agree / env.n_states


def value_alignment(policy  : PolicyNetwork,
                    teacher : RLTeacher,
                    env     : GridWorld) -> float:
    """
    L2 distance between learner's implied state values
    and teacher's optimal state values V*.

    For each state s:
        V_learner(s) = sum_a pi_learner(a|s) * Q_teacher(s,a)
        V_teacher(s) = max_a Q_teacher(s,a)  =  V*(s)

    Returns ||V_learner - V_teacher||_2 / n_states (normalised).
    Lower = better. Perfect clone = 0.

    Why this metric?
    Policy return depends on the specific reward landscape.
    Value alignment measures HOW WELL the learner understands
    the value structure, independent of episode length and
    start state randomness.
    """
    policy.eval()
    v_learner = np.zeros(env.n_states)
    v_teacher = np.zeros(env.n_states)

    with torch.no_grad():
        for s in range(env.n_states):
            state_vec = env.get_state_vector(s)
            probs     = policy.get_action_probs(state_vec).numpy()

            # Teacher's optimal value at state s
            v_teacher[s] = teacher.V[s]

            # Learner's soft value: weighted average of Q values
            # under learner's policy
            v_learner[s] = float(np.dot(probs, teacher.Q_table[s]))

    diff = v_learner - v_teacher
    return float(np.sqrt(np.mean(diff ** 2)))   # RMSE


def omega_recovery(policy  : PolicyNetwork,
                   teacher : RLTeacher,
                   env     : GridWorld) -> float:
    """
    How well does the learner's policy reflect omega_star?

    We compute the learner's implied one-step reward at each state:
        r_hat(s) = sum_a pi_learner(a|s) * omega_star[T(s,a)]

    Then compare to omega_star directly:
        ||r_hat - omega_star||_2

    Lower = better.
    """
    policy.eval()
    r_hat = np.zeros(env.n_states)

    with torch.no_grad():
        for s in range(env.n_states):
            state_vec = env.get_state_vector(s)
            probs     = policy.get_action_probs(state_vec).numpy()

            # Expected next-state reward under learner's policy
            for a in range(env.n_actions):
                next_s    = int(env.T[s, a])
                r_hat[s] += probs[a] * env.omega_star[next_s]

    return float(np.linalg.norm(r_hat - env.omega_star))


def sample_efficiency_curve(
        learner_class,
        learner_kwargs : dict,
        states         : list,
        actions        : list,
        env            : GridWorld,
        fractions      : list = [0.1, 0.25, 0.5, 0.75, 1.0],
        n_epochs       : int  = 80,
        batch_size     : int  = 20,
        n_eval_eps     : int  = 50,
        seed           : int  = 0) -> dict:
    """
    Train the learner on different fractions of the training data
    and measure final policy return at each fraction.

    This is the KEY experiment for showing TV-BC's advantage —
    with less data, TV-BC should outperform BC more clearly.

    Parameters
    ----------
    learner_class  : StandardBC or TeacherAwareBC
    learner_kwargs : dict passed to learner constructor
    states         : full training states list
    actions        : full training actions list
    env            : GridWorld for evaluation
    fractions      : data fractions to test (e.g. [0.1, 0.25, 0.5, 1.0])
    n_epochs       : training epochs per fraction
    batch_size     : mini-batch size
    n_eval_eps     : episodes for policy return evaluation
    seed           : random seed

    Returns
    -------
    results : dict with keys = fractions, values = policy_return
    """
    rng     = np.random.default_rng(seed)
    results = {}
    N       = len(states)

    for frac in fractions:
        n_samples = max(batch_size, int(frac * N))

        # Sample a random subset of the training data
        idx      = rng.choice(N, size=n_samples, replace=False)
        s_subset = [states[i]  for i in idx]
        a_subset = [actions[i] for i in idx]

        # Build fresh policy with same init
        from agents.policy_net import PolicyNetwork
        policy = PolicyNetwork(state_dim=64, action_dim=4,
                               hidden_dim=64, seed=seed)
        kwargs = {**learner_kwargs, "policy": policy}
        learner = learner_class(**kwargs)

        # Train silently
        learner.train(
            states     = s_subset,
            actions    = a_subset,
            n_epochs   = n_epochs,
            batch_size = min(batch_size, n_samples),
            eval_fn    = None,
            verbose    = False
        )

        # Evaluate
        ret = policy_return(policy, env, n_episodes=n_eval_eps, seed=99)
        results[frac] = ret
        print(f"    fraction={frac:.2f}  "
              f"n_samples={n_samples:5d}  return={ret:.3f}")

    return results


def evaluate_all(policy  : PolicyNetwork,
                 teacher : RLTeacher,
                 env     : GridWorld,
                 n_eps   : int = 100) -> dict:
    """
    Run all metrics and return as a dict.
    """
    return {
        "policy_return"   : policy_return(policy, env, n_eps),
        "policy_agreement": policy_agreement(policy, teacher, env),
        "value_alignment" : value_alignment(policy, teacher, env),
        "omega_recovery"  : omega_recovery(policy, teacher, env),
    }
