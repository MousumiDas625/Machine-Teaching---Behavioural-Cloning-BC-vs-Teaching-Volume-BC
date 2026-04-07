# agents/rl_teacher.py

import numpy as np
import torch

class RLTeacher:
    """
    Teacher agent that uses Value Iteration to find the optimal
    policy for the GridWorld.

    The teacher knows omega_star (true reward) and uses it to compute:
      - V*(s)      : optimal state values
      - Q(s,a)     : optimal action values  
      - pi(a|s)    : Boltzmann policy over actions
      - loss_at()  : reference loss l(omega*; s,a) used in TV formula

    Parameters
    ----------
    omega_star   : np.ndarray shape (64,) — true reward vector
    grid_size    : int, default 8
    gamma        : float, discount factor (0.99)
    conv_thresh  : float, convergence threshold for Value Iteration
    softmax_temp : float, temperature tau for Boltzmann policy
    seed         : int
    """

    def __init__(self,
                 omega_star,
                 grid_size    : int   = 8,
                 gamma        : float = 0.99,
                 conv_thresh  : float = 1e-6,
                 softmax_temp : float = 0.1,
                 seed         : int   = 42):

        self.omega_star   = np.array(omega_star, dtype=np.float32)
        self.grid_size    = grid_size
        self.n_states     = grid_size * grid_size   # 64
        self.n_actions    = 4
        self.gamma        = gamma
        self.conv_thresh  = conv_thresh
        self.softmax_temp = softmax_temp
        self.rng          = np.random.default_rng(seed)

        # These get filled in by train()
        self.V       = None   # shape (64,)  — state values
        self.Q_table = None   # shape (64,4) — action values
        self._trained = False

        # Pre-compute transition table T[s,a] = s'
        # Identical logic to GridWorld — we need it here too
        # so the teacher can do planning without calling env.step()
        self.T = self._build_transition_table()

    def _build_transition_table(self) -> np.ndarray:
        """
        T[s, a] = s'  for all states and actions.
        Agent stays put at grid boundaries.
        """
        # Action indices
        UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
        T = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

        for s in range(self.n_states):
            row, col = divmod(s, self.grid_size)

            T[s, UP]    = max(row-1, 0)                   * self.grid_size + col
            T[s, DOWN]  = min(row+1, self.grid_size-1)    * self.grid_size + col
            T[s, LEFT]  = row * self.grid_size + max(col-1, 0)
            T[s, RIGHT] = row * self.grid_size + min(col+1, self.grid_size-1)

        return T

    def train(self, verbose: bool = True) -> "RLTeacher":
        """
        Run Value Iteration until convergence.

        After this call:
            self.V       is filled with V*(s) for all s
            self.Q_table is filled with Q*(s,a) for all s,a
            self._trained = True
        """
        # Start with all zeros — our initial guess for V*(s)
        V      = np.zeros(self.n_states, dtype=np.float64)
        n_iter = 0

        while True:
            delta = 0.0  # tracks the biggest change this iteration

            # --------------------------------------------------
            # Vectorised update — compute all Q values at once
            # instead of looping over states one by one.
            # 
            # Q_new[s, a] = omega_star[T[s,a]] + gamma * V[T[s,a]]
            #
            # self.T has shape (64, 4)
            # self.T.ravel() flattens to 256 indices
            # V[self.T] has shape (64, 4) — V of each next state
            # omega_star[self.T] has shape (64, 4) — reward of each next state
            # --------------------------------------------------
            next_states = self.T                          # shape (64, 4)
            Q_new = (self.omega_star[next_states]         # reward on arrival
                     + self.gamma * V[next_states])       # discounted future value

            # Best action value at each state
            V_new = Q_new.max(axis=1)                     # shape (64,)

            # Check convergence — max change across all states
            delta  = np.max(np.abs(V_new - V))
            V      = V_new
            n_iter += 1

            if delta < self.conv_thresh:
                break

        # Store final values and Q-table
        self.V       = V
        self.Q_table = (self.omega_star[self.T]
                        + self.gamma * V[self.T]).astype(np.float32)
        self._trained = True

        if verbose:
            print(f"  [Teacher] Value Iteration converged in {n_iter} iterations")
            print(f"  [Teacher] V* range: [{V.min():.3f}, {V.max():.3f}]")

        return self

    def get_action_probs(self, state_idx: int) -> np.ndarray:
        """
        Compute Boltzmann policy probabilities for one state.

        pi(a|s) = softmax(Q(s,a) / tau)

        Parameters
        ----------
        state_idx : int, which cell the agent is in

        Returns
        -------
        probs : np.ndarray shape (4,) — probability of each action
        """
        assert self._trained, "Call .train() first"

        # Q values for this state, scaled by temperature
        q = self.Q_table[state_idx] / self.softmax_temp

        # Numerically stable softmax:
        # subtract max before exp to prevent overflow
        # e.g. exp(1000) overflows, but exp(1000-1000)=exp(0)=1 is fine
        q = q - q.max()
        exp_q = np.exp(q)
        return exp_q / exp_q.sum()

    def act(self, state_idx: int,
            noisy: bool = False,
            eps: float = 0.0) -> int:
        """
        Sample one action from the teacher's policy.

        Parameters
        ----------
        state_idx : int — current cell index
        noisy     : bool — if True, apply epsilon-greedy noise
        eps       : float — probability of random action (noise level)

        Returns
        -------
        action : int in {0,1,2,3}
        """
        assert self._trained, "Call .train() first"

        # Epsilon-greedy: with probability eps, pick random action
        if noisy and self.rng.random() < eps:
            return int(self.rng.integers(0, self.n_actions))

        # Otherwise sample from Boltzmann policy
        probs = self.get_action_probs(state_idx)
        return int(self.rng.choice(self.n_actions, p=probs))

    def act_greedy(self, state_idx: int) -> int:
        """
        Return the single best action (argmax of Q).
        Used during evaluation only — not during demonstration collection.
        """
        assert self._trained
        return int(np.argmax(self.Q_table[state_idx]))

    def loss_at(self, state_idx: int, action: int) -> float:
        """
        Compute l(omega*; s, a) = -log pi_teacher(a|s)

        This is the TEACHER'S OWN loss on a (state, action) pair.
        It's used as the reference point in the Teaching Volume formula:

            TV = -eta^2 * ||grad||^2  +  2*eta * [l(theta; s,a) - l(omega*; s,a)]
                                                   ^^^learner^^^    ^^^teacher^^^

        When l(theta; s,a) >> l(omega*; s,a):
            The learner is much worse than the teacher on this example
            → high TV → this example is very useful to learn from

        When l(theta; s,a) ≈ l(omega*; s,a):
            The learner is already as good as the teacher here
            → low TV → this example teaches nothing new

        Parameters
        ----------
        state_idx : int
        action    : int

        Returns
        -------
        loss : float — always >= 0
        """
        assert self._trained
        probs = self.get_action_probs(state_idx)
        # Clip to avoid log(0)
        return -float(np.log(probs[action] + 1e-12))
