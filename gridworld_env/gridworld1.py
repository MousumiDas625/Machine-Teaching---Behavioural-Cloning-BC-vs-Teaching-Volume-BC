# env/gridworld.py

import numpy as np

# ------------------------------------------------------------------
# Action constants — we give names to the integers 0,1,2,3
# so the rest of the code is readable ("UP" not "0")
# ------------------------------------------------------------------
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
N_ACTIONS = 4


class GridWorld:
    """
    An 8x8 GridWorld environment.

    States  : integers 0..63, one per cell.
              Represented externally as 64-dim one-hot vectors.
    Actions : 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    Reward  : r(s) = omega_star[s]  — reward of the cell you ENTER.
    Episode : ends after max_steps steps (no terminal goal state).

    The true reward vector omega_star is known only by the teacher.
    The learner must recover it from demonstrations.
    """

    def __init__(self,
                 grid_size  : int   = 8,
                 omega_star         = None,
                 max_steps  : int   = 50,
                 seed       : int   = 42):
        """
        Parameters
        ----------
        grid_size  : side length of the grid (8 → 64 cells total)
        omega_star : 1D array of length grid_size², the true reward.
                     If None, we generate a random one.
        max_steps  : how many steps before the episode ends
        seed       : random seed for reproducibility
        """

        self.grid_size  = grid_size
        self.n_states   = grid_size * grid_size   # 64 for 8x8
        self.n_actions  = N_ACTIONS               # always 4
        self.max_steps  = max_steps
        self.rng        = np.random.default_rng(seed)

        # ----------------------------------------------------------
        # Build omega_star — the true reward vector.
        # Each entry is the reward for being in that cell.
        # We create a "landscape" with a few high-reward goal cells
        # so the teacher's policy has clear structure to teach.
        # ----------------------------------------------------------
        if omega_star is not None:
            # Use the one provided (e.g. loaded from disk)
            self.omega_star = np.array(omega_star, dtype=np.float32)
        else:
            # Random base rewards in [-1, 1]
            self.omega_star = self.rng.uniform(-1.0, 1.0,
                                               self.n_states).astype(np.float32)

            # Pick ~10% of cells to be high-reward "goal" regions
            n_goals  = max(1, self.n_states // 10)   # 6 goals for 8x8
            goal_idx = self.rng.choice(self.n_states, n_goals, replace=False)
            self.omega_star[goal_idx] += 2.0          # push them clearly positive

            # Normalise so all values are in [-1, 1]
            # This keeps rewards at a stable scale regardless of random draw
            max_abs = np.max(np.abs(self.omega_star))
            if max_abs > 0:
                self.omega_star /= max_abs

        # ----------------------------------------------------------
        # Pre-compute the transition table T[s, a] = s'
        # For every state s and action a, what is the next state?
        # The agent stays put if it tries to walk off the grid edge.
        # We compute this ONCE here so it's fast during rollouts.
        # ----------------------------------------------------------
        self.T = self._build_transition_table()

        # Episode tracking — updated by reset() and step()
        self.agent_pos  = 0
        self.step_count = 0

    # --------------------------------------------------------------
    # Private: build the transition table
    # --------------------------------------------------------------
    def _build_transition_table(self) -> np.ndarray:
        """
        Returns T of shape (n_states, n_actions) where T[s, a] = s'.
        Deterministic: agent stays in place at grid boundaries.
        """
        T = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

        for s in range(self.n_states):
            row, col = divmod(s, self.grid_size)
            # divmod(s, 8) gives (row, col) — e.g. divmod(9, 8) = (1, 1)

            # UP: decrease row by 1, clamp at 0
            T[s, UP]    = max(row - 1, 0)                    * self.grid_size + col
            # DOWN: increase row by 1, clamp at grid_size-1
            T[s, DOWN]  = min(row + 1, self.grid_size - 1)   * self.grid_size + col
            # LEFT: decrease col by 1, clamp at 0
            T[s, LEFT]  = row * self.grid_size + max(col - 1, 0)
            # RIGHT: increase col by 1, clamp at grid_size-1
            T[s, RIGHT] = row * self.grid_size + min(col + 1, self.grid_size - 1)

        return T

    # --------------------------------------------------------------
    # MDP interface: reset and step
    # --------------------------------------------------------------
    def reset(self, start_pos: int = None) -> np.ndarray:
        """
        Start a new episode.

        Parameters
        ----------
        start_pos : int or None.
                    If None, the agent starts at a random cell.

        Returns
        -------
        state : np.ndarray of shape (n_states,) — one-hot encoding
                of the starting cell.
        """
        if start_pos is not None:
            self.agent_pos = int(start_pos)
        else:
            self.agent_pos = int(self.rng.integers(0, self.n_states))

        self.step_count = 0
        return self._one_hot(self.agent_pos)

    def step(self, action: int):
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int in {0, 1, 2, 3}

        Returns
        -------
        next_state : np.ndarray shape (n_states,) — one-hot
        reward     : float — omega_star[next_cell]
        done       : bool  — True when max_steps is reached
        info       : dict  — debug info (current cell index)
        """
        assert 0 <= action < self.n_actions, f"Invalid action: {action}"

        # Look up next state from pre-computed table
        next_pos        = int(self.T[self.agent_pos, action])
        self.agent_pos  = next_pos
        self.step_count += 1

        reward = float(self.omega_star[next_pos])
        done   = (self.step_count >= self.max_steps)

        return self._one_hot(next_pos), reward, done, {"pos": next_pos}

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _one_hot(self, pos: int) -> np.ndarray:
        """Return 64-dim one-hot vector for cell index pos."""
        v       = np.zeros(self.n_states, dtype=np.float32)
        v[pos]  = 1.0
        return v

    def get_state_vector(self, pos: int = None) -> np.ndarray:
        """Public version of _one_hot. Defaults to current agent position."""
        return self._one_hot(pos if pos is not None else self.agent_pos)

    def get_all_states(self) -> np.ndarray:
        """
        Return all 64 one-hot state vectors stacked into a matrix.
        Shape: (n_states, n_states) = (64, 64).
        Useful for evaluating the policy across all states at once.
        """
        return np.eye(self.n_states, dtype=np.float32)

    # --------------------------------------------------------------
    # A simple text render for debugging
    # --------------------------------------------------------------
    def render(self) -> str:
        """
        Print the grid with the agent's position marked as [A]
        and each cell's reward shown.
        """
        rewards = self.omega_star.reshape(self.grid_size, self.grid_size)
        lines   = []
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                cell = r * self.grid_size + c
                if cell == self.agent_pos:
                    row_str += "  [A] "
                else:
                    row_str += f"{rewards[r, c]:+.2f} "
            lines.append(row_str)
        return "\n".join(lines)
