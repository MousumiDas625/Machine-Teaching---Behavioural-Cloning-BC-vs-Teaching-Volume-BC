# gridworld_env/gridworld.py
#
# 8x8 GridWorld with a harder multi-goal + trap reward structure.
#
# Reward structure:
#   Goal cells  (+1.0, +0.8, +0.6): agent should navigate toward these
#   Trap cells  (-1.0):             agent must actively avoid these
#   Empty cells ( 0.0):             no reward, just traversal
#
# This is significantly harder than the single-goal version because:
#   1. Three goals with different values → agent must learn which is best
#   2. Traps scattered in the middle → naive UP/RIGHT strategies fail
#   3. Optimal path requires going around traps → longer horizon reasoning
#   4. Noisy demonstrations lead into traps → TV-BC must filter these out

import numpy as np


class GridWorld:
    """
    8x8 GridWorld with multi-goal and trap reward structure.

    State  : integer cell index  s ∈ {0, ..., 63}
              cell s = row * 8 + col
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Reward : r(s') where s' is the cell entered after taking action a

    Reward map (fixed, not random):
    ┌─────────────────────────────────────────┐
    │  0    0    0    0    0    0    0   +1.0 │ row 0  (goal at col 7)
    │  0    0    0   -1   -1    0    0    0   │ row 1  (traps at col 3,4)
    │  0    0    0   -1   -1    0    0    0   │ row 2  (traps at col 3,4)
    │  0    0    0    0    0    0    0   +0.6 │ row 3  (goal at col 7)
    │  0    0    0    0    0    0    0    0   │ row 4
    │  0    0   -1   -1    0    0    0    0   │ row 5  (traps at col 2,3)
    │  0    0   -1   -1    0    0    0    0   │ row 6  (traps at col 2,3)
    │ +0.8  0    0    0    0    0    0    0   │ row 7  (goal at col 0)
    └─────────────────────────────────────────┘

    Three viable strategies:
      A) Navigate to cell 7  (top-right,  reward +1.0) — best but traps block direct path
      B) Navigate to cell 63-7=56 (bottom-left, reward +0.8) — safer path around traps
      C) Navigate to cell 31 (mid-right,  reward +0.6) — closest goal but lower value
    """

    # Fixed reward map — same every run, no random seed needed
    GOAL_CELLS = {
        7 : +1.0,   # top-right corner    — best goal
        56: +0.8,   # bottom-left corner  — second goal
        31: +0.6,   # row 3, col 7        — third goal
    }
    TRAP_CELLS = {
        11: -1.0,   # row 1, col 3
        12: -1.0,   # row 1, col 4
        19: -1.0,   # row 2, col 3
        20: -1.0,   # row 2, col 4
        42: -1.0,   # row 5, col 2
        43: -1.0,   # row 5, col 3
        50: -1.0,   # row 6, col 2
        51: -1.0,   # row 6, col 3
    }

    def __init__(self, grid_size: int = 8, omega_star=None,
                 max_steps: int = 50, seed: int = 42):
        """
        Parameters
        ----------
        grid_size : int   — grid is grid_size × grid_size (default 8)
        omega_star: ignored — kept for API compatibility with old code.
                    Reward is now defined by GOAL_CELLS and TRAP_CELLS.
        max_steps : int   — episode length (no early termination)
        seed      : int   — for episode start position randomisation only
        """
        self.grid_size = grid_size
        self.n_states  = grid_size * grid_size   # 64
        self.n_actions = 4                        # UP DOWN LEFT RIGHT
        self.max_steps = max_steps
        self.rng       = np.random.default_rng(seed)

        # Build reward vector omega_star from fixed goal/trap map
        self.omega_star = np.zeros(self.n_states, dtype=np.float32)
        for cell, reward in self.GOAL_CELLS.items():
            self.omega_star[cell] = reward
        for cell, reward in self.TRAP_CELLS.items():
            self.omega_star[cell] = reward

        # Pre-compute transition table T[s, a] = s'
        self.T = self._build_transition_table()

        # Episode state
        self.current_pos  = 0
        self.steps_taken  = 0

        # Print reward map for verification
        self._print_reward_map()

    def _build_transition_table(self) -> np.ndarray:
        """
        T[s, a] = s' (next state after taking action a from state s).
        Boundary condition: agent stays in place if action moves off grid.
        """
        T = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        for s in range(self.n_states):
            row = s // self.grid_size
            col = s  % self.grid_size
            # UP (0): row - 1
            T[s, 0] = (row - 1) * self.grid_size + col \
                      if row > 0 else s
            # DOWN (1): row + 1
            T[s, 1] = (row + 1) * self.grid_size + col \
                      if row < self.grid_size - 1 else s
            # LEFT (2): col - 1
            T[s, 2] = row * self.grid_size + (col - 1) \
                      if col > 0 else s
            # RIGHT (3): col + 1
            T[s, 3] = row * self.grid_size + (col + 1) \
                      if col < self.grid_size - 1 else s
        return T

    def _print_reward_map(self):
        """Print the reward map so you can verify it looks correct."""
        print("\n  Reward map (G=goal, T=trap, .=empty):")
        for row in range(self.grid_size):
            line = "  "
            for col in range(self.grid_size):
                cell = row * self.grid_size + col
                if cell in self.GOAL_CELLS:
                    line += f" G{self.GOAL_CELLS[cell]:.1f}"
                elif cell in self.TRAP_CELLS:
                    line += "  -1 "
                else:
                    line += "   . "
            print(line)
        print()

    def reset(self, start_pos: int = None) -> int:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        start_pos : int or None
            If None, sample a random non-goal non-trap starting cell.
            If int, use that cell as the start.

        Returns
        -------
        state : int — starting cell index
        """
        if start_pos is None:
            # Avoid starting in goal or trap cells
            valid = [s for s in range(self.n_states)
                     if s not in self.GOAL_CELLS
                     and s not in self.TRAP_CELLS]
            start_pos = int(self.rng.choice(valid))

        self.current_pos = start_pos
        self.steps_taken = 0
        return self.current_pos

    def step(self, action: int):
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int — 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool
        info       : dict
        """
        next_pos = int(self.T[self.current_pos, action])
        reward   = float(self.omega_star[next_pos])

        self.current_pos  = next_pos
        self.steps_taken += 1
        done = (self.steps_taken >= self.max_steps)

        return self.current_pos, reward, done, {}

    def get_state_vector(self, cell: int) -> np.ndarray:
        """Return the one-hot state vector for a given cell index."""
        v      = np.zeros(self.n_states, dtype=np.float32)
        v[cell] = 1.0
        return v

    def get_all_states(self):
        """Return list of all state vectors."""
        return [self.get_state_vector(s) for s in range(self.n_states)]

    def render(self):
        """Print current grid position."""
        print(f"  Position: cell {self.current_pos}  "
              f"(row={self.current_pos // self.grid_size}, "
              f"col={self.current_pos % self.grid_size})  "
              f"steps={self.steps_taken}")
