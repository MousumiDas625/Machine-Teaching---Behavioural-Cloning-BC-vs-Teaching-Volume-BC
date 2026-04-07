# gridworld_env/gridworld.py
#
# 8x8 GridWorld with configurable reward structure.
#
# Supports two modes:
#   1. Fixed hard map  (multi-goal + traps, for reproducible experiments)
#   2. Random map generation ( average over many maps)
#      Rules for random maps:
#        - Start position always bottom-left (cell 56)
#        - Best reward always top-right (cell 7)
#        - Other goals and traps placed randomly

import numpy as np


class GridWorld:
    """
    8x8 GridWorld.

    State  : integer cell index s in {0, ..., 63}
              cell s = row * 8 + col
              row 0 = top, row 7 = bottom
              col 0 = left, col 7 = right

    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

    Reward : r(s') where s' is the cell entered after action a.
             Defined by omega_star vector.

    Two ways to create the reward map:
      GridWorld(map_mode='fixed')         → fixed hard map (reproducible)
      GridWorld(map_mode='random', seed)  → random map with fixed start/goal
    """

    def __init__(self,
                 grid_size : int   = 8,
                 omega_star        = None,
                 max_steps : int   = 50,
                 seed      : int   = 42,
                 map_mode  : str   = 'fixed',
                 n_goals   : int   = 3,
                 n_traps   : int   = 8):
        """
        Parameters
        ----------
        grid_size  : grid is grid_size x grid_size
        omega_star : optional np.ndarray — if provided, use this reward
                     vector directly (overrides map_mode)
        max_steps  : episode length, no early termination
        seed       : random seed for map generation and episode resets
        map_mode   : 'fixed'  → use the hard-coded multi-goal+trap map
                     'random' → randomly generate goals and traps
                     (ignored if omega_star is provided)
        n_goals    : number of goal cells in random mode (besides top-right)
        n_traps    : number of trap cells in random mode
        """
        self.grid_size = grid_size
        self.n_states  = grid_size * grid_size
        self.n_actions = 4
        self.max_steps = max_steps
        self.seed      = seed
        self.rng       = np.random.default_rng(seed)

        # Fixed cell indices ( these never change)
        self.START_CELL    = (grid_size - 1) * grid_size   # bottom-left = 56
        self.BEST_GOAL     = grid_size - 1                  # top-right   = 7

        # Build reward vector
        if omega_star is not None:
            # Use provided reward vector directly (backward compatibility)
            self.omega_star = np.array(omega_star, dtype=np.float32)
            self.goal_cells = {}
            self.trap_cells = {}
        elif map_mode == 'fixed':
            self.omega_star, self.goal_cells, self.trap_cells = \
                self._build_fixed_map()
        elif map_mode == 'random':
            self.omega_star, self.goal_cells, self.trap_cells = \
                self._build_random_map(n_goals, n_traps)
        else:
            raise ValueError(f"map_mode must be 'fixed' or 'random', "
                             f"got '{map_mode}'")

        # Pre-compute transition table T[s, a] = s'
        self.T = self._build_transition_table()

        # Episode state
        self.current_pos = self.START_CELL
        self.steps_taken = 0

    # ------------------------------------------------------------------ #
    #  Map builders                                                        #
    # ------------------------------------------------------------------ #

    def _build_fixed_map(self):
        """
        Fixed hard map with 3 goals and 8 traps.

        Layout (G=goal, T=trap, .=empty):
        .  .  .  .  .  .  .  G1.0   row 0  ← best goal top-right
        .  .  .  T  T  .  .  .       row 1
        .  .  .  T  T  .  .  .       row 2
        .  .  .  .  .  .  .  G0.6   row 3
        .  .  .  .  .  .  .  .       row 4
        .  .  T  T  .  .  .  .       row 5
        .  .  T  T  .  .  .  .       row 6
        G0.8 .  .  .  .  .  .  .    row 7  ← start cell col 0
        """
        goal_cells = {
            7 : +1.0,   # top-right  (best goal)
            31: +0.6,   # row 3 col 7
            56: +0.8,   # bottom-left (also the start area)
        }
        trap_cells = {
            11: -1.0, 12: -1.0,   # row 1 col 3,4
            19: -1.0, 20: -1.0,   # row 2 col 3,4
            42: -1.0, 43: -1.0,   # row 5 col 2,3
            50: -1.0, 51: -1.0,   # row 6 col 2,3
        }
        omega = np.zeros(self.n_states, dtype=np.float32)
        for c, r in goal_cells.items():
            omega[c] = r
        for c, r in trap_cells.items():
            omega[c] = r
        return omega, goal_cells, trap_cells

    def _build_random_map(self, n_goals, n_traps):
        """
        Random map following professor's rules:
          - Best goal  ALWAYS at top-right  (cell 7,  reward +1.0)
          - Start cell ALWAYS bottom-left   (cell 56, reward  0.0)
          - Additional goals placed randomly with reward in [+0.3, +0.8]
          - Traps placed randomly with reward -1.0
          - Start cell and best goal cell are never overwritten

        This allows averaging results over many random maps.
        """
        reserved = {self.START_CELL, self.BEST_GOAL}
        available = [s for s in range(self.n_states)
                     if s not in reserved]
        chosen = self.rng.choice(available,
                                  size=n_goals - 1 + n_traps,
                                  replace=False)

        goal_indices = chosen[:n_goals - 1]
        trap_indices = chosen[n_goals - 1:]

        goal_cells = {self.BEST_GOAL: +1.0}
        for c in goal_indices:
            reward = float(self.rng.uniform(0.3, 0.8))
            goal_cells[int(c)] = round(reward, 2)

        trap_cells = {int(c): -1.0 for c in trap_indices}

        omega = np.zeros(self.n_states, dtype=np.float32)
        for c, r in goal_cells.items():
            omega[c] = r
        for c, r in trap_cells.items():
            omega[c] = r

        return omega, goal_cells, trap_cells

    def print_map(self):
        """Print the reward map for visual verification."""
        print(f"\n  Reward map (seed={self.seed}):")
        print(f"  G=goal, T=trap, S=start, .=empty")
        for row in range(self.grid_size):
            line = "  "
            for col in range(self.grid_size):
                cell = row * self.grid_size + col
                if cell == self.START_CELL:
                    line += "  S  "
                elif cell in self.goal_cells:
                    line += f"G{self.goal_cells[cell]:+.1f}"
                elif cell in self.trap_cells:
                    line += " -1  "
                else:
                    line += "  .  "
            print(line)
        print()

    # ------------------------------------------------------------------ #
    #  Transition table                                                    #
    # ------------------------------------------------------------------ #

    def _build_transition_table(self):
        T = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        for s in range(self.n_states):
            row = s // self.grid_size
            col = s  % self.grid_size
            T[s, 0] = (row-1)*self.grid_size + col if row > 0 else s
            T[s, 1] = (row+1)*self.grid_size + col \
                      if row < self.grid_size-1 else s
            T[s, 2] = row*self.grid_size + (col-1) if col > 0 else s
            T[s, 3] = row*self.grid_size + (col+1) \
                      if col < self.grid_size-1 else s
        return T

    # ------------------------------------------------------------------ #
    #  Episode interface                                                   #
    # ------------------------------------------------------------------ #

    def reset(self, start_pos: int = None) -> int:
        """
        Reset for a new episode.
        Default start: bottom-left cell (professor's rule).
        """
        if start_pos is None:
            start_pos = self.START_CELL
        self.current_pos = start_pos
        self.steps_taken = 0
        return self.current_pos

    def step(self, action: int):
        next_pos = int(self.T[self.current_pos, action])
        reward   = float(self.omega_star[next_pos])
        self.current_pos  = next_pos
        self.steps_taken += 1
        done = (self.steps_taken >= self.max_steps)
        return self.current_pos, reward, done, {}

    def get_state_vector(self, cell: int) -> np.ndarray:
        v       = np.zeros(self.n_states, dtype=np.float32)
        v[cell] = 1.0
        return v

    def get_all_states(self):
        return [self.get_state_vector(s) for s in range(self.n_states)]
