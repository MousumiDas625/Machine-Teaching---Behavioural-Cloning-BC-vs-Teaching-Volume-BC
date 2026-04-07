# gridworld_env/gridworld_terminal.py
#
# GridWorld with TERMINAL goal states.
#
# Key difference from gridworld.py:
#   - Episode ends immediately when agent enters a goal cell
#   - done=True as soon as agent steps into any cell in GOAL_CELLS
#   - Agent cannot sit at goal and accumulate reward
#   - This makes the task harder: agent must NAVIGATE to goal,
#     not just reach it and stay
#
# Reward structure:
#   Goal cells: positive reward (received upon entry, then episode ends)
#   Trap cells: negative reward (received upon entry, episode CONTINUES)
#   Empty cells: 0 reward
#   Step penalty: small negative reward at every non-terminal step
#                 encourages finding shortest path to goal
#
# Why terminal goals change the learning problem:
#   Without terminal: agent can get high return by reaching goal early
#                     and sitting there for remaining 49 steps
#   With terminal:    agent gets exactly one reward at the goal
#                     return = sum of step penalties + trap hits + goal reward
#                     agent must balance: avoid traps vs find shortest path

import numpy as np


class GridWorldTerminal:
    """
    8x8 GridWorld where episodes terminate upon reaching a goal cell.

    Actions:  0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
              (no explicit STOP action — termination is automatic)

    Terminal condition:
      done=True when agent enters any goal cell OR steps >= max_steps

    Reward:
      r = goal_reward    when entering goal cell  (and done=True)
      r = -1.0           when entering trap cell  (done stays False)
      r = step_penalty   at every other step      (default -0.01)
    """

    # Fixed map — same goals and traps as gridworld.py for comparison
    GOAL_CELLS = {
        7 : +1.0,    # top-right corner   — best goal
        31: +0.6,    # row 3 col 7        — second goal
        56: +0.8,    # bottom-left        — third goal
    }
    TRAP_CELLS = {
        11: -1.0, 12: -1.0,    # row 1 col 3,4
        19: -1.0, 20: -1.0,    # row 2 col 3,4
        42: -1.0, 43: -1.0,    # row 5 col 2,3
        50: -1.0, 51: -1.0,    # row 6 col 2,3
    }

    def __init__(self,
                 grid_size   : int   = 8,
                 omega_star          = None,
                 max_steps   : int   = 50,
                 seed        : int   = 42,
                 map_mode    : str   = 'fixed',
                 n_goals     : int   = 3,
                 n_traps     : int   = 8,
                 step_penalty: float = -0.01):
        """
        Parameters
        ----------
        grid_size    : int   — grid is grid_size x grid_size
        omega_star   : optional np.ndarray — if provided, use directly
        max_steps    : int   — max steps before forced termination
        seed         : int   — for random map generation and resets
        map_mode     : 'fixed' or 'random'
        n_goals      : number of goals in random mode (besides top-right)
        n_traps      : number of traps in random mode
        step_penalty : float — reward at every non-goal non-trap step
                               set to 0.0 for no penalty
                               set to -0.01 for shortest-path pressure
        """
        self.grid_size    = grid_size
        self.n_states     = grid_size * grid_size
        self.n_actions    = 4
        self.max_steps    = max_steps
        self.seed         = seed
        self.step_penalty = step_penalty
        self.rng          = np.random.default_rng(seed)

        # Fixed cell indices (professor's rule)
        self.START_CELL = (grid_size - 1) * grid_size   # bottom-left = 56
        self.BEST_GOAL  = grid_size - 1                  # top-right   = 7

        # Build reward vector
        if omega_star is not None:
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
            raise ValueError(f"map_mode must be 'fixed' or 'random'")

        # Pre-compute transition table
        self.T = self._build_transition_table()

        # Episode state
        self.current_pos = self.START_CELL
        self.steps_taken = 0

    # ------------------------------------------------------------------ #
    #  Map builders                                                        #
    # ------------------------------------------------------------------ #

    def _build_fixed_map(self):
        goal_cells = dict(self.GOAL_CELLS)
        trap_cells = dict(self.TRAP_CELLS)
        omega      = np.zeros(self.n_states, dtype=np.float32)
        for c, r in goal_cells.items():
            omega[c] = r
        for c, r in trap_cells.items():
            omega[c] = r
        return omega, goal_cells, trap_cells

    def _build_random_map(self, n_goals, n_traps):
        """
        Random map with professor's rules:
          - Best goal ALWAYS top-right (cell 7, reward +1.0)
          - Start ALWAYS bottom-left  (cell 56)
          - Additional goals randomly placed with reward [+0.3, +0.8]
          - Traps randomly placed with reward -1.0
        """
        reserved  = {self.START_CELL, self.BEST_GOAL}
        available = [s for s in range(self.n_states)
                     if s not in reserved]
        chosen    = self.rng.choice(available,
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
        print(f"\n  Reward map terminal (seed={self.seed}):")
        print(f"  G=goal(terminal), T=trap, S=start, .=empty")
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
        Reset for new episode. Default start = bottom-left.
        Avoids starting inside a goal or trap cell.
        """
        if start_pos is None:
            start_pos = self.START_CELL
        self.current_pos = start_pos
        self.steps_taken = 0
        return self.current_pos

    def step(self, action: int):
        """
        Take one step.

        Returns
        -------
        next_state : int
        reward     : float
        done       : bool  — True if goal reached OR max_steps hit
        info       : dict  — contains 'reached_goal' flag
        """
        next_pos = int(self.T[self.current_pos, action])
        self.steps_taken += 1

        # Check if next cell is a goal → terminal
        if next_pos in self.goal_cells:
            reward = float(self.goal_cells[next_pos])
            done   = True
            info   = {'reached_goal': True, 'goal_cell': next_pos}

        # Check if next cell is a trap → penalty but continue
        elif next_pos in self.trap_cells:
            reward = float(self.trap_cells[next_pos])
            done   = (self.steps_taken >= self.max_steps)
            info   = {'reached_goal': False, 'trap': True}

        # Empty cell → step penalty, continue
        else:
            reward = self.step_penalty
            done   = (self.steps_taken >= self.max_steps)
            info   = {'reached_goal': False}

        self.current_pos = next_pos
        return self.current_pos, reward, done, info

    def get_state_vector(self, cell: int) -> np.ndarray:
        v       = np.zeros(self.n_states, dtype=np.float32)
        v[cell] = 1.0
        return v

    def get_all_states(self):
        return [self.get_state_vector(s) for s in range(self.n_states)]
