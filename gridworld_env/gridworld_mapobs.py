# gridworld_env/gridworld_mapobs.py
#
# GridWorld with full map observation as input.
#
# Key difference from gridworld.py:
#   - get_full_obs(cell) returns a 128-dim vector:
#       First 64:  map encoding (goal/trap locations)
#       Last  64:  agent position (one-hot)
#   - Map can change mid-experiment via change_map(new_seed)
#   - When map changes, the map encoding in observations changes
#     automatically — agent sees the new layout immediately
#
# Why 128-dim matters:
#   With position-only (64-dim), the agent has no idea where
#   goals or traps are. It memorises actions per cell.
#   With full map obs (128-dim), the agent sees the entire
#   reward structure at every step. It can learn a general
#   policy: "move toward goal cells, avoid trap cells."
#   This generalises across map changes.

import numpy as np


class GridWorldMapObs:
    """
    8x8 GridWorld with full map observation.

    Observation space: 128-dimensional vector
      obs[:64] = map encoding  (fixed for current map)
      obs[64:] = position      (changes every step)

    Map encoding values:
      +reward_value for goal cells  (e.g. +1.0, +0.6, +0.8)
      -1.0          for trap cells
       0.0          for empty cells

    Supports map changes via change_map(new_seed).
    All other behaviour identical to GridWorld.
    """

    def __init__(self,
                 grid_size   : int   = 8,
                 max_steps   : int   = 50,
                 seed        : int   = 42,
                 n_goals     : int   = 3,
                 n_traps     : int   = 8,
                 terminal    : bool  = False,
                 step_penalty: float = 0.0):
        """
        Parameters
        ----------
        grid_size    : grid is grid_size x grid_size
        max_steps    : episode length
        seed         : random seed for map generation
        n_goals      : number of goal cells (including best goal)
        n_traps      : number of trap cells
        terminal     : if True, episode ends when agent reaches goal
        step_penalty : reward at every non-goal non-trap step
                       0.0 for no penalty (Option A)
        """
        self.grid_size    = grid_size
        self.n_states     = grid_size * grid_size   # 64
        self.n_actions    = 4
        self.max_steps    = max_steps
        self.n_goals      = n_goals
        self.n_traps      = n_traps
        self.terminal     = terminal
        self.step_penalty = step_penalty

        # obs_dim = map encoding (64) + position (64) = 128
        self.obs_dim = self.n_states * 2

        # Fixed cells — professor's rules, never change
        self.START_CELL = (grid_size - 1) * grid_size   # bottom-left = 56
        self.BEST_GOAL  = grid_size - 1                  # top-right   = 7

        # Build initial map
        self._build_map(seed)

        # Pre-compute transition table (does not change with map)
        self.T = self._build_transition_table()

        # Episode state
        self.current_pos = self.START_CELL
        self.steps_taken = 0

    # ------------------------------------------------------------------ #
    #  Map building                                                        #
    # ------------------------------------------------------------------ #

    def _build_map(self, seed: int):
        """
        Build a random map following professor's rules:
          - Best goal ALWAYS at top-right  (cell 7,  reward +1.0)
          - Start     ALWAYS at bottom-left (cell 56, no reward)
          - n_goals-1 additional goals placed randomly [+0.3, +0.8]
          - n_traps   trap cells placed randomly       [-1.0]

        Stores:
          self.seed        : current map seed
          self.omega_star  : reward vector shape (64,)
          self.goal_cells  : dict {cell: reward}
          self.trap_cells  : dict {cell: -1.0}
          self.map_vector  : 64-dim encoding for observations
        """
        self.seed = seed
        rng       = np.random.default_rng(seed)

        reserved  = {self.START_CELL, self.BEST_GOAL}
        available = [s for s in range(self.n_states)
                     if s not in reserved]
        chosen    = rng.choice(available,
                                size=self.n_goals - 1 + self.n_traps,
                                replace=False)

        goal_indices = chosen[:self.n_goals - 1]
        trap_indices = chosen[self.n_goals - 1:]

        self.goal_cells = {self.BEST_GOAL: +1.0}
        for c in goal_indices:
            reward = float(rng.uniform(0.3, 0.8))
            self.goal_cells[int(c)] = round(reward, 2)

        self.trap_cells = {int(c): -1.0 for c in trap_indices}

        # omega_star: reward vector used by teacher for planning
        self.omega_star = np.zeros(self.n_states, dtype=np.float32)
        for c, r in self.goal_cells.items():
            self.omega_star[c] = r
        for c, r in self.trap_cells.items():
            self.omega_star[c] = r

        # map_vector: 64-dim encoding included in every observation
        # Encodes the reward structure the agent can see
        self.map_vector = self.omega_star.copy()

    def change_map(self, new_seed: int):
        """
        Change the map to a new random layout.

        Called every MAP_CHANGE_EVERY rounds in the experiment.
        Updates omega_star, goal_cells, trap_cells, and map_vector.
        The transition table T is unchanged (same grid structure).
        Old trajectories in the pool keep their frozen TV scores
        and their frozen 128-dim observations from the old map.

        Parameters
        ----------
        new_seed : int — seed for the new map
        """
        self._build_map(new_seed)

    # ------------------------------------------------------------------ #
    #  Observation                                                         #
    # ------------------------------------------------------------------ #

    def get_full_obs(self, cell: int) -> np.ndarray:
        """
        Return the 128-dimensional full observation for a given cell.

        Structure:
          obs[:64] = map_vector  — encodes goal/trap positions
                                   same for every cell on this map
                                   changes when map changes
          obs[64:] = pos_vector  — one-hot of agent position
                                   changes every step

        Parameters
        ----------
        cell : int — agent's current cell index (0-63)

        Returns
        -------
        obs : np.ndarray shape (128,) dtype float32
        """
        pos_vec       = np.zeros(self.n_states, dtype=np.float32)
        pos_vec[cell] = 1.0
        return np.concatenate([self.map_vector, pos_vec])

    def get_state_vector(self, cell: int) -> np.ndarray:
        """
        Alias for get_full_obs — same interface as gridworld.py
        so existing code works without changes.
        """
        return self.get_full_obs(cell)

    # ------------------------------------------------------------------ #
    #  Transition table                                                    #
    # ------------------------------------------------------------------ #

    def _build_transition_table(self) -> np.ndarray:
        T = np.zeros((self.n_states, self.n_actions), dtype=np.int64)
        for s in range(self.n_states):
            row = s // self.grid_size
            col = s  % self.grid_size
            T[s, 0] = (row-1)*self.grid_size + col \
                      if row > 0 else s
            T[s, 1] = (row+1)*self.grid_size + col \
                      if row < self.grid_size-1 else s
            T[s, 2] = row*self.grid_size + (col-1) \
                      if col > 0 else s
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
        Returns cell index (int).
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
        next_state : int   — cell index
        reward     : float
        done       : bool
        info       : dict
        """
        next_pos = int(self.T[self.current_pos, action])
        self.steps_taken += 1

        if next_pos in self.goal_cells:
            reward = float(self.goal_cells[next_pos])
            done   = True if self.terminal \
                     else (self.steps_taken >= self.max_steps)
            info   = {'reached_goal': True, 'goal_cell': next_pos}
        elif next_pos in self.trap_cells:
            reward = float(self.trap_cells[next_pos])
            done   = (self.steps_taken >= self.max_steps)
            info   = {'reached_goal': False, 'trap': True}
        else:
            reward = self.step_penalty
            done   = (self.steps_taken >= self.max_steps)
            info   = {'reached_goal': False}

        self.current_pos = next_pos
        return self.current_pos, reward, done, info

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def print_map(self):
        """Print current map layout."""
        mode = "TERMINAL" if self.terminal else "NON-TERMINAL"
        print(f"\n  Map (seed={self.seed}, {mode}):")
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
