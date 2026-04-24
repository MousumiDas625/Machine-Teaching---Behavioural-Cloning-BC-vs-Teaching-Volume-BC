# gridworld_env/gridworld_mapobs.py
# GridWorld with full map observation. Supports any grid size.

import numpy as np


class GridWorldMapObs:
    """
    NxN GridWorld with full map observation.
    Supports 8x8 (64 cells) and 16x16 (256 cells).

    Observation: [map_encoding(n_states) | position_onehot(n_states)]
    obs_dim = 2 * n_states = 128 (8x8) or 512 (16x16)
    """

    def __init__(self,
                 grid_size   : int   = 8,
                 max_steps   : int   = 100,
                 seed        : int   = 42,
                 n_goals     : int   = 3,
                 n_traps     : int   = 16,
                 terminal    : bool  = False,
                 step_penalty: float = 0.0):

        self.grid_size    = grid_size
        self.n_states     = grid_size * grid_size
        self.n_actions    = 4
        self.max_steps    = max_steps
        self.n_goals      = n_goals
        self.n_traps      = n_traps
        self.terminal     = terminal
        self.step_penalty = step_penalty
        self.obs_dim      = self.n_states * 2

        # Fixed cells — professor's rules
        self.START_CELL = (grid_size - 1) * grid_size   # bottom-left
        self.BEST_GOAL  = grid_size - 1                  # top-right

        self._build_map(seed)
        self.T = self._build_transition_table()

        self.current_pos = self.START_CELL
        self.steps_taken = 0

    def _build_map(self, seed: int):
        self.seed = seed
        rng       = np.random.default_rng(seed)

        reserved  = {self.START_CELL, self.BEST_GOAL}
        available = [s for s in range(self.n_states)
                     if s not in reserved]

        n_extra   = self.n_goals - 1 + self.n_traps
        # Safety check — cannot place more than available cells
        n_extra   = min(n_extra, len(available))
        chosen    = rng.choice(available, size=n_extra, replace=False)

        n_goal_extra = min(self.n_goals - 1, len(chosen))
        goal_indices = chosen[:n_goal_extra]
        trap_indices = chosen[n_goal_extra:]

        self.goal_cells = {self.BEST_GOAL: +1.0}
        for c in goal_indices:
            reward = float(rng.uniform(0.3, 0.8))
            self.goal_cells[int(c)] = round(reward, 2)

        self.trap_cells = {int(c): -1.0 for c in trap_indices}

        self.omega_star = np.zeros(self.n_states, dtype=np.float32)
        for c, r in self.goal_cells.items():
            self.omega_star[c] = r
        for c, r in self.trap_cells.items():
            self.omega_star[c] = r

        self.map_vector = self.omega_star.copy()

    def change_map(self, new_seed: int):
        self._build_map(new_seed)

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

    def reset(self, start_pos: int = None) -> int:
        if start_pos is None:
            start_pos = self.START_CELL
        self.current_pos = start_pos
        self.steps_taken = 0
        return self.current_pos

    def step(self, action: int):
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

    def get_full_obs(self, cell: int) -> np.ndarray:
        pos_vec       = np.zeros(self.n_states, dtype=np.float32)
        pos_vec[cell] = 1.0
        return np.concatenate([self.map_vector, pos_vec])

    def get_state_vector(self, cell: int) -> np.ndarray:
        return self.get_full_obs(cell)

    def print_map(self):
        mode = "TERMINAL" if self.terminal else "NON-TERMINAL"
        print(f"\n  Map (seed={self.seed}, {self.grid_size}x"
              f"{self.grid_size}, {mode}):")
        print(f"  G=goal, T=trap, S=start, .=empty")
        for row in range(self.grid_size):
            line = "  "
            for col in range(self.grid_size):
                cell = row * self.grid_size + col
                if cell == self.START_CELL:
                    line += " S "
                elif cell in self.goal_cells:
                    line += f"G{self.goal_cells[cell]:+.1f}"[:4]
                elif cell in self.trap_cells:
                    line += " T "
                else:
                    line += " . "
            print(line)
        print(f"  Goals: {len(self.goal_cells)}  "
              f"Traps: {len(self.trap_cells)}  "
              f"Cells: {self.n_states}")
        print()
