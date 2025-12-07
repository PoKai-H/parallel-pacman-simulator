# python/pacman_env.py
import os
import ctypes as C
import numpy as np

# ----- C struct: must match common.h -----
class AgentState(C.Structure):
    _fields_ = [
        ("x", C.c_int),
        ("y", C.c_int),
        ("alive", C.c_int),
    ]

# ----- load shared library -----
HERE = os.path.dirname(__file__)
LIB_PATH = os.path.join(HERE, "libpacman.so")
lib = C.CDLL(LIB_PATH)

# void step_env_apply_actions_sequential(
#     int grid_h, int grid_w,
#     const int *grid,
#     int n_agents,
#     const AgentState *ghosts_in,
#     const int *ghost_actions,
#     AgentState *ghosts_out,
#     int pacman_x_in,
#     int pacman_y_in,
#     int pacman_action,
#     int pacman_speed,
#     int *pacman_x_out,
#     int *pacman_y_out,
#     float *ghost_rewards,
#     float *pacman_reward,
#     int *done
# );
lib.step_env_apply_actions_sequential.argtypes = [
    C.c_int,                        # grid_h
    C.c_int,                        # grid_w
    C.POINTER(C.c_int),             # grid
    C.c_int,                        # n_agents
    C.POINTER(AgentState),          # ghosts_in
    C.POINTER(C.c_int),             # ghost_actions
    C.POINTER(AgentState),          # ghosts_out
    C.c_int,                        # pacman_x_in
    C.c_int,                        # pacman_y_in
    C.c_int,                        # pacman_action
    C.c_int,                        # pacman_speed
    C.POINTER(C.c_int),             # pacman_x_out
    C.POINTER(C.c_int),             # pacman_y_out
    C.POINTER(C.c_float),           # ghost_rewards
    C.POINTER(C.c_float),           # pacman_reward
    C.POINTER(C.c_int),             # done
]
lib.step_env_apply_actions_sequential.restype = None



class PacmanEnv:
    """
    Minimal environment wrapper.

    - Grid: 2D numpy int32, values:
        0 = empty, 1 = wall, 2 = pellet (not used in minimal version)
    - Ghosts: n_agents with positions stored in AgentState array.
    - Pacman: single (x, y) stored as separate ints.
    - Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
    - Speeds:
        * Ghosts: implicitly speed = 1
        * Pacman: speed = pacman_speed (0,1,2), chosen in Python
    """

    def __init__(self, grid: np.ndarray, n_agents: int = 14, max_steps: int = 200):
        assert grid.ndim == 2, "grid must be 2D"
        self.grid = grid.astype(np.int32).copy()
        self.h, self.w = self.grid.shape
        self.n_agents = n_agents
        self.max_steps = max_steps

        # C pointers
        self.grid_ptr = self.grid.ctypes.data_as(C.POINTER(C.c_int))

        # ghost states (double-buffered: ghosts / ghosts_next)
        self.ghosts = (AgentState * n_agents)()
        self.ghosts_next = (AgentState * n_agents)()

        # ghost actions
        self.ghost_actions = np.zeros(n_agents, dtype=np.int32)
        self.ghost_actions_ptr = self.ghost_actions.ctypes.data_as(
            C.POINTER(C.c_int)
        )

        # rewards
        self.ghost_rewards = np.zeros(n_agents, dtype=np.float32)
        self.ghost_rewards_ptr = self.ghost_rewards.ctypes.data_as(
            C.POINTER(C.c_float)
        )
        self.pac_reward = C.c_float(0.0)

        # pacman state
        self.pac_x = C.c_int(0)
        self.pac_y = C.c_int(0)
        self.pac_x_next = C.c_int(0)
        self.pac_y_next = C.c_int(0)

        # done flag
        self.done_flag = C.c_int(0)

        self.step_count = 0

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    def reset(self):
        """
        Reset the environment to a simple, deterministic initial state.

        - Ghosts are placed in a small block starting at (1,1).
        - Pacman starts roughly at the center of the grid.
        """
        for i in range(self.n_agents):
            self.ghosts[i].alive = 1
            self.ghosts[i].x = 1 + (i % max(1, (self.w - 2)))
            self.ghosts[i].y = 1 + (i // max(1, (self.w - 2)))

        self.pac_x.value = self.w // 2
        self.pac_y.value = self.h // 2

        self.step_count = 0
        self.done_flag.value = 0

        return self._get_obs()

    def _get_obs(self):
        """
        Build a minimal observation:
        - ghosts: (n_agents, 2) array of (x, y)
        - pacman: (2,) array of (x, y)
        """
        ghosts_pos = np.empty((self.n_agents, 2), dtype=np.int32)
        for i in range(self.n_agents):
            ghosts_pos[i, 0] = self.ghosts[i].x
            ghosts_pos[i, 1] = self.ghosts[i].y

        pac_pos = np.array([self.pac_x.value, self.pac_y.value], dtype=np.int32)
        return {"ghosts": ghosts_pos, "pacman": pac_pos}

    def step(
        self,
        ghost_actions: np.ndarray,
        pacman_action: int,
        pacman_speed: int = 2,
    ):
        """
        Apply one environment step.

        Parameters
        ----------
        ghost_actions : np.ndarray, shape (n_agents,)
            Discrete actions in {0,1,2,3,4} for each ghost.
        pacman_action : int
            Discrete action in {0,1,2,3,4}.
        pacman_speed : int, default=2
            Pacman's speed in {0,1,2}. 0 = stay, 1 = move 1 cell, 2 = move up to 2 cells.

        Returns
        -------
        obs : dict
            {"ghosts": (n_agents,2), "pacman": (2,)}
        reward : dict
            {"ghosts": np.ndarray (n_agents,), "pacman": float}
        done : bool
            True if episode ended.
        info : dict
            Reserved for future use.
        """
        assert ghost_actions.shape[0] == self.n_agents
        ghost_actions = ghost_actions.astype(np.int32)
        self.ghost_actions[:] = ghost_actions

        # clamp pacman_speed to [0,2] for safety
        if pacman_speed < 0:
            pacman_speed = 0
        elif pacman_speed > 2:
            pacman_speed = 2

        self.pac_reward.value = 0.0
        self.done_flag.value = 0

        lib.step_env_apply_actions_sequential(
            self.h,
            self.w,
            self.grid_ptr,
            self.n_agents,
            self.ghosts,              # ghosts_in
            self.ghost_actions_ptr,
            self.ghosts_next,         # ghosts_out
            self.pac_x.value,
            self.pac_y.value,
            C.c_int(int(pacman_action)),
            C.c_int(int(pacman_speed)),
            C.byref(self.pac_x_next),
            C.byref(self.pac_y_next),
            self.ghost_rewards_ptr,
            C.byref(self.pac_reward),
            C.byref(self.done_flag),
        )

        # swap ghost buffers
        self.ghosts, self.ghosts_next = self.ghosts_next, self.ghosts

        # update pacman state
        self.pac_x.value = self.pac_x_next.value
        self.pac_y.value = self.pac_y_next.value

        self.step_count += 1
        done = bool(self.done_flag.value or self.step_count >= self.max_steps)

        obs = self._get_obs()
        reward = {
            "ghosts": self.ghost_rewards.copy(),
            "pacman": float(self.pac_reward.value),
        }
        info = {}

        return obs, reward, done, info