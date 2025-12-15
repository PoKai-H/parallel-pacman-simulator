import os
import ctypes as C
import numpy as np

# ==========================================
# 1. C Struct Definitions (必須與 common.h 完全一致)
# ==========================================

class AgentState(C.Structure):
    _fields_ = [
        ("x", C.c_int),
        ("y", C.c_int),
        ("alive", C.c_int),
    ]

class EnvState(C.Structure):
    """
    Must match the layout in csrc/common.h exactly.
    """
    _fields_ = [
        # --- Config (Read Only) ---
        ("grid_h", C.c_int),
        ("grid_w", C.c_int),
        ("n_agents", C.c_int),
        ("grid", C.POINTER(C.c_int8)),  # int8 optimization

        # --- Input State (Read Only) ---
        ("ghosts_in", C.POINTER(AgentState)),
        ("ghost_actions", C.POINTER(C.c_int)),
        
        ("pacman_x_in", C.c_int),
        ("pacman_y_in", C.c_int),
        ("pacman_action", C.c_int),
        ("pacman_speed", C.c_int),

        # --- Random Number Generation (Input) ---
        # [NEW] 對應 common.h 的 rand_pool
        ("rand_pool", C.POINTER(C.c_float)),
        ("rand_pool_size", C.c_int),
        ("rand_idx", C.POINTER(C.c_int)),

        # --- Output State (Write Only) ---
        ("ghosts_out", C.POINTER(AgentState)),
        ("pacman_x_out", C.c_int),
        ("pacman_y_out", C.c_int),
        
        ("ghost_rewards", C.POINTER(C.c_float)),
        ("pacman_reward", C.c_float),
        ("done", C.c_int),

        # --- Observation Output ---
        # [NEW] 對應 common.h 的 obs_out
        ("obs_out", C.POINTER(C.c_float)),
    ]

# ==========================================
# 2. Load Library
# ==========================================

HERE = os.path.dirname(__file__)
LIB_PATH = os.path.join(HERE, "libpacman.so")

if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(f"Library not found at {LIB_PATH}. Did you run 'make' in csrc/?")

lib = C.CDLL(LIB_PATH)

# void step_env_apply_actions_sequential(EnvState *s);
lib.step_env_apply_actions_sequential.argtypes = [C.POINTER(EnvState)]
lib.step_env_apply_actions_sequential.restype = None

# ==========================================
# 3. Python Environment Wrapper
# ==========================================

# 定義常數，對應 common.h
MAX_NEIGHBORS = 4
OBS_DIM = 2 + 3 + (MAX_NEIGHBORS * 3) # 17

class PacmanEnv:
    def __init__(self, grid: np.ndarray, n_agents: int = 14, max_steps: int = 200):
        assert grid.ndim == 2, "grid must be 2D"
        
        # Grid Optimization (int8)
        self.grid = grid.astype(np.int8).copy()
        self.h, self.w = self.grid.shape
        self.n_agents = n_agents
        self.max_steps = max_steps

        # --- Memory Buffers ---
        # Ghost states (Double buffering)
        self.ghosts_buf_1 = (AgentState * n_agents)()
        self.ghosts_buf_2 = (AgentState * n_agents)()
        self.ptr_ghosts_in = self.ghosts_buf_1
        self.ptr_ghosts_out = self.ghosts_buf_2

        # Actions & Rewards
        self.ghost_actions = np.zeros(n_agents, dtype=np.int32)
        self.ghost_rewards = np.zeros(n_agents, dtype=np.float32)

        # [NEW] Random Pool Initialization
        # 預先生成 100 萬個 [0, 1) 的浮點數給 C 使用
        pool_size = 1_000_000
        self.rand_pool = np.random.uniform(0, 1, size=pool_size).astype(np.float32)
        self.rand_pool_ptr = self.rand_pool.ctypes.data_as(C.POINTER(C.c_float))
        self.rand_idx = C.c_int(0) # Index counter

        # [NEW] Observation Buffer
        # 每個 Agent 17 個 float
        self.obs_buffer = np.zeros(n_agents * OBS_DIM, dtype=np.float32)
        self.obs_buffer_ptr = self.obs_buffer.ctypes.data_as(C.POINTER(C.c_float))

        # --- Initialize Shared C Struct ---
        self.state = EnvState()
        
        # Config
        self.state.grid_h = self.h
        self.state.grid_w = self.w
        self.state.n_agents = self.n_agents
        self.state.grid = self.grid.ctypes.data_as(C.POINTER(C.c_int8))
        
        # Pointers that don't change
        self.state.ghost_actions = self.ghost_actions.ctypes.data_as(C.POINTER(C.c_int))
        self.state.ghost_rewards = self.ghost_rewards.ctypes.data_as(C.POINTER(C.c_float))
        
        # [NEW] Pointers for RNG and Obs
        self.state.rand_pool = self.rand_pool_ptr
        self.state.rand_pool_size = pool_size
        self.state.rand_idx = C.pointer(self.rand_idx)
        self.state.obs_out = self.obs_buffer_ptr

        # Python tracking
        self.pac_x = 0
        self.pac_y = 0
        self.step_count = 0

    def reset(self):
        """Reset environment to initial state."""
        # 1. Reset Ghosts (Simple formation)
        for i in range(self.n_agents):
            self.ptr_ghosts_in[i].alive = 1
            self.ptr_ghosts_in[i].x = 1 + (i % max(1, (self.w - 2)))
            self.ptr_ghosts_in[i].y = 1 + (i // max(1, (self.w - 2)))

        # 2. Reset Pacman
        self.pac_x = self.w // 2
        self.pac_y = self.h // 2

        # 3. Reset Counters
        self.step_count = 0
        self.rand_idx.value = 0 # Reset RNG index
        
        return self._get_obs()

    def _get_obs(self):
        """
        Construct observation dictionary.
        Returns C-computed observations reshaped for Python usage.
        """
        # 1. 取得 C 算好的 Observation Vector (Zero-copy)
        # obs_buffer 是 1D array, 這裡把它 reshape 成 (N, 17)
        # 這樣 Agent i 的 observation 就是 matrix[i]
        ghost_obs_tensor = self.obs_buffer.reshape((self.n_agents, OBS_DIM)).copy()

        # 2. Pacman 位置 (給 Render 用，或者如果是全域觀測需要用到)
        pac_pos = np.array([self.pac_x, self.pac_y], dtype=np.int32)
        
        return {
            "ghost_tensor": ghost_obs_tensor, # (N, 17)
            "pacman": pac_pos
        }

    def step(self, ghost_actions: np.ndarray, pacman_action: int, pacman_speed: int = 2):
        # 1. Prepare Inputs
        self.ghost_actions[:] = ghost_actions[:] 

        # Update dynamic pointers (Double Buffering)
        self.state.ghosts_in = C.cast(self.ptr_ghosts_in, C.POINTER(AgentState))
        self.state.ghosts_out = C.cast(self.ptr_ghosts_out, C.POINTER(AgentState))
        
        # Update Scalars
        self.state.pacman_x_in = self.pac_x
        self.state.pacman_y_in = self.pac_y
        self.state.pacman_action = int(pacman_action)
        self.state.pacman_speed = int(pacman_speed)
        
        # 2. Call C Kernel
        lib.step_env_apply_actions_sequential(C.byref(self.state))

        # 3. Update Python State
        self.pac_x = self.state.pacman_x_out
        self.pac_y = self.state.pacman_y_out
        
        # Swap buffers
        self.ptr_ghosts_in, self.ptr_ghosts_out = self.ptr_ghosts_out, self.ptr_ghosts_in

        # 4. Finalize
        self.step_count += 1
        is_done = bool(self.state.done or self.step_count >= self.max_steps)

        obs = self._get_obs()
        reward = {
            "ghosts": self.ghost_rewards.copy(),
            "pacman": float(self.state.pacman_reward),
        }
        
        return obs, reward, is_done, {}