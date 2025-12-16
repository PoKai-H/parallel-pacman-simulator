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
        ("_padding", C.c_char * 128)
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
OBS_DIM_ALIGNED = 32

class PacmanEnv:
    def __init__(self, grid: np.ndarray, n_agents: int = 16, max_steps: int = 200):
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
        self.obs_buffer = np.zeros(n_agents * OBS_DIM_ALIGNED, dtype=np.float32)
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
        關鍵修改：
        1. Reshape 成 (N, 32) 包含 padding
        2. Slice 取前 17 個有效值
        """
        # Step 1: 把 1D buffer 轉成 (N, 32)
        raw_tensor = self.obs_buffer.reshape((self.n_agents, OBS_DIM_ALIGNED))
        
        # Step 2: [關鍵] 切片！只取前 17 個欄位
        # [:, :17] 意思是：取所有 Agents，但只取 Feature 0 到 16
        # .copy() 很重要，確保回傳的是乾淨的記憶體區塊
        ghost_obs_tensor = raw_tensor[:, :OBS_DIM].copy()

        # Step 3: Pacman 位置
        pac_pos = np.array([self.pac_x, self.pac_y], dtype=np.int32)
        
        return {
            "ghost_tensor": ghost_obs_tensor, # 這裡回傳的會是 (N, 17)
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
    

class PacmanVecEnv:
    """
    Level 2 Vectorized Environment.
    Manages a batch of environments in a single contiguous memory block
    to allow OpenMP parallelization in C.
    """
    def __init__(self, grid: np.ndarray, n_envs: int = 16, n_agents: int = 16, max_steps: int = 200):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.h, self.w = grid.shape
        
        # 1. 配置連續記憶體 (Contiguous Memory Allocation)
        # 這是 Level 2 效能的關鍵，所有環境的狀態必須擠在一起
        self.env_states = (EnvState * n_envs)()
        
        # 2. 共用資源 (Grid & Random Pool)
        # 為了省記憶體，所有環境共用同一張地圖
        self.grid_np = grid.astype(np.int8).copy()
        self.grid_ptr = self.grid_np.ctypes.data_as(C.POINTER(C.c_int8))
        
        # 準備亂數池
        pool_size = 1_000_000
        self.rand_pool = np.random.uniform(0, 1, size=pool_size).astype(np.float32)
        self.rand_pool_ptr = self.rand_pool.ctypes.data_as(C.POINTER(C.c_float))
        
        # 3. 為每個環境分配內部的 Pointer
        # 這些 list 用來防止 Python Garbage Collection 回收記憶體
        self._keep_alive = []
        
        for i in range(n_envs):
            s = self.env_states[i]
            
            # Config
            s.grid_h = self.h
            s.grid_w = self.w
            s.n_agents = n_agents
            s.grid = self.grid_ptr
            s.rand_pool = self.rand_pool_ptr
            s.rand_pool_size = pool_size
            
            # Allocation
            # 每個環境還是需要自己獨立的 agents, actions, obs 記憶體
            ghosts_in = (AgentState * n_agents)()
            ghosts_out = (AgentState * n_agents)()
            ghost_actions = (C.c_int * n_agents)()
            ghost_rewards = (C.c_float * n_agents)()
            
            # 32 float aligned observation
            obs = (C.c_float * (n_agents * OBS_DIM_ALIGNED))()
            rand_idx = C.c_int(i * 100) # 錯開每個環境的亂數起點
            
            # Linking
            s.ghosts_in = C.cast(ghosts_in, C.POINTER(AgentState))
            s.ghosts_out = C.cast(ghosts_out, C.POINTER(AgentState))
            s.ghost_actions = C.cast(ghost_actions, C.POINTER(C.c_int))
            s.ghost_rewards = C.cast(ghost_rewards, C.POINTER(C.c_float))
            s.obs_out = C.cast(obs, C.POINTER(C.c_float))
            s.rand_idx = C.pointer(rand_idx)
            
            # Initialize Scalars
            s.pacman_x_in = self.w // 2
            s.pacman_y_in = self.h // 2
            s.pacman_action = 0
            s.pacman_speed = 2
            s.done = 0
            
            # Keep alive
            self._keep_alive.append((ghosts_in, ghosts_out, ghost_actions, ghost_rewards, obs, rand_idx))
            
        # 4. 建立 Numpy Views (方便 Python 讀寫，不用 copy)
        # 我們需要一個方法快速把 Action 塞進去，並把 Observation 拿出來
        # 這裡為了簡單，我們在 step 裡面做 copy，追求極致效能可以用更進階的 buffer protocol
            
    def reset(self):
        """重置所有環境"""
        for i in range(self.n_envs):
            s = self.env_states[i]
            s.pacman_x_in = self.w // 2
            s.pacman_y_in = self.h // 2
            s.done = 0
            
            # Reset Ghosts
            ghosts_in = self._keep_alive[i][0] # Retrieve from keep_alive
            for j in range(self.n_agents):
                ghosts_in[j].alive = 1
                ghosts_in[j].x = 1
                ghosts_in[j].y = 1
                
        # 雖然剛 reset，但也算一次 step 讓 C 產生初始 observation
        # 這裡偷懶直接回傳全零或是呼叫一次 C
        return self._get_batch_obs()

    def step(self, actions):
        """
        actions: shape (n_envs, n_agents) of int32
        pacman_actions: shape (n_envs,) of int32 (Optional, default 0)
        """
        # 1. 將 Python Action 填入 C Struct
        # 這是 Python overhead 最大的地方，可以用 Cython 優化，但現在先用迴圈
        for i in range(self.n_envs):
            # 取得該環境的 ghost_actions array (C type)
            c_actions = self._keep_alive[i][2] 
            # 這裡假設 actions[i] 是 numpy array
            # 用 ctypes 的 memmove 或 slice assignment 會比較快
            for j in range(self.n_agents):
                c_actions[j] = actions[i, j]
                
            # Update double buffering pointers (swap in/out)
            # 在 Batch 模式下，我們需要手動交換 struct 裡的指標
            s = self.env_states[i]
            temp = s.ghosts_in
            s.ghosts_in = s.ghosts_out
            s.ghosts_out = temp
            
            # 更新 Pacman 位置 (將上一步的 out 變成這一步的 in)
            s.pacman_x_in = s.pacman_x_out
            s.pacman_y_in = s.pacman_y_out
            
        # 2. 呼叫 C 語言核心 (Level 2 Parallelism Happens Here!)
        lib.step_env_apply_actions_batch(self.env_states, self.n_envs)
        
        # 3. 收集結果
        obs = self._get_batch_obs()
        
        # 收集 Rewards & Done
        rewards = np.zeros((self.n_envs, self.n_agents), dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        
        for i in range(self.n_envs):
            c_rewards = self._keep_alive[i][3]
            # 簡單搬運
            for j in range(self.n_agents):
                rewards[i, j] = c_rewards[j]
            dones[i] = bool(self.env_states[i].done)
            
        return obs, rewards, dones, {}

    def _get_batch_obs(self):
        """
        高效取出所有環境的 Observation
        Return: (n_envs, n_agents, 17)
        """
        batch_obs = np.zeros((self.n_envs, self.n_agents, 17), dtype=np.float32)
        
        for i in range(self.n_envs):
            # 取出 C array
            c_obs = self._keep_alive[i][4] # obs buffer
            # [修正] 1. 先轉成 1D Numpy Array
            flat_view = np.ctypeslib.as_array(c_obs)
            
            # [修正] 2. 手動 Reshape 成 (N_AGENTS, 32)
            full_view = flat_view.reshape((self.n_agents, OBS_DIM_ALIGNED))
            # 切片取前 17
            batch_obs[i] = full_view[:, :OBS_DIM]
            
        return batch_obs