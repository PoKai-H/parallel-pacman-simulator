import sys
import os
import numpy as np
import ctypes as C
import copy

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
# 往上跳三層: speedup -> level1 -> tests -> python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)


from pacman_env import EnvState, AgentState, lib, OBS_DIM_ALIGNED

# 設定固定種子，確保每次初始化的地圖和亂數都一樣
np.random.seed(42)

def run_simulation(n_threads, steps=50):
    print(f"Running simulation with {n_threads} threads...")
    
    # 設定 OpenMP 環境變數 (雖然這是在 Python 內設，但對 ctypes 載入的 lib 可能無效)
    # 最好的方式是在外部設，但這裡我們假設使用者會從外部控制，或者 lib 初始化時讀取
    # 這裡僅作標記
    
    N_ENVS = 2048 # 測試 16 個環境
    N_AGENTS = 128
    
    # 1. 初始化資料 (這部分跟之前一樣，但要確保隨機值固定)
    env_states = (EnvState * N_ENVS)()
    keep_alive = []
    
    grid_np = np.zeros((40, 40), dtype=np.int8)
    grid_ptr = grid_np.ctypes.data_as(C.POINTER(C.c_int8))
    keep_alive.append(grid_np)

    final_ghost_positions = []

    for i in range(N_ENVS):
        s = env_states[i]
        s.grid_h = 40
        s.grid_w = 40
        s.n_agents = N_AGENTS
        s.grid = grid_ptr 
        
        ghosts_in = (AgentState * N_AGENTS)()
        ghosts_out = (AgentState * N_AGENTS)()
        ghost_actions = (C.c_int * N_AGENTS)()
        obs = (C.c_float * (N_AGENTS * OBS_DIM_ALIGNED))() 
        ghost_rewards = (C.c_float * N_AGENTS)() # 記得加這個！
        
        # 亂數池：用 numpy 固定種子生成
        rng_data = np.random.uniform(0, 1, 1000).astype(np.float32)
        rand_pool = rng_data.ctypes.data_as(C.POINTER(C.c_float))
        rand_idx = C.c_int(0)
        
        s.ghosts_in = C.cast(ghosts_in, C.POINTER(AgentState))
        s.ghosts_out = C.cast(ghosts_out, C.POINTER(AgentState))
        s.ghost_actions = C.cast(ghost_actions, C.POINTER(C.c_int))
        s.obs_out = C.cast(obs, C.POINTER(C.c_float))
        s.ghost_rewards = C.cast(ghost_rewards, C.POINTER(C.c_float))
        s.rand_pool = rand_pool
        s.rand_pool_size = 1000
        s.rand_idx = C.pointer(rand_idx)
        
        s.pacman_x_in = 20; s.pacman_y_in = 20; s.pacman_action = 0; s.pacman_speed = 2
        
        # 初始化 Ghost 位置 (固定)
        for g in range(N_AGENTS):
            ghosts_in[g].x = 5
            ghosts_in[g].y = 5
            ghosts_in[g].alive = 1
            
        keep_alive.append((ghosts_in, ghosts_out, ghost_actions, obs, rng_data, rand_idx, ghost_rewards))

    # 2. 執行模擬
    lib.step_env_apply_actions_batch(env_states, N_ENVS)
    
    # 3. 收集結果 (只看 Env 0 的 Ghost 0 位置作為雜湊值)
    for i in range(N_ENVS):
        # 讀取 ghosts_out 的第一個 ghost
        g = env_states[i].ghosts_out[0]
        final_ghost_positions.append((g.x, g.y))
        
    return final_ghost_positions

if __name__ == "__main__":
    # 使用環境變數控制不太準確，這裡我們用「人眼比對」方式，
    # 請使用者在 Terminal 跑兩次這個腳本，一次 OMP=1，一次 OMP=4
    res = run_simulation(n_threads="Current Env")
    print(f"Result Checksum (First 3 Envs): {res[:3]}")