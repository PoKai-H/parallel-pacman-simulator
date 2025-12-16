import sys
import os
import time
import numpy as np
import ctypes as C

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
# 往上跳三層: speedup -> level1 -> tests -> python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

# 匯入所有必要的結構
from pacman_env import PacmanEnv, EnvState, AgentState, lib, OBS_DIM_ALIGNED

def test_batch_throughput():
    # --- 設定參數 ---
    N_ENVS = 128        # Batch Size
    STEPS = 100        
    N_AGENTS = 4096    
    
    print(f"=== Level 2 Benchmark: {N_ENVS} Environments x {STEPS} Steps ===")
    
    env_states = (EnvState * N_ENVS)()
    
    grid_np = np.zeros((200, 200), dtype=np.int8)
    grid_ptr = grid_np.ctypes.data_as(C.POINTER(C.c_int8))
    
    keep_alive = [] 
    keep_alive.append(grid_np)

    for i in range(N_ENVS):
        s = env_states[i]
        
        # 1. Config
        s.grid_h = 40
        s.grid_w = 40
        s.n_agents = N_AGENTS
        s.grid = grid_ptr 
        
        # 2. Memory Allocation
        ghosts_in = (AgentState * N_AGENTS)()
        ghosts_out = (AgentState * N_AGENTS)()
        ghost_actions = (C.c_int * N_AGENTS)()
        obs = (C.c_float * (N_AGENTS * OBS_DIM_ALIGNED))() 
        rand_pool = (C.c_float * 1000)()
        rand_idx = C.c_int(0)

        # [修正點] 補上 ghost_rewards 的記憶體配置！
        ghost_rewards = (C.c_float * N_AGENTS)()
        
        # 3. Pointer Assignment
        s.ghosts_in = C.cast(ghosts_in, C.POINTER(AgentState))
        s.ghosts_out = C.cast(ghosts_out, C.POINTER(AgentState))
        s.ghost_actions = C.cast(ghost_actions, C.POINTER(C.c_int))
        s.obs_out = C.cast(obs, C.POINTER(C.c_float))
        
        # [修正點] 指派 ghost_rewards 指標
        s.ghost_rewards = C.cast(ghost_rewards, C.POINTER(C.c_float))

        s.rand_pool = C.cast(rand_pool, C.POINTER(C.c_float))
        s.rand_pool_size = 1000
        s.rand_idx = C.pointer(rand_idx)
        
        # 4. Scalars
        s.pacman_x_in = 20
        s.pacman_y_in = 20
        s.pacman_action = 0
        s.pacman_speed = 2
        
        # 加入 keep_alive 防止 GC
        keep_alive.append((ghosts_in, ghosts_out, ghost_actions, obs, rand_pool, rand_idx, ghost_rewards))

    # 2. 執行測試
    start_time = time.time()
    
    for _ in range(STEPS):
        lib.step_env_apply_actions_batch(env_states, N_ENVS)
        
    end_time = time.time()
    duration = end_time - start_time
    
    total_ops = N_ENVS * STEPS
    throughput = total_ops / duration
    
    print(f"Total Time: {duration:.4f} s")
    print(f"System Throughput: {throughput:.2f} EnvSteps/sec")

if __name__ == "__main__":
    test_batch_throughput()