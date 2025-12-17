import sys
import os
import ctypes as C
import numpy as np
import pytest

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
# 往上跳三層: speedup -> level1 -> tests -> python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pacman_env import lib, EnvState, AgentState, OBS_DIM, OBS_DIM_ALIGNED

# 檢查 C Library 是否包含必要的函數
if not hasattr(lib, "step_env_apply_actions_batch"):
    pytest.skip("Missing step_env_apply_actions_batch in libpacman.so", allow_module_level=True)

# 強制設定 C 函數的參數型別 (Argtypes) 以防止 Segfault
# 這是 ctypes 介接 C 語言時最重要的保護機制
lib.step_env_apply_actions_batch.argtypes = [C.POINTER(EnvState), C.c_int]
lib.step_env_apply_actions_batch.restype = None

lib.step_env_apply_actions_batch_sequential.argtypes = [C.POINTER(EnvState), C.c_int]
lib.step_env_apply_actions_batch_sequential.restype = None

lib.step_env_apply_actions_sequential.argtypes = [C.POINTER(EnvState)]
lib.step_env_apply_actions_sequential.restype = None

def _set_omp_threads(n: int) -> None:
    """設定 OpenMP 執行緒數"""
    os.environ["OMP_NUM_THREADS"] = str(int(n))

def _make_wall_grid(h: int = 40, w: int = 40) -> np.ndarray:
    """建立一個中間有牆的地圖"""
    g = np.zeros((h, w), dtype=np.int8)
    mid = w // 2
    g[:, mid] = 1
    return g

# ==========================================================
# 2. 核心初始化邏輯 (HPC 記憶體佈局優化版)
# ==========================================================

def _init_states(n_envs: int, n_agents: int, seed: int, grid_np: np.ndarray | None = None):
    """
    初始化 EnvState 陣列，採用 Block Allocation 策略。
    
    解決了以下平行運算難題：
    1. Grid Race Condition: 每個環境有獨立的 Grid 記憶體複本。
    2. False Sharing: 使用大區塊分配 + Padding，強制隔離不同執行緒的寫入熱點。
    3. Random Overlap: 加大亂數索引間隔。
    """
    rng = np.random.default_rng(seed)

    # --- 準備 Grid 模板 ---
    if grid_np is None:
        grid_template = np.zeros((40, 40), dtype=np.int8)
    else:
        grid_template = np.asarray(grid_np, dtype=np.int8).copy()
    h, w = grid_template.shape

    # --- 1. 準備亂數池 (Read-Only) ---
    pool_stride = 20000  # 每個環境間隔 20000 個亂數，確保不重疊
    pool_size = max(500_000, n_envs * pool_stride)
    rand_pool_np = rng.uniform(0, 1, size=pool_size).astype(np.float32)
    rand_pool_ptr = rand_pool_np.ctypes.data_as(C.POINTER(C.c_float))

    states = (EnvState * n_envs)()
    keep_alive = []

    # --- 2. 區塊分配 (Block Allocation) ---
    # 為所有環境一次性分配大塊記憶體，並手動切分。
    # 這樣可以保證記憶體佈局受控，避免 Python malloc 造成的碎片化或偽共用。

    # [Padding Config]
    # rand_idx 是寫入最頻繁的變數，每個 int 給它 32 個 int 的空間 (128 bytes > 64 bytes Cache Line)
    PAD_INT = 32
    all_rand_idx = (C.c_int * (n_envs * PAD_INT))()

    # Obs Buffer 也要 Padding，避免不同環境的 obs 尾端和頭端在同一條 Cache Line
    obs_per_env = n_agents * OBS_DIM_ALIGNED + 64 # +64 floats padding
    all_obs = (C.c_float * (n_envs * obs_per_env))()

    # Rewards 也要 Padding
    rew_per_env = n_agents + 32
    all_rewards = (C.c_float * (n_envs * rew_per_env))()

    for e in range(n_envs):
        s = states[e]

        # --- 3. Grid 獨立複製 (關鍵修正) ---
        # 必須用 ascontiguousarray 確保傳給 C 的是連續記憶體
        env_grid_np = np.ascontiguousarray(grid_template.copy())
        env_grid_ptr = env_grid_np.ctypes.data_as(C.POINTER(C.c_int8))

        s.grid_h = h
        s.grid_w = w
        s.n_agents = n_agents
        s.grid = env_grid_ptr

        # --- 4. 分配 Struct Arrays ---
        # 這些 struct 陣列本身只有讀取 (AgentState 裡的數值才是寫入)，所以標準分配即可
        ghosts_in = (AgentState * n_agents)()
        ghosts_out = (AgentState * n_agents)()
        ghost_actions = (C.c_int * n_agents)()

        # --- 5. 連結 Block Memory (手動切分) ---
        # 計算此環境在的大陣列中的偏移量
        idx_offset = e * PAD_INT
        obs_offset = e * obs_per_env
        rew_offset = e * rew_per_env

        # 設定亂數初始值
        all_rand_idx[idx_offset] = e * pool_stride
        
        # 取得指向該區塊的指標 (使用 C.byref + C.cast)
        # 這是最底層、最精確的指標操作方式
        p_rand_idx = C.cast(C.byref(all_rand_idx, idx_offset * C.sizeof(C.c_int)), C.POINTER(C.c_int))
        p_obs = C.cast(C.byref(all_obs, obs_offset * C.sizeof(C.c_float)), C.POINTER(C.c_float))
        p_rew = C.cast(C.byref(all_rewards, rew_offset * C.sizeof(C.c_float)), C.POINTER(C.c_float))

        # --- 6. 初始化 Ghost 位置 ---
        denom = max(1, (w - 2))
        for i in range(n_agents):
            ghosts_in[i].alive = 1
            ghosts_in[i].x = 1 + (i % denom)
            ghosts_in[i].y = 1 + (i // denom)
            ghosts_out[i].alive = 1
            ghosts_out[i].x = ghosts_in[i].x
            ghosts_out[i].y = ghosts_in[i].y

        # --- 7. 綁定指標到 Struct ---
        s.ghosts_in = C.cast(ghosts_in, C.POINTER(AgentState))
        s.ghosts_out = C.cast(ghosts_out, C.POINTER(AgentState))
        s.ghost_actions = C.cast(ghost_actions, C.POINTER(C.c_int))
        
        # 綁定我們手動管理的 Block Memory
        s.ghost_rewards = p_rew
        s.obs_out = p_obs
        s.rand_idx = p_rand_idx
        
        s.rand_pool = rand_pool_ptr
        s.rand_pool_size = pool_size

        # 設定純量
        s.pacman_x_in = w // 2
        s.pacman_y_in = h // 2
        s.pacman_action = 0
        s.pacman_speed = 2
        s.pacman_x_out = s.pacman_x_in
        s.pacman_y_out = s.pacman_y_in

        # --- 8. Keep Alive (防止 Python GC 回收) ---
        keep_alive.append({
            "grid": env_grid_np,   # 每個 Grid 都要活著
            "ghosts_in": ghosts_in,
            "ghosts_out": ghosts_out,
            "ghost_actions": ghost_actions,
            # Block Memory 的 root 會在迴圈外 keep alive
        })

    # 將大區塊加入 keep_alive
    keep_alive.append({
        "BLOCK_RAND_IDX": all_rand_idx,
        "BLOCK_OBS": all_obs,
        "BLOCK_REWARDS": all_rewards,
        "RAND_POOL": rand_pool_np
    })

    return states, keep_alive, grid_template, rand_pool_np

# ==========================================================
# 3. 輔助函數 (I/O)
# ==========================================================

def _write_inputs(states, keep_alive, ghost_actions_step, pacman_actions_step, pacman_speed=2):
    n_envs = len(states)
    # keep_alive 最後一個元素是 Block Memory，不是 Env Dict，所以只遍歷到 n_envs
    n_agents = len(keep_alive[0]["ghost_actions"])
    
    for e in range(n_envs):
        act_arr = keep_alive[e]["ghost_actions"]
        for i in range(n_agents):
            act_arr[i] = int(ghost_actions_step[e, i])
        states[e].pacman_action = int(pacman_actions_step[e])
        states[e].pacman_speed = int(pacman_speed)

def _advance(states):
    n_envs = len(states)
    for e in range(n_envs):
        tmp = states[e].ghosts_in
        states[e].ghosts_in = states[e].ghosts_out
        states[e].ghosts_out = tmp
        states[e].pacman_x_in = states[e].pacman_x_out
        states[e].pacman_y_in = states[e].pacman_y_out

def _read_ghosts(ptr, n_agents):
    arr = C.cast(ptr, C.POINTER(AgentState * n_agents)).contents
    out = np.empty((n_agents, 3), dtype=np.int32)
    for i in range(n_agents):
        out[i,0], out[i,1], out[i,2] = int(arr[i].x), int(arr[i].y), int(arr[i].alive)
    return out

def _read_obs(ptr, n_agents):
    flat = np.ctypeslib.as_array(ptr, shape=(n_agents * OBS_DIM_ALIGNED,))
    full = flat.reshape((n_agents, OBS_DIM_ALIGNED))
    return full[:, :OBS_DIM].astype(np.float32).copy()

def _assert_env_equal(a_states, b_states, n_envs, n_agents):
    for e in range(n_envs):
        # 比對純量
        assert int(a_states[e].pacman_x_out) == int(b_states[e].pacman_x_out)
        assert int(a_states[e].pacman_y_out) == int(b_states[e].pacman_y_out)
        
        # 比對 Ghost 狀態
        ga = _read_ghosts(a_states[e].ghosts_out, n_agents)
        gb = _read_ghosts(b_states[e].ghosts_out, n_agents)
        np.testing.assert_array_equal(ga, gb)

        # 比對觀測值 (容許極微小浮點數誤差)
        oa = _read_obs(a_states[e].obs_out, n_agents)
        ob = _read_obs(b_states[e].obs_out, n_agents)
        np.testing.assert_allclose(oa, ob, rtol=0, atol=1e-5)

# ==========================================================
# 4. 測試案例 (Test Cases)
# ==========================================================

@pytest.mark.parametrize("n_envs", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("threads", [2, 8])
@pytest.mark.parametrize("steps", [1, 20])
def test_level2_n_envs_scaling_matches_baseline(n_envs, threads, steps):
    """測試不同環境數量下的平行一致性"""
    _set_omp_threads(threads)
    n_agents, seed = 16, 123
    
    # 初始化兩組完全一樣的狀態
    a_states, a_keep, _, _ = _init_states(n_envs, n_agents, seed)
    b_states, b_keep, _, _ = _init_states(n_envs, n_agents, seed)
    
    rng = np.random.default_rng(seed + 777)
    g_act = rng.integers(0, 5, size=(steps, n_envs, n_agents), dtype=np.int32)
    p_act = rng.integers(0, 5, size=(steps, n_envs), dtype=np.int32)

    for t in range(steps):
        _write_inputs(a_states, a_keep, g_act[t], p_act[t])
        _write_inputs(b_states, b_keep, g_act[t], p_act[t])
        
        # A組跑序列版 (Sequential Baseline)
        lib.step_env_apply_actions_batch_sequential(a_states, n_envs)
        # B組跑平行版 (Parallel Target)
        lib.step_env_apply_actions_batch(b_states, n_envs)
        
        # 比對結果
        _assert_env_equal(a_states, b_states, n_envs, n_agents)
        
        # 前進到下一步
        _advance(a_states)
        _advance(b_states)

def test_level2_wall_grid_collision_matches_baseline():
    """測試撞牆邊界條件 (這是最容易觸發 False Sharing 的案例)"""
    n_envs, n_agents, steps, threads = 8, 16, 10, 8
    seed = 1618
    _set_omp_threads(threads)
    
    # 建立有牆的地圖
    grid = _make_wall_grid(40, 40)
    a_states, a_keep, _, _ = _init_states(n_envs, n_agents, seed, grid_np=grid)
    b_states, b_keep, _, _ = _init_states(n_envs, n_agents, seed, grid_np=grid)
    
    # 手動設定 Pacman 和 Ghost 在牆邊
    mid = 40 // 2
    for e in range(n_envs):
        a_states[e].pacman_x_in = mid - 1; a_states[e].pacman_y_in = 20
        b_states[e].pacman_x_in = mid - 1; b_states[e].pacman_y_in = 20
        
        a_keep[e]["ghosts_in"][0].x = mid - 1; a_keep[e]["ghosts_in"][0].y = 19
        b_keep[e]["ghosts_in"][0].x = mid - 1; b_keep[e]["ghosts_in"][0].y = 19

    g_act = np.zeros((steps, n_envs, n_agents), dtype=np.int32)
    g_act[:, :, 0] = 4 # 往右撞牆
    p_act = np.full((steps, n_envs), 4, dtype=np.int32) # 往右撞牆

    for t in range(steps):
        _write_inputs(a_states, a_keep, g_act[t], p_act[t], pacman_speed=2)
        _write_inputs(b_states, b_keep, g_act[t], p_act[t], pacman_speed=2)
        
        lib.step_env_apply_actions_batch_sequential(a_states, n_envs)
        lib.step_env_apply_actions_batch(b_states, n_envs)
        
        _assert_env_equal(a_states, b_states, n_envs, n_agents)
        _advance(a_states)
        _advance(b_states)