import sys
import os
import ctypes as C
import numpy as np
import pytest


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pacman_env import lib, EnvState, AgentState, OBS_DIM, OBS_DIM_ALIGNED

if not hasattr(lib, "step_env_apply_actions_batch"):
    pytest.skip("Missing step_env_apply_actions_batch in libpacman.so", allow_module_level=True)

# Preventing Segfault
lib.step_env_apply_actions_batch.argtypes = [C.POINTER(EnvState), C.c_int]
lib.step_env_apply_actions_batch.restype = None

lib.step_env_apply_actions_batch_sequential.argtypes = [C.POINTER(EnvState), C.c_int]
lib.step_env_apply_actions_batch_sequential.restype = None

lib.step_env_apply_actions_sequential.argtypes = [C.POINTER(EnvState)]
lib.step_env_apply_actions_sequential.restype = None

def _set_omp_threads(n: int) -> None:
    """ OpenMP threads"""
    os.environ["OMP_NUM_THREADS"] = str(int(n))

def _make_wall_grid(h: int = 40, w: int = 40) -> np.ndarray:
    """ A map with walls in the middle"""
    g = np.zeros((h, w), dtype=np.int8)
    mid = w // 2
    g[:, mid] = 1
    return g



def _init_states(n_envs: int, n_agents: int, seed: int, grid_np: np.ndarray | None = None):
    """
    Block Allocation Strategy
    
    1. Grid Race Condition
    2. False Sharing
    3. Random Overlap
    """
    rng = np.random.default_rng(seed)

    
    if grid_np is None:
        grid_template = np.zeros((40, 40), dtype=np.int8)
    else:
        grid_template = np.asarray(grid_np, dtype=np.int8).copy()
    h, w = grid_template.shape

    
    pool_stride = 20000  
    pool_size = max(500_000, n_envs * pool_stride)
    rand_pool_np = rng.uniform(0, 1, size=pool_size).astype(np.float32)
    rand_pool_ptr = rand_pool_np.ctypes.data_as(C.POINTER(C.c_float))

    states = (EnvState * n_envs)()
    keep_alive = []

    # --- 2. 區塊分配 (Block Allocation) ---
    # manual splitting memory space to prevent false sharing

    # [Padding Config]
    PAD_INT = 32
    all_rand_idx = (C.c_int * (n_envs * PAD_INT))()

    # Obs Buffer Padding，
    obs_per_env = n_agents * OBS_DIM_ALIGNED + 64 # +64 floats padding
    all_obs = (C.c_float * (n_envs * obs_per_env))()

    # Rewards Padding
    rew_per_env = n_agents + 32
    all_rewards = (C.c_float * (n_envs * rew_per_env))()

    for e in range(n_envs):
        s = states[e]

        # --- 3. Grid Copying Indivdually ---
        env_grid_np = np.ascontiguousarray(grid_template.copy())
        env_grid_ptr = env_grid_np.ctypes.data_as(C.POINTER(C.c_int8))

        s.grid_h = h
        s.grid_w = w
        s.n_agents = n_agents
        s.grid = env_grid_ptr

        # --- 4. Assign Struct Arrays ---
        ghosts_in = (AgentState * n_agents)()
        ghosts_out = (AgentState * n_agents)()
        ghost_actions = (C.c_int * n_agents)()

        # --- 5. Connecting Block Memory (Manual Splitting) ---
        idx_offset = e * PAD_INT
        obs_offset = e * obs_per_env
        rew_offset = e * rew_per_env

        all_rand_idx[idx_offset] = e * pool_stride
        
        p_rand_idx = C.cast(C.byref(all_rand_idx, idx_offset * C.sizeof(C.c_int)), C.POINTER(C.c_int))
        p_obs = C.cast(C.byref(all_obs, obs_offset * C.sizeof(C.c_float)), C.POINTER(C.c_float))
        p_rew = C.cast(C.byref(all_rewards, rew_offset * C.sizeof(C.c_float)), C.POINTER(C.c_float))

        # --- 6. Ghost index initializing ---
        denom = max(1, (w - 2))
        for i in range(n_agents):
            ghosts_in[i].alive = 1
            ghosts_in[i].x = 1 + (i % denom)
            ghosts_in[i].y = 1 + (i // denom)
            ghosts_out[i].alive = 1
            ghosts_out[i].x = ghosts_in[i].x
            ghosts_out[i].y = ghosts_in[i].y

        # --- 7. pointer to Struct ---
        s.ghosts_in = C.cast(ghosts_in, C.POINTER(AgentState))
        s.ghosts_out = C.cast(ghosts_out, C.POINTER(AgentState))
        s.ghost_actions = C.cast(ghost_actions, C.POINTER(C.c_int))
        
    
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

        # --- 8. Keep Alive  ---
        keep_alive.append({
            "grid": env_grid_np,   
            "ghosts_in": ghosts_in,
            "ghosts_out": ghosts_out,
            "ghost_actions": ghost_actions,
            
        })


    keep_alive.append({
        "BLOCK_RAND_IDX": all_rand_idx,
        "BLOCK_OBS": all_obs,
        "BLOCK_REWARDS": all_rewards,
        "RAND_POOL": rand_pool_np
    })

    return states, keep_alive, grid_template, rand_pool_np

# ==========================================================
# 3. (I/O)
# ==========================================================

def _write_inputs(states, keep_alive, ghost_actions_step, pacman_actions_step, pacman_speed=2):
    n_envs = len(states)
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
        
        assert int(a_states[e].pacman_x_out) == int(b_states[e].pacman_x_out)
        assert int(a_states[e].pacman_y_out) == int(b_states[e].pacman_y_out)
        
       
        ga = _read_ghosts(a_states[e].ghosts_out, n_agents)
        gb = _read_ghosts(b_states[e].ghosts_out, n_agents)
        np.testing.assert_array_equal(ga, gb)

        
        oa = _read_obs(a_states[e].obs_out, n_agents)
        ob = _read_obs(b_states[e].obs_out, n_agents)
        np.testing.assert_allclose(oa, ob, rtol=0, atol=1e-5)

# ==========================================================
# Test Cases
# ==========================================================

@pytest.mark.parametrize("n_envs", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("threads", [2, 8])
@pytest.mark.parametrize("steps", [1, 20])
def test_level2_n_envs_scaling_matches_baseline(n_envs, threads, steps):

    _set_omp_threads(threads)
    n_agents, seed = 16, 123
    
   
    a_states, a_keep, _, _ = _init_states(n_envs, n_agents, seed)
    b_states, b_keep, _, _ = _init_states(n_envs, n_agents, seed)
    
    rng = np.random.default_rng(seed + 777)
    g_act = rng.integers(0, 5, size=(steps, n_envs, n_agents), dtype=np.int32)
    p_act = rng.integers(0, 5, size=(steps, n_envs), dtype=np.int32)

    for t in range(steps):
        _write_inputs(a_states, a_keep, g_act[t], p_act[t])
        _write_inputs(b_states, b_keep, g_act[t], p_act[t])
        
        # A (Sequential Baseline)
        lib.step_env_apply_actions_batch_sequential(a_states, n_envs)
        # B (Parallel Target)
        lib.step_env_apply_actions_batch(b_states, n_envs)
        
        _assert_env_equal(a_states, b_states, n_envs, n_agents)
        
        # next step
        _advance(a_states)
        _advance(b_states)

def test_level2_wall_grid_collision_matches_baseline():
    """Collision into walls"""
    n_envs, n_agents, steps, threads = 8, 16, 10, 8
    seed = 1618
    _set_omp_threads(threads)
    
    
    grid = _make_wall_grid(40, 40)
    a_states, a_keep, _, _ = _init_states(n_envs, n_agents, seed, grid_np=grid)
    b_states, b_keep, _, _ = _init_states(n_envs, n_agents, seed, grid_np=grid)
    
    # set Pacman and ghosts near the wall
    mid = 40 // 2
    for e in range(n_envs):
        a_states[e].pacman_x_in = mid - 1; a_states[e].pacman_y_in = 20
        b_states[e].pacman_x_in = mid - 1; b_states[e].pacman_y_in = 20
        
        a_keep[e]["ghosts_in"][0].x = mid - 1; a_keep[e]["ghosts_in"][0].y = 19
        b_keep[e]["ghosts_in"][0].x = mid - 1; b_keep[e]["ghosts_in"][0].y = 19

    g_act = np.zeros((steps, n_envs, n_agents), dtype=np.int32)
    g_act[:, :, 0] = 4 
    p_act = np.full((steps, n_envs), 4, dtype=np.int32) 

    for t in range(steps):
        _write_inputs(a_states, a_keep, g_act[t], p_act[t], pacman_speed=2)
        _write_inputs(b_states, b_keep, g_act[t], p_act[t], pacman_speed=2)
        
        lib.step_env_apply_actions_batch_sequential(a_states, n_envs)
        lib.step_env_apply_actions_batch(b_states, n_envs)
        
        _assert_env_equal(a_states, b_states, n_envs, n_agents)
        _advance(a_states)
        _advance(b_states)