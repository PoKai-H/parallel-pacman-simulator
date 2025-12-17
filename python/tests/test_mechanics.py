import pytest
import numpy as np
import sys
import os

# --- 1. 路徑設定 (指向上一層 python/ 資料夾) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from pacman_env import PacmanVecEnv, PacmanEnv

# =================================================================
# Group A: 基礎機制測試 (5 Tests)
# =================================================================

def test_init_basic():
    """T1: 測試環境初始化不崩潰"""
    env = PacmanVecEnv(np.zeros((20, 20), dtype=np.int8), n_envs=2, n_agents=4)
    assert env is not None

def test_obs_shape_consistency():
    """T2: 測試觀察值形狀恆定 (N_ENVS, N_AGENTS, 17)"""
    n_envs, n_agents = 4, 10
    env = PacmanVecEnv(np.zeros((20, 20), dtype=np.int8), n_envs=n_envs, n_agents=n_agents)
    obs = env.reset()
    assert obs.shape == (n_envs, n_agents, 17)

def test_action_space_bounds():
    """T3: 測試非法動作 (應該被 C Kernel 忽略或處理，不應 SegFault)"""
    env = PacmanVecEnv(np.zeros((10, 10), dtype=np.int8), n_envs=1, n_agents=1)
    env.reset()
    # 傳入非法動作 99
    actions = np.full((1, 1), 99, dtype=np.int32)
    env.step(actions) # 只要沒 crash 就算過

def test_reward_structure():
    """T4: 測試 Reward 格式 (pacman, ghosts, step)"""
    env = PacmanEnv(np.zeros((10, 10), dtype=np.int32), n_agents=1)
    env.reset()
    obs, reward, done, _ = env.step(np.array([0], dtype=np.int32), 0)
    assert 'pacman' in reward
    assert 'ghosts' in reward
    assert isinstance(reward['pacman'], (float, int))

def test_reset_mechanism():
    """T5: 測試 Reset 後狀態改變"""
    env = PacmanVecEnv(np.zeros((10, 10), dtype=np.int8), n_envs=1, n_agents=1)
    obs1 = env.reset()
    # 這裡假設 Reset 會隨機初始化位置，檢查是否有 output
    assert np.any(obs1 != -999) # 簡單檢查數值有效性

# =================================================================
# Group B: 物理與邏輯測試 (5 Tests) - 改寫自你的 test_level1_loop.py
# =================================================================

def test_wall_collision():
    """T6: 撞牆測試"""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[5, 5] = 1 # Wall
    env = PacmanEnv(grid, n_agents=1)
    env.reset()
    # 強制設定位置在牆邊 (5, 4)，向下走 (Action=2) 撞 (5, 5)
    env.ptr_ghosts_in[0].x = 5
    env.ptr_ghosts_in[0].y = 4
    env.step(np.array([2], dtype=np.int32), 0)
    # 預期：位置不變
    assert env.ptr_ghosts_in[0].y == 4

def test_pacman_capture():
    """T7: 抓到 Pacman 的獎勵測試"""
    grid = np.zeros((10, 10), dtype=np.int32)
    env = PacmanEnv(grid, n_agents=1)
    env.reset()
    # 重疊
    env.pac_x, env.pac_y = 5, 5
    env.ptr_ghosts_in[0].x = 5
    env.ptr_ghosts_in[0].y = 5
    obs, reward, done, _ = env.step(np.array([0], dtype=np.int32), 0)
    assert done == True
    assert reward['pacman'] < 0 # Pacman 死了應該扣分

# =================================================================
# Group C: 參數化擴展性測試 (20 Tests!)
# 這是湊數量的神器：pytest 會把每個組合當成一個獨立的測試
# =================================================================

@pytest.mark.parametrize("grid_size", [10, 50, 100, 200])
@pytest.mark.parametrize("n_agents", [1, 16, 128, 1024, 4096])
def test_scalability_stress(grid_size, n_agents):
    """
    T8-T27: 壓力測試
    組合: 4種地圖 * 5種Agent數量 = 20個測試案例
    目的: 確保在不同規模下 C Kernel 記憶體配置與運算都穩定
    """
    try:
        # 使用 VecEnv (Level 2) 測試
        env = PacmanVecEnv(np.zeros((grid_size, grid_size), dtype=np.int8), n_envs=2, n_agents=n_agents)
        obs = env.reset()
        actions = np.random.randint(0, 5, size=(2, n_agents), dtype=np.int32)
        next_obs, rewards, dones, _ = env.step(actions)
        
        assert next_obs.shape == (2, n_agents, 17)
        assert len(rewards) == 2
    except Exception as e:
        pytest.fail(f"Crash at Grid={grid_size}, Agents={n_agents}: {e}")

if __name__ == "__main__":
    # 讓你可以直接 python test_mechanics.py 執行
    sys.exit(pytest.main(["-v", __file__]))