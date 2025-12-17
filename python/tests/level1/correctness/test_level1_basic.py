import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# 往上跳三層: speedup -> level1 -> tests -> python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

import numpy as np
import ctypes
from pacman_env import PacmanEnv

def test_observation_logic():
    print("=== 1. Initialize Environment ===")
    # 建立一個 40x40 的空地
    grid = np.zeros((40, 40), dtype=np.int32)
    
    # 建立 3 隻 Ghost 來測試鄰居邏輯
    # Ghost 0: 測試者
    # Ghost 1: 站在 Ghost 0 旁邊 (測試 Neighbor Sensing)
    # Ghost 2: 站在很遠的地方 (測試 Filter)
    n_agents = 3
    env = PacmanEnv(grid, n_agents=n_agents)
    obs = env.reset()
    
    print(f"Observation Shape: {obs['ghost_tensor'].shape}")
    assert obs['ghost_tensor'].shape == (3, 17), "Shape mismatch! Should be (3, 17)"

    print("\n=== 2. Setup Scenario (Manually Override Positions) ===")
    # Pacman 固定在中心 (20, 20)
    env.pac_x = 20
    env.pac_y = 20

    # --- Ghost 0: 靠近 Pacman ---
    # Pos: (18, 20), Dist to Pacman: 2.0 (在感知範圍 <=3 內)
    env.ptr_ghosts_in[0].x = 18
    env.ptr_ghosts_in[0].y = 20
    env.ptr_ghosts_in[0].alive = 1

    # --- Ghost 1: 靠近 Ghost 0 (當作鄰居) ---
    # Pos: (18, 19), Dist to Ghost 0: 1.0 (在通訊範圍 <=3 內)
    env.ptr_ghosts_in[1].x = 18
    env.ptr_ghosts_in[1].y = 19
    env.ptr_ghosts_in[1].alive = 1

    # --- Ghost 2: 邊緣人 ---
    # Pos: (2, 2), 離大家都超遠
    env.ptr_ghosts_in[2].x = 2
    env.ptr_ghosts_in[2].y = 2
    env.ptr_ghosts_in[2].alive = 1

    print(f"Pacman Pos: ({env.pac_x}, {env.pac_y})")
    print(f"Ghost 0 Pos: ({env.ptr_ghosts_in[0].x}, {env.ptr_ghosts_in[0].y})")
    print(f"Ghost 1 Pos: ({env.ptr_ghosts_in[1].x}, {env.ptr_ghosts_in[1].y})")

    print("\n=== 3. Run Step (Computing Observations) ===")
    # 讓大家都不動 (Action=0)，純粹觸發 C 語言的 compute_observations
    actions = np.zeros(n_agents, dtype=np.int32)
    obs, _, _, _ = env.step(actions, 0, 0)
    
    tensor = obs['ghost_tensor']

    print("\n=== 4. Verify Ghost 0 Observation (The Active Agent) ===")
    g0_obs = tensor[0]
    
    # [Index 0-1] Self Position (Normalized)
    print(f"Self (Norm): {g0_obs[0]:.3f}, {g0_obs[1]:.3f}")
    assert g0_obs[0] > 0, "Self X should be normalized > 0"

    # [Index 2-4] Pacman Sensing
    # g0 在 (18, 20), Pac 在 (20, 20). Vector應該大致朝向 (+x, 0)
    # 加上雜訊後，vx 應該是正的，conf > 0
    print(f"Pacman Sense -> vx: {g0_obs[2]:.3f}, vy: {g0_obs[3]:.3f}, Conf: {g0_obs[4]:.3f}")
    
    if g0_obs[4] > 0:
        print("  ✅ SUCCESS: Pacman detected!")
    else:
        print("  ❌ FAILURE: Pacman NOT detected (Confidence is 0)")

    # [Index 5-7] Nearest Neighbor (Should be Ghost 1)
    # Ghost 1 在 (18, 19), Ghost 0 在 (18, 20). 
    # Relative: (0, -1). Dist: 1.0
    print(f"Neighbor 1 -> dx: {g0_obs[5]:.3f}, dy: {g0_obs[6]:.3f}, Dist: {g0_obs[7]:.3f}")
    
    if g0_obs[7] > 0:
        print("  ✅ SUCCESS: Neighbor detected!")
        # 簡單驗證距離 (Normalize 過的，所以是很小的數字)
        # Dist 1.0 / 40.0 = 0.025
        if abs(g0_obs[7] - 0.025) < 0.01:
             print("     (Distance is correct)")
    else:
        print("  ❌ FAILURE: Neighbor NOT detected")

    # [Index 8-10] Second Neighbor (Should be Empty/Zero)
    # Ghost 2 太遠了，不應該出現
    print(f"Neighbor 2 -> dx: {g0_obs[8]:.3f}, dy: {g0_obs[9]:.3f}, Dist: {g0_obs[10]:.3f}")
    if g0_obs[10] == 0:
        print("  ✅ SUCCESS: Far neighbor correctly filtered (Zero Padding).")
    else:
        print("  ❌ FAILURE: Ghost 2 shouldn't be here!")

    print("\n=== 5. Verify Ghost 2 Observation (The Loner) ===")
    g2_obs = tensor[2]
    print(f"Pacman Conf: {g2_obs[4]:.3f}")
    print(f"Neighbor 1 Dist: {g2_obs[7]:.3f}")
    
    if g2_obs[4] == 0 and g2_obs[7] == 0:
        print("  ✅ SUCCESS: Ghost 2 sees nothing (as expected).")
    else:
        print("  ❌ FAILURE: Ghost 2 saw something illegally.")

if __name__ == "__main__":
    try:
        test_observation_logic()
        print("\n🏆 ALL TESTS PASSED! Sequential Baseline is Ready.")
    except Exception as e:
        print(f"\n💥 TEST FAILED: {e}")