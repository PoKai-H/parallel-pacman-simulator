import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
# 往上跳三層: speedup -> level1 -> tests -> python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pacman_env import PacmanEnv

def test_game_dynamics():
    print("=== Testing Game Dynamics (Physics & Rewards) ===")
    
    # 1. 初始化
    grid = np.zeros((40, 40), dtype=np.int32)
    # 畫一道牆在 (10, 10)
    grid[10, 10] = 1
    
    env = PacmanEnv(grid, n_agents=1)
    env.reset()
    
    # 2. 測試撞牆 (Collision)
    print("\n[Test 1] Wall Collision")
    # 把鬼放在牆壁旁邊 (10, 9)，讓它向下走 (Action=2) 撞牆
    env.ptr_ghosts_in[0].x = 10
    env.ptr_ghosts_in[0].y = 9
    env.ptr_ghosts_in[0].alive = 1
    
    ghost_actions = np.array([2], dtype=np.int32) # Down
    env.step(ghost_actions, pacman_action=0)
    
    print(f"Ghost Pos after hitting wall: {env.ptr_ghosts_in[0].x}, {env.ptr_ghosts_in[0].y}")
    
    if env.ptr_ghosts_in[0].y == 9:
        print("  ✅ SUCCESS: Ghost correctly stopped by wall.")
    else:
        print(f"  ❌ FAILURE: Ghost went through wall! (y={env.ptr_ghosts_in[0].y})")

    # 3. 測試捕捉與獎勵 (Capture & Reward)
    print("\n[Test 2] Capture Logic")
    # Pacman 在 (20, 20)
    env.pac_x = 20
    env.pac_y = 20
    
    # 把鬼直接放在 Pacman 上面 (模擬這一回合抓到)
    env.ptr_ghosts_in[0].x = 20
    env.ptr_ghosts_in[0].y = 20
    
    # 隨便動一下
    obs, reward, done, _ = env.step(np.array([0], dtype=np.int32), 0)
    
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    
    if done and reward['pacman'] < 0:
        print("  ✅ SUCCESS: Game Over trigger works! Negative reward received.")
    else:
        print("  ❌ FAILURE: Game did not end or reward is wrong.")

if __name__ == "__main__":
    test_game_dynamics()