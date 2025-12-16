import sys
import os
import time
import numpy as np

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pacman_env import PacmanEnv

def run_benchmark(steps=10000):
    print(f"=== Running Benchmark ({steps} steps) ===")
    N_AGENTS = 512
    # 初始化
    grid = np.zeros((80, 80), dtype=np.int32)
    env = PacmanEnv(grid, n_agents=N_AGENTS) # 16 隻鬼
    env.reset()
    
    # 預先生成 Actions (避免 Python 迴圈內生成拖慢速度)
    actions = np.random.randint(0, 5, size=(steps, N_AGENTS), dtype=np.int32)
    
    start_time = time.time()
    
    # 暴力迴圈
    for i in range(steps):
        env.step(actions[i], pacman_action=0)
        
    end_time = time.time()
    duration = end_time - start_time
    sps = steps / duration
    
    print(f"Total Time: {duration:.4f} sec")
    print(f"Throughput: {sps:.2f} Steps/Sec")

if __name__ == "__main__":
    run_benchmark()