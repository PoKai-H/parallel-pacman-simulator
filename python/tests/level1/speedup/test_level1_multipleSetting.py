import sys
import os
import time
import numpy as np
import argparse  # 新增這個庫

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pacman_env import PacmanEnv

def run_benchmark(grid_size, n_agents, steps):
    print(f"=== Config: Grid={grid_size}x{grid_size}, Agents={n_agents}, Steps={steps} ===")
    
    # 初始化 (使用傳入的參數)
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    env = PacmanEnv(grid, n_agents=n_agents) 
    env.reset()
    
    # 預先生成 Actions
    actions = np.random.randint(0, 5, size=(steps, n_agents), dtype=np.int32)
    
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
    # 設定參數解析器
    parser = argparse.ArgumentParser(description='Pacman Performance Test')
    parser.add_argument('--grid_size', type=int, default=80, help='Size of the grid (NxN)')
    parser.add_argument('--n_agents', type=int, default=256, help='Number of agents')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps')
    
    args = parser.parse_args()
    
    # 呼叫主函式並傳入參數
    run_benchmark(args.grid_size, args.n_agents, args.steps)