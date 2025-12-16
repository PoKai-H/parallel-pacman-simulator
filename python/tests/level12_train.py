import argparse
import numpy as np
import time
import sys
import os

# 路徑修正
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)

from pacman_env import PacmanVecEnv

def softmax(x):
    """用 Numpy 實作 Softmax"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--n_agents", type=int, default=1024) 
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    N_ENVS = args.n_envs
    N_AGENTS = args.n_agents
    STEPS = args.steps
    OBS_DIM = 17 # 你的特徵數
    ACTION_DIM = 5 # 上下左右停
    
    # --- 1. 初始化環境 ---
    print(f"Init: {N_ENVS} Envs x {N_AGENTS} Agents. (Backend: Pure NumPy)")
    grid = np.zeros((40, 40), dtype=np.int8)
    env = PacmanVecEnv(grid, n_envs=N_ENVS, n_agents=N_AGENTS)
    
    # --- 2. 初始化「假」神經網路權重 ---
    # 這模擬了一個簡單的 Linear Policy: Action = argmax(Obs @ W + b)
    # 形狀: (17, 5)
    weights = np.random.randn(OBS_DIM, ACTION_DIM).astype(np.float32)
    bias = np.random.randn(ACTION_DIM).astype(np.float32)

    obs = env.reset()
    
    print(">>> Start Training Loop (Simulated)...")
    start_time = time.time()
    
    for step in range(STEPS):
        # --- 3. 模擬 Model Inference (Forward Pass) ---
        # 這裡的數學運算量跟 PyTorch 的 Linear 層完全一樣
        # Obs: (N, 16, 17) -> Flatten -> (N*16, 17)
        
        # A. Reshape (把 batch 和 agents 攤平以進行矩陣運算)
        flat_obs = obs.reshape(-1, OBS_DIM) 
        
        # B. 矩陣乘法 (Matrix Multiplication) - 這是最吃 CPU 的部分
        logits = flat_obs @ weights + bias 
        
        # C. 選擇動作 (Argmax)
        flat_actions = np.argmax(logits, axis=1).astype(np.int32)
        
        # D. Reshape 回去給環境用
        actions = flat_actions.reshape(N_ENVS, N_AGENTS)
        
        # --- 4. 環境互動 ---
        next_obs, rewards, dones, _ = env.step(actions)
        
        # --- 5. 模擬 Backpropagation (Backward Pass) ---
        # 真正的訓練會在這裡更新 weights
        # 我們做一個簡單的加法來模擬 CPU 寫入記憶體的開銷
        weights += 0.0001 * np.random.randn(OBS_DIM, ACTION_DIM).astype(np.float32)
        
        obs = next_obs

    end_time = time.time()
    total_time = end_time - start_time
    sps = (N_ENVS * STEPS) / total_time
    
    print(f"Final Result - Throughput: {sps:.2f} env_steps/s")

if __name__ == "__main__":
    main()