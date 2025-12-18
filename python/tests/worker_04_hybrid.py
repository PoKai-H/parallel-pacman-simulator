import argparse
import numpy as np
import time
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)

from pacman_env import PacmanVecEnv

def softmax(x):
    """Uses Numpy to build Softmax"""
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
    OBS_DIM = 17 # features
    ACTION_DIM = 5 
    
    # --- 1. init ---
    print(f"Init: {N_ENVS} Envs x {N_AGENTS} Agents. (Backend: Pure NumPy)")
    grid = np.zeros((40, 40), dtype=np.int8)
    env = PacmanVecEnv(grid, n_envs=N_ENVS, n_agents=N_AGENTS)
    
    # --- 2. fake weights ---
    # Linear Policy: Action = argmax(Obs @ W + b)
    # shape: (17, 5)
    weights = np.random.randn(OBS_DIM, ACTION_DIM).astype(np.float32)
    bias = np.random.randn(ACTION_DIM).astype(np.float32)

    obs = env.reset()
    
    print(">>> Start Training Loop (Simulated)...")
    start_time = time.time()
    
    for step in range(STEPS):
        # --- 3.  Model Inference (Forward Pass) ---
        # Obs: (N, 16, 17) -> Flatten -> (N*16, 17)
        
        # A. Reshape 
        flat_obs = obs.reshape(-1, OBS_DIM) 
        
        # B. Matrix Multiplication
        logits = flat_obs @ weights + bias 
        
        # C. Argmax
        flat_actions = np.argmax(logits, axis=1).astype(np.int32)
        
        # D. Reshape for env
        actions = flat_actions.reshape(N_ENVS, N_AGENTS)
        
        # --- 4. environment interaction ---
        next_obs, rewards, dones, _ = env.step(actions)
        
        # --- 5. Backpropagation (Backward Pass) ---
        weights += 0.0001 * np.random.randn(OBS_DIM, ACTION_DIM).astype(np.float32)
        
        obs = next_obs

    end_time = time.time()
    total_time = end_time - start_time
    sps = (N_ENVS * STEPS) / total_time
    
    print(f"Final Result - Throughput: {sps:.2f} env_steps/s")

if __name__ == "__main__":
    main()