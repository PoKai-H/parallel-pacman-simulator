import sys
import os
import time
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pacman_env import PacmanVecEnv

def test_vec_env():
    N_ENVS = 64
    N_AGENTS = 16
    grid = np.zeros((40, 40), dtype=np.int8)

    print("Initializing PacmanVecEnv...")
    env = PacmanVecEnv(grid, n_envs=N_ENVS, n_agents=N_AGENTS)

    obs = env.reset()
    print(f"Obs Shape: {obs.shape}") # Should be (64, 16, 17)

    actions = np.random.randint(0, 5, size=(N_ENVS, N_AGENTS), dtype=np.int32)

    print("Stepping...")
    start = time.time()

    steps=100
    for _ in range(steps):
        next_obs, rewards, dones, _ = env.step(actions)

    end = time.time()
    print(f"Done. Throughput: {(N_ENVS * steps) / (end-start):.2f} steps/s")

if __name__ == "__main__":
    test_vec_env()