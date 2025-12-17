import sys
import os
import numpy as np
import argparse
from mpi4py import MPI

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
# 往上跳三層: speedup -> level1 -> tests -> python
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from pacman_env import PacmanVecEnv

def calculate_checksum(obs):
    """計算 Observation 的總和作為 Checksum，簡單暴力"""
    return np.sum(obs)

def run_simulation(total_envs, rank, size, steps=50):
    # 1. 任務分配 (Workload Distribution)
    # 確保每個 Rank 知道自己負責全域中的哪些環境 ID
    envs_per_rank = total_envs // size
    start_env_id = rank * envs_per_rank
    
    # 2. 關鍵：設定 Seed
    # 我們希望第 N 號環境，不管是在 Rank 0 還是在 Rank 3 跑，
    # 它的行為都由 'seed + N' 決定，這樣才能確保結果一致。
    # 注意：這裡假設 PacmanVecEnv 支援 seed 參數，或者我們手動 seed numpy
    
    # 如果你的 VecEnv 內部是用 numpy.random，我們需要在這裡設定全域 seed
    # 但因為是 VecEnv，我們通常無法個別設定每個內部 env 的 seed
    # 所以我們依賴「初始化時傳入的 Grid/Agents」或「全域 Seed + Rank Offset」
    
    # 簡單解法：設定全域 Seed 為 Rank 相關，但這只保證 "同 Rank 重跑" 一致
    # 要保證 "單核 vs 多核" 一致，我們需要在 C 層級或 Python 層級
    # 確保 Env[i] 的亂數序列只跟 i 有關。
    
    # 這裡我們先做一個 "Weak Correctness" 測試：
    # 驗證 MPI 在多核運作下，資料收集 (Gather) 是否正確，沒有掉包。
    
    env = PacmanVecEnv(np.zeros((40,40), dtype=np.int8), n_envs=envs_per_rank, n_agents=100)
    
    # 讓每個環境的初始狀態不同，方便追蹤
    obs = env.reset()
    
    # 3. 執行模擬
    local_checksum = 0.0
    for _ in range(steps):
        # 使用固定的 Action，排除 Policy 的隨機性
        actions = np.ones((envs_per_rank, 100), dtype=np.int32) 
        obs, _, _, _ = env.step(actions)
        local_checksum += calculate_checksum(obs)
        
    return local_checksum, start_env_id

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_envs", type=int, default=16)
    args = parser.parse_args()

    # 執行任務
    my_checksum, my_start_id = run_simulation(args.total_envs, rank, size)
    
    # 收集結果
    all_data = comm.gather((my_start_id, my_checksum), root=0)
    
    if rank == 0:
        print(f"=== MPI Correctness Verification (Ranks: {size}) ===")
        total_checksum = 0.0
        sorted_data = sorted(all_data, key=lambda x: x[0])
        
        for start_id, chk in sorted_data:
            print(f"Rank covering Env {start_id:02d}+ : Checksum = {chk:.4f}")
            total_checksum += chk
            
        print(f"TOTAL GLOBAL CHECKSUM: {total_checksum:.4f}")
        
        # 將結果寫入檔案供比對
        mode = "serial" if size == 1 else "parallel"
        with open(f"checksum_{mode}.txt", "w") as f:
            f.write(f"{total_checksum:.6f}")

if __name__ == "__main__":
    main()