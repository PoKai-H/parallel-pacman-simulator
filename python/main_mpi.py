# python/main_mpi.py (Debug Version)
import sys
import os
import time
import argparse
import numpy as np
import traceback # 用來印出詳細錯誤
from mpi4py import MPI

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 嘗試 Import，失敗就報錯
try:
    from pacman_env import PacmanVecEnv
except ImportError as e:
    print(f"❌ Import Error: {e}", flush=True)
    sys.exit(1)

def run_worker(rank, n_envs, n_agents, steps):
    grid = np.zeros((40, 40), dtype=np.int8)
    
    # [MOD] 讓每個 Rank 都回報狀態，不要只讓 Rank 0 講話
    print(f"[Rank {rank}] 1. Initializing VecEnv ({n_envs} envs)...", flush=True)

    try:
        # 這裡是最容易當掉的地方 (C++ 初始化)
        env = PacmanVecEnv(grid, n_envs=n_envs, n_agents=n_agents)
        obs = env.reset()
        
        # Pre-generate actions
        actions = np.random.randint(0, 5, size=(n_envs, n_agents), dtype=np.int32)
        
        print(f"[Rank {rank}] 2. Init Done. Waiting at Barrier...", flush=True)
        
        # [Checkpoint] 
        # 如果 Rank 1 沒印出這行，代表它死在上面那幾行
        MPI.COMM_WORLD.Barrier()
        
        if rank == 0:
            print(f"[Rank {rank}] 3. Everyone Ready! Starting Loop...", flush=True)
        
        start_time = time.time()
        for _ in range(steps):
            env.step(actions)
        end_time = time.time()
        
        print(f"[Rank {rank}] 4. Finished!", flush=True)
        
        local_steps = n_envs * steps
        return local_steps / (end_time - start_time)

    except Exception as e:
        print(f"❌ [Rank {rank}] CRASHED: {e}", flush=True)
        traceback.print_exc() # 印出詳細錯誤
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_envs_per_rank", type=int, default=16)
    parser.add_argument("--n_agents", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"=== MPI Experiment (Ranks: {size}) ===", flush=True)
        # 印出關鍵環境變數，確認你有沒有設對
        print(f"Debug: OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'NOT SET (DANGER!)')}", flush=True)

    # 執行 Worker
    local_throughput = run_worker(rank, args.n_envs_per_rank, args.n_agents, args.steps)
    
    # 收集並加總
    try:
        all_throughputs = comm.gather(local_throughput, root=0)
    except Exception as e:
        print(f"❌ [Rank {rank}] Gather Failed: {e}", flush=True)
        sys.exit(1)

    if rank == 0:
        total_throughput = sum(all_throughputs)
        print(f"🚀 Final Total Throughput: {total_throughput:.2f} env_steps/s", flush=True)

if __name__ == "__main__":
    main()