# python/main_mpi.py
from mpi4py import MPI
import numpy as np
import time
from pacman_env import PacmanEnv

def run_simulation_batch(n_episodes, rank):
    """
    Worker Function: 跑 n 個 episodes，回傳統計數據
    """
    # 1. 初始化環境 (每個 Rank 都有獨立的 C Kernel 實例)
    #    注意：C 內部的 OpenMP 會在這裡發揮 Level 1/2 的加速作用
    grid = np.zeros((40, 40), dtype=np.int32)
    # (這邊可以載入真實地圖)
    
    env = PacmanEnv(grid, n_agents=16)
    
    local_results = []
    
    start_time = time.time()
    
    for i in range(n_episodes):
        # Reset
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # 這裡可以用簡單的 Policy，或者隨機與追蹤混合
            # 為了測試 HPC 效能，用隨機 Action 即可，重點是操爆 C Kernel
            ghost_actions = np.random.randint(0, 5, size=16, dtype=np.int32)
            pacman_action = np.random.randint(0, 5)
            
            obs, reward, done, _ = env.step(ghost_actions, pacman_action)
            total_reward += reward['pacman']
            steps += 1
            
        local_results.append({
            "episode_id": f"Rank{rank}_Ep{i}",
            "steps": steps,
            "pacman_reward": total_reward
        })
        
        if (i+1) % 10 == 0:
            print(f"[Rank {rank}] Finished {i+1}/{n_episodes} episodes")

    end_time = time.time()
    print(f"[Rank {rank}] Done. Throughput: {steps / (end_time - start_time):.2f} steps/sec")
    
    return local_results

def main():
    # 1. MPI 初始化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 2. 定義總工作量 (例如總共要跑 1000 個 episodes)
    TOTAL_EPISODES = 1024
    
    # 3. 分配工作 (簡單的整除分配)
    # Rank 0 也可以跑，或者 Rank 0 只當 Manager (看你們策略)
    # 這裡假設大家一起跑
    my_episodes = TOTAL_EPISODES // size
    
    # 餘數分配給最後一個 Rank
    if rank == size - 1:
        my_episodes += TOTAL_EPISODES % size
        
    print(f"Rank {rank}/{size} starting... assigned {my_episodes} episodes.")
    
    # 4. 開始執行 (Call C Kernel via Python Wrapper)
    my_data = run_simulation_batch(my_episodes, rank)
    
    # 5. 收集結果 (Gather)
    # all_data 會是一個 list，包含每個 Rank 回傳的 list
    all_data = comm.gather(my_data, root=0)
    
    # 6. Rank 0 統整並輸出
    if rank == 0:
        print("\n=== MPI Simulation Complete ===")
        # Flatten list of lists
        flat_results = [item for sublist in all_data for item in sublist]
        
        total_steps = sum(r['steps'] for r in flat_results)
        avg_steps = total_steps / len(flat_results)
        
        print(f"Total Episodes: {len(flat_results)}")
        print(f"Average Steps: {avg_steps:.2f}")
        # 這裡 Member D 可以接手畫圖
        
if __name__ == "__main__":
    main()