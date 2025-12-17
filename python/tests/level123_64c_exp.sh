#!/bin/bash
# csrc/run_mpi_levels.sh

# 確保 Python script 路徑正確 (請依實際情況調整)
PY_SCRIPT="../main_mpi.py"

echo "=========================================================="
echo "🛡️  HPC Multi-Level Parallelism Benchmark (Total Cores: 64)"
echo "=========================================================="

# ----------------------------------------------------------------
# 實驗 1: 純 Level 3 (Pure MPI)
# ----------------------------------------------------------------
# 配置: 64 個 MPI Processes，每個 Process 1 核心 (單執行緒)
# 意義: 模擬傳統的分散式訓練，沒有用 OpenMP 加速
# ----------------------------------------------------------------
echo -e "\n[Experiment 1] Pure Level 3 (MPI Only)"
echo "Config: 64 Ranks, 1 Thread/Rank, 1 Env/Rank (Total 64 Envs)"
export OMP_NUM_THREADS=1
# 每個 rank 跑 1 個環境，總共 64 個環境
mpirun -np 64 --bind-to core python3 $PY_SCRIPT --n_envs_per_rank 1 --n_agents 4096 --steps 100

# ----------------------------------------------------------------
# 實驗 2: Level 3 + Level 2 (MPI + Env Parallelism)
# ----------------------------------------------------------------
# 配置: 4 個 MPI Processes，每個 Process 16 核心
# 每個 Process 跑 16 個環境 (Level 2 負責這 16 個環境的平行)
# 意義: 這是高吞吐量訓練的最佳解
# ----------------------------------------------------------------
echo -e "\n[Experiment 2] Level 3 + Level 2 (MPI + Env Parallelism)"
echo "Config: 4 Ranks, 16 Threads/Rank, 16 Envs/Rank (Total 64 Envs)"
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread,close
# 每個 Rank 負責 16 個環境，Level 2 會把這 16 個平行化
mpirun -np 4 --bind-to socket python3 $PY_SCRIPT --n_envs_per_rank 16 --n_agents 4096 --steps 100

# ----------------------------------------------------------------
# 實驗 3: Level 3 + Level 1 (MPI + Agent Parallelism)
# ----------------------------------------------------------------
# 配置: 4 個 MPI Processes，每個 Process 16 核心
# 但每個 Process 只跑「1 個環境」!
# 意義: 這是「低延遲/即時推論」的最佳解。Level 2 沒事做，Level 1 必須跳出來用 16 核加速那 1 個環境。
# ----------------------------------------------------------------
echo -e "\n[Experiment 3] Level 3 + Level 1 (MPI + Agent Parallelism)"
echo "Config: 4 Ranks, 16 Threads/Rank, 1 Env/Rank (Total 4 Envs)"
export OMP_NUM_THREADS=16
# 關鍵：每個 Rank 只有 1 個環境，強迫 OpenMP 去切分 Agents (Level 1)
mpirun -np 4 --bind-to socket python3 $PY_SCRIPT --n_envs_per_rank 1 --n_agents 4096 --steps 100

# ----------------------------------------------------------------
# 實驗 4: Full Hybrid (Level 3 + 2 + 1)
# ----------------------------------------------------------------
# 配置: 4 個 MPI Processes，每個 Process 16 核心
# OpenMP 開啟巢狀平行 (4 Envs x 4 Agents)
# 意義: 當環境數適中 (例如每個 Rank 4 個)，需要同時利用兩層來吃滿 16 核
# ----------------------------------------------------------------
echo -e "\n[Experiment 4] Full Hybrid (L3 + L2 + L1)"
echo "Config: 4 Ranks, Nested OMP (4x4), 4 Envs/Rank (Total 16 Envs)"
export OMP_NUM_THREADS=4,4
export OMP_MAX_ACTIVE_LEVELS=2
# 每個 Rank 4 個環境，外層 4 threads 負責 envs，內層 4 threads 負責 agents
mpirun -np 4 --bind-to socket python3 $PY_SCRIPT --n_envs_per_rank 4 --n_agents 4096 --steps 100