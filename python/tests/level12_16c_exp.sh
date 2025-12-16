#!/bin/bash

# 確保這兩個參數根據你的機器核心數調整
TOTAL_CORES=16  # 假設我們要用 16 核心來做實驗

# 啟用巢狀平行 (這是關鍵！)
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close
export OMP_PLACES=threads


# ====================================================
# 場景 A：高 Batch Size (適合 Level 2)
# 16 個環境，每個環境 1024 隻鬼
# ====================================================
ENVS_A=16
AGENTS_A=1024
STEPS=100

echo "=========================================================="
echo "🧪 Scenario A: High Batch Size (N_ENVS=$ENVS_A, AGENTS=$AGENTS_A)"
echo "   Goal: 證明當環境夠多時，Level 2 平行化最好"
echo "=========================================================="

# 策略 1: 純 Level 2 (16 x 1) -> 預期最快
echo "Running Strategy [16, 1] (Pure Level 2)..."
env OMP_NUM_THREADS=16,1 python3 level12_train.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"

# 策略 2: 混合 (4 x 4) -> 預期中等 (因為同時只能跑 4 個環境，另外 12 個在排隊)
echo "Running Strategy [ 4, 4] (Hybrid)..."
env OMP_NUM_THREADS=4,4 python3 level12_train.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"

# 策略 3: 純 Level 1 (1 x 16) -> 預期最慢 (完全序列化，16 個環境排隊跑)
echo "Running Strategy [ 1,16] (Pure Level 1)..."
env OMP_NUM_THREADS=1,16 python3 level12_train.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"


# ====================================================
# 場景 B：低 Batch Size (適合 Level 1)
# 4 個環境，每個環境 4096 隻鬼 (運算量很大，但環境數很少)
# ====================================================
ENVS_B=4
AGENTS_B=4096

echo ""
echo "=========================================================="
echo "🧪 Scenario B: Low Batch Size (N_ENVS=$ENVS_B, AGENTS=$AGENTS_B)"
echo "   Goal: 證明當環境數少於核心數時，需要 Level 1 補位"
echo "=========================================================="

# 策略 1: 純 Level 2 (16 x 1) -> 這裡會浪費核心！
# 因為只有 4 個環境，所以只有 4 個核心在工作，另外 12 個在納涼。
echo "Running Strategy [16, 1] (Pure Level 2)..."
env OMP_NUM_THREADS=16,1 python3 level12_train.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

# 策略 2: 混合 (4 x 4) -> 預期最快！
# 4 個環境同時跑 (L2=4)，每個環境再用 4 核心加速 (L1=4)。
# 剛好 4x4=16 核心全滿。
echo "Running Strategy [ 4, 4] (Hybrid)..."
env OMP_NUM_THREADS=4,4 python3 level12_train.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

# 策略 3: 純 Level 1 (1 x 16) -> 預期中等
# 環境排隊跑，但每個環境跑得很快。
echo "Running Strategy [ 1,16] (Pure Level 1)..."
env OMP_NUM_THREADS=1,16 python3 level12_train.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"