#!/bin/bash

# ====================================================
# 64-Core Architecture Validation
# ====================================================

# 1. 啟用巢狀平行 (關鍵設定)
export OMP_MAX_ACTIVE_LEVELS=2
# spread: 第一層盡量散開到不同 CPU 插槽
# close:  第二層盡量緊貼在第一層的 Cache 附近
export OMP_PROC_BIND=spread,close
export OMP_PLACES=threads

# 共用參數
AGENTS_A=1024  # 場景 A 的鬼數量
AGENTS_B=4096  # 場景 B 的鬼數量 (加重負載)
STEPS=100

echo "=========================================================="
echo "🔥 64-Core Hybrid Parallelism Experiment"
echo "=========================================================="

# ====================================================
# 場景 A：高吞吐量 (High Throughput)
# 條件：環境數 (64) >= 核心數 (64)
# 預期：Level 2 全開最快
# ====================================================
ENVS_A=64

echo ""
echo "🧪 Scenario A: Massive Batch (N_ENVS=$ENVS_A)"
echo "   Goal: 證明當任務夠多時，Level 2 (Env Parallelism) 效率最高"
echo "----------------------------------------------------------"

# 策略 1: 純 Level 2 [64 Env x 1 Thread]
# 64 個環境同時跑，剛好填滿 64 核。沒有切換成本。
echo -n "1. Strategy [64, 1] (Pure Level 2): "
env OMP_NUM_THREADS=64,1 python3 level12_train.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"

# 策略 2: 混合模式 [16 Env x 4 Threads]
# 同時跑 16 個環境，每個環境用 4 核加速。總共 64 核。
echo -n "2. Strategy [16, 4] (Hybrid Mode ): "
env OMP_NUM_THREADS=16,4 python3 level12_train.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"

# 策略 3: 純 Level 1 [1 Env x 64 Threads]
# 64 個環境排隊，每次只跑 1 個，但用 64 核全力跑。
echo -n "3. Strategy [ 1,64] (Pure Level 1): "
env OMP_NUM_THREADS=1,64 python3 level12_train.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"


# ====================================================
# 場景 B：低延遲 / 記憶體受限 (Low Latency)
# 條件：環境數 (4) < 核心數 (64)
# 預期：純 Level 2 會慘敗，混合模式 (Hybrid) 會大勝
# ====================================================
ENVS_B=4

echo ""
echo "🧪 Scenario B: Small Batch / Latency Critical (N_ENVS=$ENVS_B)"
echo "   Goal: 證明當環境少於核心數時，需要 Level 1 補位來吃滿算力"
echo "----------------------------------------------------------"

# 策略 1: 純 Level 2 [64, 1] -> 災難！
# 你開了 64 條線，但只有 4 個環境。
# 結果：4 個核心在工作，60 個核心在睡覺 (CPU 使用率 6%)。
echo -n "1. Strategy [64, 1] (Pure Level 2): "
env OMP_NUM_THREADS=64,1 python3 level12_train.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

# 策略 2: 混合模式 [4, 16] -> 完美！
# 4 個環境同時跑。剩下的算力全部分配給內部加速。
# 4 * 16 = 64 核心全滿 (CPU 使用率 100%)。
echo -n "2. Strategy [ 4,16] (Hybrid Mode ): "
env OMP_NUM_THREADS=4,16 python3 level12_train.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

# 策略 3: 純 Level 1 [1, 64] -> 次佳
# 環境排隊跑。雖然沒浪費核心，但輸在序列化的等待時間。
echo -n "3. Strategy [ 1,64] (Pure Level 1): "
env OMP_NUM_THREADS=1,64 python3 level12_train.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

echo "=========================================================="