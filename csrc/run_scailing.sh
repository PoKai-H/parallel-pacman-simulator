#!/bin/bash

# ==========================================
# 64-Core Thread Allocation Experiment
# ==========================================

# 編譯確保最新
make bench > /dev/null

echo "================================================================================="
echo "| Strategy | L2 (Envs) | L1 (Agents)| Total Threads | Throughput (steps/s) |"
echo "================================================================================="

# 定義我們要測試的組合 (L2, L1)
# 假設機器有 64 核心
# 格式: "L2執行緒數,L1執行緒數"
CONFIGS=(
    "64,1"   # 純 L2: 同時跑 64 個環境，每個環境單核 (最常見做法)
    "32,2"   # 混合: 32 環境，每個用 2 核
    "16,4"   # 混合: 16 環境，每個用 4 核 (你目前的設定)
    "8,8"    # 混合: 深度平衡
    "4,16"   # 混合: 少量環境，重度加速
    "2,32"   # 幾乎純 L1
    "1,64"   # 純 L1: 一次跑 1 個環境，但用 64 核全力算 (序列化)
)

# 固定環境變數
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close  # 關鍵：讓執行緒正確散開
export OMP_PLACES=threads

for CONF in "${CONFIGS[@]}"; do
    # 解析 L2 和 L1 的數值
    L2=$(echo $CONF | cut -d',' -f1)
    L1=$(echo $CONF | cut -d',' -f2)
    
    # 執行 Benchmark
    # 我們只抓取包含 "Throughput" 的那一行，並用 awk 提取數字
    RESULT=$(env OMP_NUM_THREADS=$CONF ./benchmark | grep "Throughput" | awk '{print $2}')
    
    # 格式化輸出
    printf "| %-8s | %-9s | %-10s | %-13s | \033[1;32m%-20s\033[0m |\n" \
           "$CONF" "$L2" "$L1" "$((L2 * L1))" "$RESULT"
done

echo "================================================================================="