#!/bin/bash

# 確保腳本在錯誤時停止
set -e

# 定義測試腳本的路徑 (假設你在 python/ 目錄下執行此 sh)
TEST_SCRIPT="./level/speedup/test_level1_largeScaleLargeStep.py"

echo "========================================"
echo "Starting Performance Benchmark"
echo "Target Script: $TEST_SCRIPT"
echo "========================================"

# Case 1: 單核心 (Baseline)
echo ""
echo "[1/5] Running with 1 Thread..."
OMP_NUM_THREADS=1 python3 $TEST_SCRIPT

# Case 2: 4 核心 (預期甜蜜點)
echo ""
echo "[2/5] Running with 2 Threads..."
OMP_NUM_THREADS=2 python3 $TEST_SCRIPT

# Case 2: 4 核心 (預期甜蜜點)
echo ""
echo "[3/5] Running with 4 Threads..."
OMP_NUM_THREADS=4 python3 $TEST_SCRIPT

# Case 2: 4 核心 (預期甜蜜點)
echo ""
echo "[4/5] Running with 8 Threads..."
OMP_NUM_THREADS=8 python3 $TEST_SCRIPT

# Case 3: 16 核心 (測試擴展性極限)
echo ""
echo "[5/5] Running with 16 Threads..."
OMP_NUM_THREADS=16 python3 $TEST_SCRIPT

echo ""
echo "========================================"
echo "Benchmark Complete."