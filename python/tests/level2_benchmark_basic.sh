#!/bin/bash

# ================= 設定區 =================
# Python 測試腳本的路徑 (相對路徑)
TARGET_SCRIPT="level2/speedup/test_level2.py"

# 你想測試的 Thread 數量
THREADS_LIST=(1 2 4 8 16 32)

# =========================================

# 1. 設定 PYTHONPATH 以便找到 pacman_env.py
# 假設此腳本在 python/tests/ 下執行，上一層就是 python/
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# 檢查腳本是否存在
if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "❌ Error: Cannot find $TARGET_SCRIPT"
    echo "Please run this script from 'python/tests/' directory."
    exit 1
fi

echo "=========================================================="
echo "🚀 Level 2 Performance Benchmark (Throughput & Speedup)"
echo "Target: $TARGET_SCRIPT"
echo "=========================================================="
printf "%-10s | %-20s | %-10s\n" "Threads" "Throughput (Steps/s)" "Speedup"
echo "----------------------------------------------------------"

# 變數用來存 Baseline (Thread=1) 的數值
BASELINE_SPS=0

for t in "${THREADS_LIST[@]}"; do
    
    # 1. 執行 Python 並抓取輸出
    # 使用 grep 抓取包含 "System Throughput" 的那一行
    OUTPUT=$(env OMP_NUM_THREADS=$t python3 "$TARGET_SCRIPT" 2>&1)
    
    # 2. 解析輸出中的數字 (假設輸出格式為: "System Throughput: 1234.56 EnvSteps/sec")
    # 使用 awk 抓取冒號後面的數字 (第 3 個欄位)
    SPS=$(echo "$OUTPUT" | grep "System Throughput" | awk '{print $3}')
    
    # 如果抓不到數字 (例如 Segfault)，設為 0
    if [ -z "$SPS" ]; then
        SPS=0
        SPEEDUP="N/A"
        # 印出錯誤訊息以便 Debug
        echo "Error output for $t threads:"
        echo "$OUTPUT"
    else
        # 3. 計算 Speedup
        if [ "$t" -eq 1 ]; then
            BASELINE_SPS=$SPS
            SPEEDUP="1.00x (Base)"
        else
            # 使用 awk 做浮點數除法
            if (( $(echo "$BASELINE_SPS > 0" | bc -l) )); then
                SPEEDUP=$(awk "BEGIN {printf \"%.2fx\", $SPS / $BASELINE_SPS}")
            else
                SPEEDUP="N/A"
            fi
        fi
    fi

    # 4. 印出結果 (格式化輸出)
    printf "%-10s | %-20s | %-10s\n" "$t" "$SPS" "$SPEEDUP"
done

echo "----------------------------------------------------------"
echo "✅ Benchmark Complete."