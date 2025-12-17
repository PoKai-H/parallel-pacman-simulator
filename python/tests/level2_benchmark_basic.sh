#!/bin/bash
# 存檔位置: python/tests/level2_benchmark_basic.sh

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
    echo "Current directory: $(pwd)"
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
    
    # [修正] 執行 Python 並抓取輸出，將 stderr 也導向 stdout
    OUTPUT=$(env OMP_NUM_THREADS=$t python3 "$TARGET_SCRIPT" 2>&1)
    
    # [修正] 檢查是否有成功抓到 "System Throughput"
    if echo "$OUTPUT" | grep -q "System Throughput"; then
        # 解析數字 (抓冒號後面的數字)
        SPS=$(echo "$OUTPUT" | grep "System Throughput" | awk -F': ' '{print $2}' | awk '{print $1}')
        
        # 計算 Speedup
        if [ "$t" -eq 1 ]; then
            BASELINE_SPS=$SPS
            SPEEDUP="1.00x (Base)"
        else
            # 使用 awk 做浮點數除法 (比 bc 更穩健)
            SPEEDUP=$(awk "BEGIN {printf \"%.2fx\", $SPS / $BASELINE_SPS}")
        fi

        # 印出結果
        printf "%-10s | %-20s | %-10s\n" "$t" "$SPS" "$SPEEDUP"
    else
        # [關鍵] 如果失敗，直接印出 Python 報錯訊息
        echo "❌ Thread $t FAILED! Python Output:"
        echo "$OUTPUT"
        echo "----------------------------------------------------------"
        # 為了不讓腳本完全中斷，我們繼續跑下一個，但你可以按 Ctrl+C 停下來
    fi
done

echo "----------------------------------------------------------"
echo "✅ Benchmark Complete."