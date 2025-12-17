#!/bin/bash
# 存檔位置: python/tests/run_mpi_scaling.sh

PY_SCRIPT="../main_mpi.py"
LOG_FILE="mpi_scaling_results.txt"

# 輕量化設定，確保跑得快
STEPS=5
AGENTS=1024

echo "MPI Scaling Test Results" > $LOG_FILE
echo "Ranks,Throughput,Speedup,Efficiency" >> $LOG_FILE

echo "=========================================================="
echo "🚀 MPI Strong Scaling Test (Total Workload: 64 Envs)"
echo "   Settings: $AGENTS Agents/Env, $STEPS Steps"
echo "=========================================================="

export OMP_NUM_THREADS=1

# --- 1. 跑基準 (Rank=1) ---
echo "Running Baseline (1 Rank)..."

BASE_OUT=$(mpirun -np 1 python3 $PY_SCRIPT --n_envs_per_rank 64 --n_agents $AGENTS --steps $STEPS 2>&1)

# [MOD] 修改抓取邏輯：
# 1. grep 抓出那一行
# 2. awk -F': ' 以冒號加空白為分隔，取第2部分 (數字 + 單位)
# 3. 再用 awk '{print $1}' 取出純數字
if echo "$BASE_OUT" | grep -q "Final Total Throughput"; then
    BASE_FPS=$(echo "$BASE_OUT" | grep "Final Total Throughput" | awk -F': ' '{print $2}' | awk '{print $1}')
    echo " -> Baseline FPS: $BASE_FPS"
    echo "1,$BASE_FPS,1.0,100%" >> $LOG_FILE
else
    echo "❌ Baseline Failed! Output:"
    echo "$BASE_OUT"
    exit 1
fi

# --- 2. 跑擴展測試 ---
for R in 2 4 8 16 32 64; do
    ENVS_PER_RANK=$((64 / R))
    
    echo "Running with $R Ranks (Load: $ENVS_PER_RANK Envs/Rank)..."
    
    OUTPUT=$(mpirun -np $R --bind-to core python3 $PY_SCRIPT --n_envs_per_rank $ENVS_PER_RANK --n_agents $AGENTS --steps $STEPS 2>&1)
    
    if echo "$OUTPUT" | grep -q "Final Total Throughput"; then
        # [MOD] 同樣的修正邏輯
        FPS=$(echo "$OUTPUT" | grep "Final Total Throughput" | awk -F': ' '{print $2}' | awk '{print $1}')
        
        # 計算加速比
        SPEEDUP=$(echo "$FPS / $BASE_FPS" | bc -l)
        EFF=$(echo "$SPEEDUP / $R * 100" | bc -l)
        
        echo " -> $R Ranks FPS: $FPS | Speedup: x$(printf "%.2f" $SPEEDUP) | Eff: $(printf "%.0f" $EFF)%"
        echo "$R,$FPS,$SPEEDUP,$EFF" >> $LOG_FILE
    else
        echo "❌ Rank $R Failed! Output:"
        echo "$OUTPUT"
    fi
done

echo "=========================================================="
echo "✅ Done! Results saved to $LOG_FILE"
cat $LOG_FILE