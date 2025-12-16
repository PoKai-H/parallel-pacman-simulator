 #!/bin/bash

# ================= 設定區 =================
# 設定你想跑的 Grid Size 列表
GRID_SIZES=(80 100 200)

# 設定你想跑的 Agents 數量列表
AGENTS_LIST=(128 256 512)

# 設定你想跑的 Steps
STEPS=10000

# 設定你想跑的 Thread 數列表
THREADS_LIST=(1 2 4 8 16)

# 指定 Python 腳本的路徑 (相對路徑)
TARGET_SCRIPT="level1/speedup/test_level1_multipleSetting.py"

# 設定輸出結果的資料夾 (修正錯字 resutls -> results)
OUTPUT_DIR="../results"
# =========================================

# 1. [關鍵修正] 設定 PYTHONPATH
# 這樣 Python 才知道要往上一層 (..) 去找 pacman_env.py
# 假設此 .sh 檔位於 python/tests/ 資料夾下
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# 2. 自動建立輸出資料夾 (如果不存在)
mkdir -p "$OUTPUT_DIR"

echo "==========================================="
echo "Starting Benchmark for Level 1"
echo "Target Script: $TARGET_SCRIPT"
echo "Output Directory: $OUTPUT_DIR"
echo "==========================================="

# 第一層迴圈：不同的 Grid Size
for G in "${GRID_SIZES[@]}"; do
    # 第二層迴圈：不同的 Agent 數量
    for A in "${AGENTS_LIST[@]}"; do
        
        # 定義輸出的檔案名稱
        OUT_FILE="${OUTPUT_DIR}/result_G${G}_A${A}.txt"
        
        # 建立新檔案並寫入檔頭
        echo "Benchmark Report" > "$OUT_FILE"
        echo "Grid: ${G}x${G}, Agents: ${A}, Steps: ${STEPS}" >> "$OUT_FILE"
        echo "Date: $(date)" >> "$OUT_FILE"
        echo "-------------------------------------------" >> "$OUT_FILE"
        
        # 第三層迴圈：不同的 Threads
        for T in "${THREADS_LIST[@]}"; do
            echo "[Running] Grid=$G, Agents=$A, Threads=$T ..."
            
            # 寫入分隔線到 Log 檔
            echo "" >> "$OUT_FILE"
            echo ">>> [OMP_NUM_THREADS=$T] <<<" >> "$OUT_FILE"
            
            # 執行 Python
            # 注意：你的 Python 腳本必須使用 argparse 接收這些參數
            env OMP_NUM_THREADS=$T python3 "$TARGET_SCRIPT" \
                --grid_size $G \
                --n_agents $A \
                --steps $STEPS >> "$OUT_FILE" 2>&1
                
        done
        
        echo "✅ Saved to: $OUT_FILE"
    done
done

echo "All Benchmarks Completed!"