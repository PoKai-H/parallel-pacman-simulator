#!/bin/bash

# ================= 設定區 =================
# 設定你想跑的 Grid Size 列表 (例如 80, 100, 200)
GRID_SIZES=(80 100 200)

# 設定你想跑的 Agents 數量列表 (例如 256, 512, 1024)
AGENTS_LIST=(128 256 512)

# 設定你想跑的 Steps (固定或變動皆可)
STEPS=10000

# 設定你想跑的 Thread 數列表
THREADS_LIST=(1 2 4 8 16)
# =========================================

echo "Starting Benchmark..."

# 第一層迴圈：不同的 Grid Size
for G in "${GRID_SIZES[@]}"; do
    # 第二層迴圈：不同的 Agent 數量
    for A in "${AGENTS_LIST[@]}"; do
        
        # 定義輸出的檔案名稱，讓檔名包含參數資訊
        OUT_FILE="./resutls/result_G${G}_A${A}.txt"
        
        # 清空或是建立新檔案，並寫入檔頭
        echo "Running Benchmark for Grid=${G}x${G}, Agents=${A}" > "$OUT_FILE"
        echo "Date: $(date)" >> "$OUT_FILE"
        echo "-------------------------------------------" >> "$OUT_FILE"
        
        # 第三層迴圈：不同的 Threads
        for T in "${THREADS_LIST[@]}"; do
            echo "Running with OMP_NUM_THREADS=$T ..."
            
            # 寫入分隔線跟標題到檔案中，方便閱讀
            echo "" >> "$OUT_FILE"
            echo ">>> [OMP_NUM_THREADS=$T] <<<" >> "$OUT_FILE"
            
            # 設定環境變數並執行 Python，傳入對應參數
            # 2>&1 確保錯誤訊息也會被寫入 log
            env OMP_NUM_THREADS=$T python3 test_perf_all.py \
                --grid_size $G \
                --n_agents $A \
                --steps $STEPS >> "$OUT_FILE" 2>&1
                
        done
        
        echo "Finished: $OUT_FILE"
    done
done

echo "All Benchmarks Completed!"