#!/bin/bash
# 存檔位置: python/tests/run_performance_suite.sh

# 設定權限
chmod +x *.sh 2>/dev/null

# 建立存放結果的資料夾
RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

echo "========================================================"
echo "🎯 FINAL PROJECT PERFORMANCE SUITE (AUTO-LOGGING)"
echo "   Logs will be saved to '$RESULTS_DIR/'"
echo "========================================================"

# --- Test 1: Level 1 (OpenMP Intra-Env) ---
echo -e "\n[Test 1] Level 1 Baseline vs Optimized"
echo "--------------------------------------------------------"
# 使用 tee 同時輸出到螢幕和檔案
./level1_benchmark_single_setting.sh | tee $RESULTS_DIR/level1_log.txt

# --- Test 2: Level 2 (OpenMP Inter-Env) ---
echo -e "\n[Test 2] Level 2 Throughput Scaling"
echo "--------------------------------------------------------"
./level2_benchmark_basic.sh | tee $RESULTS_DIR/level2_log.txt

# --- Test 3: Level 3 (Pure MPI) ---
echo -e "\n[Test 3] MPI Strong Scaling"
echo "--------------------------------------------------------"
# 注意：level3_speedup.sh 本身會產出 mpi_scaling_results.txt
# 我們還是存一份 log 備查
./level3_speedup.sh | tee $RESULTS_DIR/level3_log.txt
# 把產生的 CSV 也搬進 results 資料夾 (如果存在)
[ -f mpi_scaling_results.txt ] && mv mpi_scaling_results.txt $RESULTS_DIR/mpi_scaling.csv

# --- Test 4: Hybrid Architecture (16 Cores) ---
echo -e "\n[Test 4] Hybrid Validation (16 Cores)"
echo "--------------------------------------------------------"
./level12_16c_exp.sh | tee $RESULTS_DIR/hybrid_16c_log.txt

# --- Test 5: Full System Stress Test (64 Cores) ---
echo -e "\n[Test 5] Heavy Load Stability (64 Cores)"
echo "--------------------------------------------------------"
./level12_64c_exp.sh | tee $RESULTS_DIR/hybrid_64c_log.txt

# --- Optional: Correctness Tests ---
echo -e "\n[Test 6] Running Correctness Tests (Pytest)"
echo "--------------------------------------------------------"
# 將 pytest 結果也存起來
python3 test_mechanics.py > $RESULTS_DIR/correctness_log.txt 2>&1
echo "Correctness tests completed. Check $RESULTS_DIR/correctness_log.txt"

echo "========================================================"
echo "✅ All Benchmarks Completed."
echo "   Generating Plots..."
echo "========================================================"

# 自動呼叫畫圖腳本 (如果有的話)
if [ -f "plot_results.py" ]; then
    python3 plot_results.py
    echo "📊 Plots generated in '$RESULTS_DIR/'"
else
    echo "⚠️  plot_results.py not found. Please create it to generate plots."
fi