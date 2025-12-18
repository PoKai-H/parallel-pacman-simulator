#!/bin/bash
#python/tests/run_all_experiments.sh


chmod +x *.sh 2>/dev/null


RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

echo "========================================================"
echo "🎯 FINAL PROJECT PERFORMANCE SUITE (AUTO-LOGGING)"
echo "   Logs will be saved to '$RESULTS_DIR/'"
echo "========================================================"

# --- Test 1: Micro-Benchmark (Sensitivity) ---
echo -e "\n[Test 1] Parameter Sensitivity (Grid/Agent/Thread Scaling)"
echo "--------------------------------------------------------"
./exp_01_micro_scaling.sh | tee $RESULTS_DIR/exp_01_log.txt

# --- Test 2: Level 2 (OpenMP Inter-Env) ---
echo -e "\n[Test 2] Level 2 Throughput Scaling"
echo "--------------------------------------------------------"
./exp_02_throughput.sh | tee $RESULTS_DIR/exp_02_log.txt

# --- Test 3: Level 3 (MPI) ---
echo -e "\n[Test 3] Level 3 MPI Scaling"
echo "--------------------------------------------------------"
./exp_03_mpi_scaling.sh | tee $RESULTS_DIR/exp_03_log.txt

# --- Test 4: Hybrid Architecture Trade-ff ---
echo -e "\n[Test 4] Hybrid Architecture Validation"
echo "--------------------------------------------------------"
./exp_04_hybrid_tradeoff.sh | tee $RESULTS_DIR/exp_04_log.txt

# --- Test 5: Hybrid Multilevel ---
echo -e "\n[Test 5] Hybrid Multilevel Validation"
echo "--------------------------------------------------------"
./exp_05_hybrid_multilevel.sh | tee $RESULTS_DIR/exp_05_log.txt

echo "========================================================"
echo "✅ Performance Suite Completed."
echo "   You can now generate plots from '$RESULTS_DIR/'"
echo "========================================================"