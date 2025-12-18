#!/bin/bash

GRID_SIZES=(80 100 200)
AGENTS_LIST=(128 256 512)
STEPS=10000
THREADS_LIST=(1 2 4 8 16)

TARGET_SCRIPT="level1/speedup/worker_01_micro.py"
# =========================================

# PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

echo "==========================================="
echo "Starting Benchmark for Level 1 (Micro)"
echo "Target: $TARGET_SCRIPT"
echo "==========================================="

for G in "${GRID_SIZES[@]}"; do
    for A in "${AGENTS_LIST[@]}"; do
        echo ""
        echo "----------------------------------------------------------"
        echo "Config: Grid=${G}x${G}, Agents=${A}"
        printf "%-10s | %-20s\n" "Threads" "Throughput"
        echo "----------------------------------------------------------"
        
        for T in "${THREADS_LIST[@]}"; do
            
            OUTPUT=$(env OMP_NUM_THREADS=$T python3 "$TARGET_SCRIPT" --grid_size $G --n_agents $A --steps $STEPS 2>&1)
            
        
            if echo "$OUTPUT" | grep -q "Throughput:"; then
                TP=$(echo "$OUTPUT" | grep "Throughput:" | awk '{print $2}')
                printf "%-10s | %-20s\n" "$T" "$TP"
            else
                printf "%-10s | %-20s\n" "$T" "ERROR"
            fi
        done
    done
done
echo "==========================================="
echo "All Benchmarks Completed!"