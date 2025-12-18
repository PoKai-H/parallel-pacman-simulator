#!/bin/bash

# ==========================================
# 64-Core Thread Allocation Experiment
# ==========================================


make bench > /dev/null

echo "================================================================================="
echo "| Strategy | L2 (Envs) | L1 (Agents)| Total Threads | Throughput (steps/s) |"
echo "================================================================================="

# L2 # of threads, L1 # of threads
CONFIGS=(
    "64,1"   
    "32,2"   
    "16,4"   
    "8,8"    
    "4,16"   
    "2,32"   
    "1,64"   
)


export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close  
export OMP_PLACES=threads

for CONF in "${CONFIGS[@]}"; do
    
    L2=$(echo $CONF | cut -d',' -f1)
    L1=$(echo $CONF | cut -d',' -f2)
    
    # benchmark
    RESULT=$(env OMP_NUM_THREADS=$CONF ./benchmark | grep "Throughput" | awk '{print $2}')
    
    printf "| %-8s | %-9s | %-10s | %-13s | \033[1;32m%-20s\033[0m |\n" \
           "$CONF" "$L2" "$L1" "$((L2 * L1))" "$RESULT"
done

echo "================================================================================="