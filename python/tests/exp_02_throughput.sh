#!/bin/bash


TARGET_SCRIPT="level2/speedup/worker_02_throughput.py"


THREADS_LIST=(1 2 4 8 16 32)



export PYTHONPATH=$PYTHONPATH:$(pwd)/..


if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Error: Cannot find $TARGET_SCRIPT"
    echo "Please run this script from 'python/tests/' directory."
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "=========================================================="
echo "Level 2 Performance Benchmark (Throughput & Speedup)"
echo "Target: $TARGET_SCRIPT"
echo "=========================================================="
printf "%-10s | %-20s | %-10s\n" "Threads" "Throughput (Steps/s)" "Speedup"
echo "----------------------------------------------------------"


BASELINE_SPS=0

for t in "${THREADS_LIST[@]}"; do
    
    OUTPUT=$(env OMP_NUM_THREADS=$t python3 "$TARGET_SCRIPT" 2>&1)
    
    if echo "$OUTPUT" | grep -q "System Throughput"; then
        SPS=$(echo "$OUTPUT" | grep "System Throughput" | awk -F': ' '{print $2}' | awk '{print $1}')
        
        if [ "$t" -eq 1 ]; then
            BASELINE_SPS=$SPS
            SPEEDUP="1.00x (Base)"
        else
            SPEEDUP=$(awk "BEGIN {printf \"%.2fx\", $SPS / $BASELINE_SPS}")
        fi

        printf "%-10s | %-20s | %-10s\n" "$t" "$SPS" "$SPEEDUP"
    else
        echo "Thread $t FAILED! Python Output:"
        echo "$OUTPUT"
        echo "----------------------------------------------------------"
    fi
done

echo "----------------------------------------------------------"
echo "Benchmark Complete."