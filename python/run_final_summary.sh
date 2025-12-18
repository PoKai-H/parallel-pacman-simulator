#!/bin/bash
# ---------------------------------------------------------
# 02_final_summary.sh
# Final Grand Benchmark: The Champion Comparison
# ---------------------------------------------------------

# Benchmark Parameters
TOTAL_ENVS=11520
STEPS=100
HOSTFILE="../host.txt"
OUTPUT_FILE="results/final_summary_comparison.csv"
HIDDEN_DIM=512 

# Initialize CSV Header
echo "Config_Name,Total_Ranks,Threads_Per_Rank,Throughput" > $OUTPUT_FILE

run_test() {
    NAME=$1
    RANKS=$2
    THREADS=$3
    
    echo "==========================================================="
    echo "Running: $NAME (Ranks: $RANKS, Threads: $THREADS)"
    echo "==========================================================="
    
    # 1. OpenMP Settings
    export OMP_NUM_THREADS=$THREADS
    export OMP_PROC_BIND=FALSE
    unset OMP_PLACES
    
    # 2. Calculate Ranks Per Node (assuming 4 nodes)
    RANKS_PER_NODE=$((RANKS / 4))
    if [ "$RANKS_PER_NODE" -eq 0 ]; then RANKS_PER_NODE=1; fi

    # 3. Execute MPI
    OUTPUT=$(mpiexec --hostfile $HOSTFILE -n $RANKS \
        --map-by ppr:$RANKS_PER_NODE:node \
        --mca gds hash \
        -x OMP_NUM_THREADS \
        python3 benchmark_rl_simulation.py \
        --total_episodes $TOTAL_ENVS \
        --max_steps $STEPS \
        --n_agents 16 \
        --hidden_dim $HIDDEN_DIM)
        
    # 4. Parse Results
    TP=$(echo "$OUTPUT" | grep "Throughput" | awk '{print $2}')
    if [ -z "$TP" ]; then TP=0; fi
    
    echo "Result: $TP steps/s"
    echo "$NAME,$RANKS,$THREADS,$TP" >> $OUTPUT_FILE
}

# ---------------------------------------------------------
# Test Cases (The "Big Four" Comparison)
# ---------------------------------------------------------

# 1. Pure MPI (The Champion: 36 Ranks/Node) 👑
run_test "Pure_MPI_Champion" 144 1

# 2. Pure MPI (The Safe Runner-up: 32 Ranks/Node)
run_test "Pure_MPI_Stable" 128 1

# 3. Hybrid Balanced (16 Ranks, 18 Threads)
run_test "Hybrid_Balanced" 16 18

# 4. Pure OpenMP (4 Ranks, 72 Threads)
run_test "Pure_OpenMP" 4 72

echo "-----------------------------------------------------------"
echo "Summary Benchmark Finished! Check $OUTPUT_FILE"
echo "-----------------------------------------------------------"