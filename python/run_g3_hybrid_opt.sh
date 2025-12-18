#!/bin/bash
# ---------------------------------------------------------
# Group 3: Hybrid Optimization
# ---------------------------------------------------------
TOTAL_ENVS=11520
STEPS=100
HOSTFILE="../host.txt"
OUTPUT_FILE="results/full_grid_search_results.csv"
HIDDEN_DIM=512

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Experiment_Group,Config_Name,Nodes,Ranks,L2_Threads(Env),L1_Threads(Agent),Total_Cores,Throughput" > $OUTPUT_FILE
fi

run_fixed_test() {
    GROUP=$1; NAME=$2; NODES=$3; RANKS=$4; L2=$5; L1=$6
    TOTAL_CORES=$((RANKS * L2 * L1))
    RANKS_PER_NODE=$((RANKS / NODES))
    
    echo "=== [$GROUP] $NAME ($NODES Nodes, $RANKS Ranks) ==="
    MPI_CMD="mpiexec --hostfile $HOSTFILE -n $RANKS --map-by ppr:$RANKS_PER_NODE:node --mca gds hash"

    set +e
    OUTPUT=$($MPI_CMD -x OMP_NESTED=TRUE -x OMP_MAX_ACTIVE_LEVELS=2 -x OMP_NUM_THREADS=$L2,$L1 \
        python3 benchmark_rl_simulation.py \
        --total_episodes $TOTAL_ENVS --max_steps $STEPS --n_agents 16 --hidden_dim $HIDDEN_DIM)
    set -e
    TP=$(echo "$OUTPUT" | grep "Throughput" | awk '{print $2}')
    [ -z "$TP" ] && TP="CRASHed"
    echo ">> Result: $TP"
    echo "$GROUP,$NAME,$NODES,$RANKS,$L2,$L1,$TOTAL_CORES,$TP" >> $OUTPUT_FILE
}

# --- Execution ---
run_fixed_test "G3_Hybrid_Opt" "HighDensity_32x2" 4 128 1 2
run_fixed_test "G3_Hybrid_Opt" "MediumDensity_16x4" 4 64 1 4