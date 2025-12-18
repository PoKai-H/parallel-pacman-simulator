#!/bin/bash
# ---------------------------------------------------------
# Group 1: Single Node Baselines
# This script would take longer  to run
# ---------------------------------------------------------
TOTAL_ENVS=11520
STEPS=100
HOSTFILE="../host.txt"
OUTPUT_FILE="results/full_grid_search_results.csv"
HIDDEN_DIM=512

# If CSV doesn't exist, we start by creating headers.
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Experiment_Group,Config_Name,Nodes,Ranks,L2_Threads(Env),L1_Threads(Agent),Total_Cores,Throughput" > $OUTPUT_FILE
fi

run_fixed_test() {
    GROUP=$1; NAME=$2; NODES=$3; RANKS=$4; L2=$5; L1=$6
    TOTAL_CORES=$((RANKS * L2 * L1))
    RANKS_PER_NODE=$((RANKS / NODES))
    [ "$RANKS_PER_NODE" -eq 0 ] && RANKS_PER_NODE=1

    echo "=== [$GROUP] $NAME ($NODES Nodes, $RANKS Ranks) ==="
    
    # single node
    MPI_CMD="mpiexec -n $RANKS"

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
# The parameters are `Nodes` `Ranks` `L2 Threads(envs)` `L1 Threads (agents)`
run_fixed_test "G1_SingleNode" "1_Pure_Thread_L2" 1 1 64 1
run_fixed_test "G1_SingleNode" "2_Nested_Thread_16x4" 1 1 16 4
#run_fixed_test "G1_SingleNode" "3_Nested_Thread_4x16" 1 1 4 16
run_fixed_test "G1_SingleNode" "4_Hybrid_Standard" 1 4 16 1
run_fixed_test "G1_SingleNode" "5_Pure_Process" 1 64 1 1