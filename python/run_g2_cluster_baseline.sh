#!/bin/bash
# ---------------------------------------------------------
# Group 2: Cluster Standard (Fixed 256 Cores)
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
    
    # cluster mode, include gds hash
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
# The parameters are `Nodes` `Ranks` `L2 Threads(envs)` `L1 Threads (agents)`
run_fixed_test "G2_Cluster_Std" "1_Dist_Thread_L2" 4 4 64 1
run_fixed_test "G2_Cluster_Std" "2_Dist_Nested_16x4" 4 4 16 4
# run_fixed_test "G2_Cluster_Std" "3_Dist_Nested_4x16" 4 4 4 16 # avoid this test
run_fixed_test "G2_Cluster_Std" "4_Cluster_Hybrid" 4 16 16 1
run_fixed_test "G2_Cluster_Std" "5_Cluster_Process_Max" 4 256 1 1