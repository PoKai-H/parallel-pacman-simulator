#!/bin/bash

# ====================================================
# 64-Core Architecture Validation
# ====================================================


export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close
export OMP_PLACES=threads

# 共用參數
AGENTS_A=1024  
AGENTS_B=4096  
STEPS=100

echo "=========================================================="
echo "64-Core Hybrid Parallelism Experiment"
echo "=========================================================="

ENVS_A=64

echo ""
echo "Scenario A: Massive Batch (N_ENVS=$ENVS_A)"
echo "----------------------------------------------------------"

# 1: Level 2 [64 Env x 1 Thread]
echo -n "1. Strategy [64, 1] (Pure Level 2): "
env OMP_NUM_THREADS=64,1 python3 worker_04_hybrid.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"

# 2: [16 Env x 4 Threads]
echo -n "2. Strategy [16, 4] (Hybrid Mode ): "
env OMP_NUM_THREADS=16,4 python3 worker_04_hybrid.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"

# 3: Level 1 [1 Env x 64 Threads]
echo -n "3. Strategy [ 1,64] (Pure Level 1): "
env OMP_NUM_THREADS=1,64 python3 worker_04_hybrid.py --n_envs $ENVS_A --n_agents $AGENTS_A --steps $STEPS | grep "Throughput"


# ====================================================
# B：(Low Latency
# ====================================================
ENVS_B=4

echo ""
echo "Scenario B: Small Batch / Latency Critical (N_ENVS=$ENVS_B)"
echo "----------------------------------------------------------"

# 1: Level 2 [64, 1] 
echo -n "1. Strategy [64, 1] (Pure Level 2): "
env OMP_NUM_THREADS=64,1 python3 worker_04_hybrid.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

# 2: [4, 16] 

echo -n "2. Strategy [ 4,16] (Hybrid Mode ): "
env OMP_NUM_THREADS=4,16 python3 worker_04_hybrid.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

# 3: Level 1 [1, 64]

echo -n "3. Strategy [ 1,64] (Pure Level 1): "
env OMP_NUM_THREADS=1,64 python3 worker_04_hybrid.py --n_envs $ENVS_B --n_agents $AGENTS_B --steps $STEPS | grep "Throughput"

echo "=========================================================="