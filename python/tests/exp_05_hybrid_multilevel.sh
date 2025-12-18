#!/bin/bash
# csrc/run_mpi_levels.sh


PY_SCRIPT="../main_mpi.py"

echo "=========================================================="
echo "HPC Multi-Level Parallelism Benchmark (Total Cores: 64)"
echo "=========================================================="

# ----------------------------------------------------------------
# 1: (Pure MPI)
# ----------------------------------------------------------------

echo -e "\n[Experiment 1] Pure Level 3 (MPI Only)"
echo "Config: 64 Ranks, 1 Thread/Rank, 1 Env/Rank (Total 64 Envs)"
export OMP_NUM_THREADS=1
mpirun -np 64 --bind-to core python3 $PY_SCRIPT --n_envs_per_rank 1 --n_agents 4096 --steps 100

# ----------------------------------------------------------------
# 2: Level 3 + Level 2 (MPI + Env Parallelism)
# ----------------------------------------------------------------

echo -e "\n[Experiment 2] Level 3 + Level 2 (MPI + Env Parallelism)"
echo "Config: 4 Ranks, 16 Threads/Rank, 16 Envs/Rank (Total 64 Envs)"
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=spread,close

mpirun -np 4 --bind-to socket python3 $PY_SCRIPT --n_envs_per_rank 16 --n_agents 4096 --steps 100

# ----------------------------------------------------------------
# 3: Level 3 + Level 1 (MPI + Agent Parallelism)
# ----------------------------------------------------------------

echo -e "\n[Experiment 3] Level 3 + Level 1 (MPI + Agent Parallelism)"
echo "Config: 4 Ranks, 16 Threads/Rank, 1 Env/Rank (Total 4 Envs)"
export OMP_NUM_THREADS=16

mpirun -np 4 --bind-to socket python3 $PY_SCRIPT --n_envs_per_rank 1 --n_agents 4096 --steps 100

# ----------------------------------------------------------------
# 4: Full Hybrid (Level 3 + 2 + 1)
# ----------------------------------------------------------------

echo -e "\n[Experiment 4] Full Hybrid (L3 + L2 + L1)"
echo "Config: 4 Ranks, Nested OMP (4x4), 4 Envs/Rank (Total 16 Envs)"
export OMP_NUM_THREADS=4,4
export OMP_MAX_ACTIVE_LEVELS=2

mpirun -np 4 --bind-to socket python3 $PY_SCRIPT --n_envs_per_rank 4 --n_agents 4096 --steps 100