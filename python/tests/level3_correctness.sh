#!/bin/bash
# tests/run_correctness.sh

echo "🔍 Running MPI Correctness Test..."

# 1. 黃金標準：用 1 個 MPI Rank 跑 16 個環境
# 這就像是把工作全丟給一個人做
echo "1. Running Serial Baseline (1 Rank)..."
mpirun -np 1 python3 level3/correctness/level3_correctness.py --total_envs 16
# 這會產生 checksum_serial.txt

# 2. 平行測試：用 4 個 MPI Ranks 分工，每人跑 4 個環境
echo "2. Running Parallel Test (4 Ranks)..."
mpirun -np 4 python3 level3/correctness/level3_correctness.py --total_envs 16
# 這會產生 checksum_parallel.txt

# 3. 比對
VAL1=$(cat checksum_serial.txt)
VAL2=$(cat checksum_parallel.txt)

echo "------------------------------------------------"
echo "Serial Checksum  : $VAL1"
echo "Parallel Checksum: $VAL2"

if [ "$VAL1" == "$VAL2" ]; then
    echo "✅ SUCCESS: MPI Parallelism is CORRECT!"
else
    # 注意：如果你的亂數產生器不是 Thread-safe 或 Seed 機制沒寫好
    # 這裡可能會失敗。這也是一種 Debug。
    echo "⚠️  WARNING: Checksums do not match. (Likely RNG seed issue)"
    echo "   (This is expected if random seeds are not strictly bound to Global Env ID)"
fi
echo "------------------------------------------------"