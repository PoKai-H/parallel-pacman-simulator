# Case 1: 單核心
env OMP_NUM_THREADS=1 
echo "Running with 1 Thread..."
python3 test_perf.py


# Case 2: 4 核心 (Level 1 的甜蜜點通常在這裡)
env OMP_NUM_THREADS=4 
echo "Running with 4 Threads..."
python3 test_perf.py


# Case 3: 16 核心 (可能會因為 Overhead 變慢，這是正常的)
env OMP_NUM_THREADS=16 
echo "Running with 16 Threads..."
python3 test_perf.py