# Level 1: OpenMP Parallelism Development Log

## 1. Environment & Hardware Context
**Target Machine**: Intel Xeon Phi 7290F (Knights Landing)
- **Architecture**: Many-Core (眾核架構)
- **CPU**: 72 Physical Cores / 288 Threads
- **Frequency**: 1.50 GHz (Weak Cores)
- **Constraint**: 單核效能較弱，依賴大量平行化與高運算密度來發揮效能。

---

## 2. Implementation (C Code)
**File**: `csrc/step_apply_sequential.c`
**Method**: OpenMP Shared Memory Parallelism

主要修改：
1.  **RNG Thread Safety**: 使用 Strided Indexing (`i * 131`) 讓每個 thread 存取獨立的亂數池。
2.  **Data Privatization**: 將 `NeighborCandidate` 陣列宣告於 OpenMP 迴圈內部 (Stack memory)。
3.  **Directives**: `#pragma omp parallel for schedule(static)`

**Compilation Command**:
```bash
# 在 csrc folder 執行
make clean
make

**test 正確性&效能**
**在tests folder下執行**
正確性 
(test_basic.py 
# --- 加入這行 ---
np.random.seed(42)  # 固定種子，確保每次跑出來的 Random Pool 都一模一樣)
# 1. Run Sequential Baseline
OMP_NUM_THREADS=1 python3 python/tests/test_basic.py

# 2. Run Parallel Version (4 Threads)
OMP_NUM_THREADS=4 python3 python/tests/test_basic.py

效能
# 1. csh test_SW.csh


