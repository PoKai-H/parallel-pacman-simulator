import sys
import os
import subprocess
import shutil
import re
import pytest
from pathlib import Path

# =========================================================================
# 1. 動態路徑偵測
# =========================================================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
if not (PROJECT_ROOT / "pacman_env.py").exists():
    PROJECT_ROOT = CURRENT_FILE.parents[2]

# =========================================================================
# 2. 定義 MPI Worker (數學邏輯修正版)
# =========================================================================
MPI_WORKER_CODE = """
import sys
import os
import numpy as np
from mpi4py import MPI

sys.path.append(r"{project_root}")

try:
    from pacman_env import PacmanVecEnv
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import pacman_env. Path: {{sys.path}}")
    sys.exit(1)

def run_worker():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    total_envs = int(sys.argv[1])
    steps = int(sys.argv[2])
    
    # --- Load Balancing ---
    base_load = total_envs // size
    remainder = total_envs % size
    
    if rank < remainder:
        my_n_envs = base_load + 1
    else:
        my_n_envs = base_load
        
    # --- Simulation ---
    local_checksum = 0.0
    
    if my_n_envs > 0:
        # 初始化有圍牆的地圖
        grid = np.zeros((40, 40), dtype=np.int8)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        env = PacmanVecEnv(grid, n_envs=my_n_envs, n_agents=16)
        obs = env.reset()
        
        for _ in range(steps):
            actions = np.zeros((my_n_envs, 16), dtype=np.int32)
            next_obs, rewards, dones, _ = env.step(actions)
            
            # [FIX] 數學修正：心跳值必須乘以環境數量 (my_n_envs)
            # 這樣無論分給幾個 Rank，總 Checksum (Total Envs * Steps * 0.01) 才會守恆
            step_val = np.sum(next_obs) + (my_n_envs * 0.01)
            
            local_checksum += step_val

    # --- MPI Reduce ---
    total_checksum = comm.reduce(local_checksum, op=MPI.SUM, root=0)
    total_envs_run = comm.reduce(my_n_envs, op=MPI.SUM, root=0)
    
    if rank == 0:
        print(f"FINAL_CHECKSUM: {{total_checksum:.4f}}")
        print(f"TOTAL_ENVS_RUN: {{total_envs_run}}")

if __name__ == "__main__":
    run_worker()
"""

# =========================================================================
# 3. 測試輔助函數
# =========================================================================

def _get_mpiexec():
    mpi = shutil.which("mpiexec") or shutil.which("mpirun")
    if not mpi:
        pytest.skip("mpiexec/mpirun not found.")
    return mpi

def _create_worker_script(tmp_path):
    worker_file = tmp_path / "mpi_worker_temp.py"
    code = MPI_WORKER_CODE.format(project_root=str(PROJECT_ROOT))
    worker_file.write_text(code, encoding="utf-8")
    return str(worker_file)

def _run_mpi_case(np, total_envs, steps, script_path):
    mpi = _get_mpiexec()
    
    cmd = [
        mpi, "-n", str(np),
        sys.executable, script_path,
        str(total_envs),
        str(steps)
    ]
    
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    result = subprocess.run(
        cmd, 
        capture_output=True, 
        text=True, 
        env=env, 
        timeout=60
    )

    if result.returncode != 0:
        raise RuntimeError(f"MPI Run Failed (np={np}):\n{result.stderr}\n{result.stdout}")

    sum_match = re.search(r"FINAL_CHECKSUM:\s*([\d\.-]+)", result.stdout)
    count_match = re.search(r"TOTAL_ENVS_RUN:\s*(\d+)", result.stdout)
    
    if not sum_match or not count_match:
        raise ValueError(f"Could not parse output:\n{result.stdout}")
        
    return float(sum_match.group(1)), int(count_match.group(1))

# =========================================================================
# 4. 測試案例
# =========================================================================

@pytest.fixture
def worker_script(tmp_path):
    return _create_worker_script(tmp_path)

def test_l3_basic_consistency(worker_script):
    """Case 1: 基礎一致性 (NP=1 vs NP=4)"""
    total_envs = 16
    steps = 10
    
    print("\nRunning Baseline (NP=1)...")
    c1, n1 = _run_mpi_case(1, total_envs, steps, worker_script)
    
    print("Running Parallel (NP=4)...")
    c4, n4 = _run_mpi_case(4, total_envs, steps, worker_script)
    
    assert n1 == total_envs
    assert n4 == total_envs
    # 現在應該會完全一致
    assert abs(c1 - c4) < 1e-4, f"Mismatch! NP=1 checksum: {c1}, NP=4 checksum: {c4}"

def test_l3_remainder_distribution(worker_script):
    """Case 2: 餘數分配 (10 envs / 3 ranks)"""
    total_envs = 10
    np = 3
    steps = 5
    
    checksum, executed_envs = _run_mpi_case(np, total_envs, steps, worker_script)
    
    assert executed_envs == total_envs, \
        f"Workload lost! Expected {total_envs}, got {executed_envs}."
    assert checksum > 0

def test_l3_oversubscription(worker_script):
    """Case 3: 過度配置 (2 envs / 4 ranks)"""
    total_envs = 2
    np = 4
    steps = 5
    
    checksum, executed_envs = _run_mpi_case(np, total_envs, steps, worker_script)
    
    assert executed_envs == total_envs
    
    # 比較 NP=1 (2 envs) 和 NP=4 (2 envs distributed)
    base_sum, _ = _run_mpi_case(1, total_envs, steps, worker_script)
    assert abs(checksum - base_sum) < 1e-4, \
        f"Oversubscription mismatch! NP=1: {base_sum}, NP=4: {checksum}"

def test_l3_heavy_load(worker_script):
    """Case 4: 壓力測試 (64 envs / 4 ranks)"""
    total_envs = 64
    np = 4
    steps = 50
    
    checksum, executed_envs = _run_mpi_case(np, total_envs, steps, worker_script)
    
    assert executed_envs == 64
    assert checksum > 0

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))