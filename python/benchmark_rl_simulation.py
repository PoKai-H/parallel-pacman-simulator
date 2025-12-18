import time
import argparse
import numpy as np
import sys
import os
from mpi4py import MPI

# ==========================================================
# 1. Path Setup
#    Ensure we can import 'pacman_env' regardless of where
#    this script is executed.
# ==========================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from pacman_env import PacmanVecEnv
except ImportError:
    # Fallback: look in the parent directory
    sys.path.append(os.path.dirname(current_dir))
    try:
        from pacman_env import PacmanVecEnv
    except ImportError:
        print("CRITICAL ERROR: Cannot find pacman_env.py")
        sys.exit(1)

def simulate_neural_network_inference(batch_size, input_dim, hidden_dim, output_dim, iterations=1):
    """
    Simulates the computational load of a Deep Neural Network (DNN) inference.
    
    This function uses Matrix Multiplication (MatMul) to consume CPU cycles 
    and memory bandwidth, mimicking a real Reinforcement Learning policy 
    forward pass.
    
    Args:
        batch_size (int): Number of environments (rows in input matrix).
        input_dim (int): Dimension of the observation vector.
        hidden_dim (int): Size of the hidden layer (controls compute intensity).
        output_dim (int): Number of actions.
        iterations (int): Number of forward passes to simulate network depth.
    
    Returns:
        np.ndarray: Simulated actions (indices).
    """
    # Generate random input data (Batch Size, Input Dim)
    # Generating data here stresses memory bandwidth, similar to loading obs.
    x = np.random.random((batch_size, input_dim)).astype(np.float32)
    
    # Simulate Weights (Input -> Hidden)
    w1 = np.random.random((input_dim, hidden_dim)).astype(np.float32)
    
    # Simulate Weights (Hidden -> Output)
    w2 = np.random.random((hidden_dim, output_dim)).astype(np.float32)

    # Perform Forward Pass
    out = None
    for _ in range(iterations):
        # Layer 1: MatMul + Activation (Simulating ReLU)
        # np.dot utilizes BLAS/MKL, which respects OMP_NUM_THREADS
        h = np.maximum(0, np.dot(x, w1)) 
        
        # Layer 2: MatMul
        out = np.dot(h, w2)
        
    # Select Actions (Simulating Argmax/Softmax)
    actions = np.argmax(out, axis=1).astype(np.int32)
    return actions

def main():
    # ==========================================================
    # 2. MPI Initialization
    # ==========================================================
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ==========================================================
    # 3. Argument Parsing
    # ==========================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_episodes", type=int, required=True, help="Total number of environments across all ranks")
    parser.add_argument("--max_steps", type=int, default=100, help="Simulation steps per environment")
    parser.add_argument("--n_agents", type=int, default=16, help="Number of agents per environment")
    
    # Arguments for controlling the "Fake" Neural Network load
    parser.add_argument("--hidden_dim", type=int, default=256, help="Size of hidden layer to simulate compute load")
    parser.add_argument("--compute_iter", type=int, default=1, help="Number of forward passes per step")
    
    args = parser.parse_args()

    # ==========================================================
    # 4. Load Balancing
    #    Distribute total workload evenly among ranks.
    # ==========================================================
    base_load = args.total_episodes // size
    remainder = args.total_episodes % size

    if rank < remainder:
        local_n_envs = base_load + 1
    else:
        local_n_envs = base_load

    # ==========================================================
    # 5. Environment Initialization
    # ==========================================================
    env = None
    INPUT_DIM = 512 # Simulated feature size
    OUTPUT_DIM = 5  # Pacman actions (Up, Down, Left, Right, Stop)
    
    if local_n_envs > 0:
        # Create a grid with walls to ensure sensors detect something
        grid = np.zeros((40, 40), dtype=np.int8)
        grid[0, :] = 1; grid[-1, :] = 1
        grid[:, 0] = 1; grid[:, -1] = 1
        
        # Initialize C++ Environment
        env = PacmanVecEnv(grid, n_envs=local_n_envs, n_agents=args.n_agents)
        
        # Get initial observations
        obs = env.reset() 

    # ==========================================================
    # 6. Synchronization (Start Timer)
    # ==========================================================
    comm.Barrier()
    start_time = time.time()

    # ==========================================================
    # 7. Main Simulation Loop
    # ==========================================================
    if local_n_envs > 0 and env is not None:
        for _ in range(args.max_steps):
            
            # --- Phase A: Neural Network Inference (Compute Bound) ---
            # This simulates the CPU load of deciding actions.
            # Larger 'local_n_envs' or 'hidden_dim' increases this load.
            actions_flat = simulate_neural_network_inference(
                batch_size=local_n_envs, 
                input_dim=INPUT_DIM, 
                hidden_dim=args.hidden_dim, 
                output_dim=OUTPUT_DIM,
                iterations=args.compute_iter
            )
            
            # Broadcast the single action decision to all agents in the env
            # (In a real scenario, we would compute per-agent, but this 
            # is sufficient for load testing).
            actions = np.tile(actions_flat[:, None], (1, args.n_agents)).astype(np.int32)
            
            # --- Phase B: Environment Step (Physics/Memory Bound) ---
            # This calls the optimized C++/CUDA/OpenMP backend.
            obs, rewards, dones, infos = env.step(actions)

    # ==========================================================
    # 8. Synchronization (Stop Timer) & Reporting
    # ==========================================================
    comm.Barrier()
    end_time = time.time()

    # Only Rank 0 prints the final report
    if rank == 0:
        total_time = end_time - start_time
        total_frames = args.total_episodes * args.max_steps
        throughput = total_frames / total_time
        
        print(f"========================================")
        print(f"Fake RL Training Benchmark:")
        print(f"  Ranks: {size}")
        print(f"  Total Envs: {args.total_episodes}")
        print(f"  Steps per Env: {args.max_steps}")
        print(f"  NN Configuration: {INPUT_DIM} -> {args.hidden_dim} -> {OUTPUT_DIM}")
        print(f"----------------------------------------")
        print(f"Results:")
        print(f"  Total Time: {total_time:.4f} s")
        # The shell script 'grep' looks for this specific string:
        print(f"  Throughput: {throughput:.2f} steps/s") 
        print(f"========================================")

if __name__ == "__main__":
    main()