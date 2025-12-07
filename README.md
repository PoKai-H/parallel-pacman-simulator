# 📘 CSCI 5451 Final Project — Team Roles & Responsibilities

## Po-Kai — Huang Architect / Documentation / RL Designer

### **Responsibilities**
- Design and maintain the full environment architecture  
  (`common.h`, sequential kernel, state transitions)
- Implement Python wrappers (`PacmanEnv`, `VectorEnv`) for calling C kernels
- Provide baseline policies (random / heuristic) for testing & benchmarking
- Define experiment protocols for Level 1 / Level 2 / Level 3 parallelism  
  (inputs, seeds, measurement rules)
- Integrate results from A/B/C/D and write the core analysis
- Write major sections of the final report:
  - Problem definition  
  - Environment & architecture  
  - Parallel hierarchy  
  - Experimental results & discussion
- Prepare demo scripts and presentation materials
- *(Optional extension)* Add RL / constraint-based experiments to showcase extensibility

---

## 🅰️ Team Member A — Level-1 Parallelism (Agent-Level OpenMP)

### **Responsibilities**
- Implement `step_env_apply_level1.c`
- Add OpenMP parallelism to the ghost-update loop
- Validate correctness vs sequential kernel (multi-step, fixed seed)
- Run thread-scaling experiments (threads = 1, 2, 4, 8, ...)
- Produce Level-1 performance plots (speedup & efficiency)
- Provide Level-1 results summary for the final report

---

## 🅱️ Team Member B — Level-2 Parallelism (Environment-Level OpenMP)

### **Responsibilities**
- Implement `step_env_apply_level2.c`
- Parallelize the environment batch loop using OpenMP
- Validate correctness: batch sequential vs Level-2 batch
- Run multi-env scaling experiments (e.g., n_envs = 1, 4, 16, 32)
- Generate Level-2 speedup & efficiency plots
- Provide Level-2 results summary for the final report

---

## 🅲 Team Member C — Level-3 Parallelism (Episode-Level MPI)

### **Responsibilities**
- Implement `step_env_apply_level3.c`
- Divide episodes across MPI ranks and orchestrate parallel execution
- Validate correctness: MPI(np=1) ≡ single-process sequential
- Run strong-scaling experiments (np = 1, 2, 4, 8, ...)
- Produce Level-3 speedup & efficiency graphs
- Provide MPI results summary for the final report

---

## 🅳 Team Member D — Testing / Benchmarking / Plotting Engineer

### **Responsibilities**
- Build the full correctness testing suite:
  - `test_level1.py`
  - `test_level2.py`
  - `test_level3.py`
- Create benchmarking scripts:
  - `benchmark_level1.py`
  - `benchmark_level2.py`
  - `benchmark_mpi.sh`
- Ensure deterministic runs (seed control, reproducibility)
- Generate all performance visualizations (matplotlib):
  - Level-1 thread scaling  
  - Level-2 environment scaling  
  - MPI scaling curves  
- Contribute reproducibility documentation to README

---

# 🔧 Overall Workflow Summary

Po-Kai → Environment architect + sequential baseline + Python interface
A → Level-1 (per-agent OpenMP)
B → Level-2 (multi-environment OpenMP)
C → Level-3 (MPI per episode)
D → Testing, benchmarking, plotting, reproducibility
All → Integration, report, and final presentation


# 📌 Notes for the Team
- Sequential version is the **ground truth**.  
  All parallel implementations must match it exactly under fixed seeds.
- Testing and validation are *mandatory* across all levels.
- Benchmark and speedup analysis form a major portion of the final grade.
- This division ensures equal workload, clear ownership, and measurable deliverables.


# Experimental Configs
Level 1 (Agent Parallelism):
- n_agents = [16, 32, 64, 128]
- grid_size = 40x40
- steps_per_episode = 100
- episodes = 1
- threads = [1,2,4,8,16]

Level 2 (Environment Parallelism):
- n_envs = [1, 4, 8, 16, 32]
- n_agents = 16
- steps = 100
- threads = [1,2,4,8,16]

Level 3 (Episode MPI Parallel):
- total_episodes = 256
- np = [1,2,4,8]
- n_envs_per_rank = 1


# Github workflow
### only once when creating branch
git clone 

git checkout develop
git pull origin develop

git checkout -b feature/level1-A
git checkout -b feature/level2-B
git checkout -b feature/level3-C
git checkout -b feature/testing-D

git push origin <branch>


### push changes
git add .
git commin -m "..."
git push origin <branch>

than create pr -> merge with develop

### keeping branch updated with develop
git checkout <branch>
git pull origin develop
git push origin <branch>
