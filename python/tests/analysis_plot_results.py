# 存檔位置: python/tests/plot_results.py
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

RESULTS_DIR = "results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def parse_level1_log(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping Plot 1.")
        return []

    experiments = []
    current_config = "Unknown"
    current_threads = []
    current_tps = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            
            if "Config:" in line:
                if current_threads:
                    experiments.append((current_config, current_threads, current_tps))
                    current_threads = []
                    current_tps = []
                
            
                current_config = line.split("Config:")[1].strip()

            
            if "|" in line and "Threads" not in line and "Throughput" not in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    try:
                        t = int(parts[0].strip())
                        tp_str = parts[1].strip()
                        if tp_str == "ERROR": continue
                        tp = float(tp_str)
                        
                        current_threads.append(t)
                        current_tps.append(tp)
                    except ValueError:
                        continue
    
    if current_threads:
        experiments.append((current_config, current_threads, current_tps))
            
    return experiments

def parse_level2_log(filepath):
    threads = []
    speedups = []
    
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping Plot 2.")
        return None, None
        
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) == 3:
                try:
                    t = int(parts[0].strip())
                    s_str = parts[2].strip().replace('x', '').split(' ')[0]
                    if s_str == "N/A": continue
                    s = float(s_str)
                    threads.append(t)
                    speedups.append(s)
                except ValueError:
                    continue
    return threads, speedups

def parse_hybrid_log(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping Hybrid Plot.")
        return None

    data = {} # {Scenario: {Strategy: Throughput}}
    current_scenario = "Unknown"
    
    with open(filepath, 'r') as f:
        for line in f:
            if "Scenario" in line:
                if "Scenario A" in line: current_scenario = "High Batch (A)"
                if "Scenario B" in line: current_scenario = "Low Batch (B)"
            
            if "Strategy" in line and "Throughput" in line:
                strat_match = re.search(r"Strategy \[(.*?)\]", line)
                tp_match = re.search(r"Throughput: ([\d.]+) env_steps/s", line)
                
                if strat_match and tp_match:
                    strat = strat_match.group(1).replace(" ", "")
                    tp = float(tp_match.group(1))
                    
                    if current_scenario not in data: data[current_scenario] = {}
                    data[current_scenario][strat] = tp
    return data

def parse_exp05_log(filepath):
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping Plot 5.")
        return None

    data = {}
    current_exp_name = "Unknown"
    
    name_map = {
        "Pure Level 3": "Pure MPI (L3)",
        "Level 3 + Level 2": "MPI + Env (L3+L2)",
        "Level 3 + Level 1": "MPI + Agent (L3+L1)",
        "Full Hybrid": "Full Hybrid (L3+L2+L1)"
    }

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if "[Experiment" in line:
                for key, val in name_map.items():
                    if key in line:
                        current_exp_name = val
                        break
            
            if "Final Total Throughput" in line:
                match = re.search(r"Throughput:\s*([\d\.]+)", line)
                if match:
                    tp = float(match.group(1))
                    data[current_exp_name] = tp

    return data

def plot_level1(experiments):

    if not experiments: return

    plt.figure(figsize=(10, 6))
    
    
    for config_name, threads, throughputs in experiments:
        plt.plot(threads, throughputs, 'o-', linewidth=2, label=config_name)

    plt.xlabel("Number of Threads (OpenMP)")
    plt.ylabel("Throughput (Steps/Sec)")
    plt.title("Level 1 Scaling: Agent Parallelism (Micro-Benchmark)")
    plt.legend()
    plt.grid(True)
    
    if experiments:
        all_threads = sorted(list(set(sum([e[1] for e in experiments], []))))
        plt.xticks(all_threads)

    output_path = f"{RESULTS_DIR}/plot_level1_scaling.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Generated: {output_path}")

def plot_level2(threads, speedups):
    if not threads: return
    plt.figure(figsize=(8, 5))
    plt.plot(threads, speedups, 's-', color='orange', linewidth=2, label="Measured")
    plt.plot(threads, threads, 'k--', label="Ideal Linear")
    
    plt.xlabel("Number of Threads (OpenMP)")
    plt.ylabel("Speedup Factor")
    plt.title("Level 2 Scaling: Environment Parallelism")
    plt.legend()
    plt.grid(True)
    plt.xticks(threads)
    plt.savefig(f"{RESULTS_DIR}/plot_level2_speedup.png")
    plt.close()
    print("Generated: plot_level2_speedup.png")

def plot_mpi(csv_path):
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found, skipping MPI Plot.")
        return
    
    try:
        df = pd.read_csv(csv_path, skiprows=1)
        
        df.columns = [c.strip() for c in df.columns]

        plt.figure(figsize=(8, 5))
        plt.plot(df['Ranks'], df['Speedup'], 'D-', color='green', linewidth=2, label="MPI Measured")
        plt.plot(df['Ranks'], df['Ranks'], 'k--', label="Ideal Linear")
        
        plt.xlabel("Number of MPI Ranks")
        plt.ylabel("Speedup Factor")
        plt.title("Level 3 Scaling: Distributed MPI")
        plt.legend()
        plt.grid(True)
        plt.xticks(df['Ranks']) 
        plt.savefig(f"{RESULTS_DIR}/plot_level3_mpi.png")
        plt.close()
        print("Generated: plot_level3_mpi.png")
        
    except Exception as e:
        print(f"Error plotting MPI: {e}")
        if 'df' in locals():
            print(f"Read columns: {df.columns.tolist()}")

def plot_hybrid_bar(data, filename, title_suffix):
    if not data: return
    
    scenarios = list(data.keys())
    strategies = list(next(iter(data.values())).keys())
    
    x = np.arange(len(scenarios))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for strat in strategies:
        measurements = []
        for sc in scenarios:
            measurements.append(data.get(sc, {}).get(strat, 0))
        
        offset = width * multiplier
        rects = ax.bar(x + offset, measurements, width, label=f"Strat {strat}")
        ax.bar_label(rects, padding=3, fmt='%.0f')
        multiplier += 1

    ax.set_ylabel('Throughput (env_steps/s)')
    ax.set_title(f'Hybrid Architecture Comparison ({title_suffix})')
    ax.set_xticks(x + width, scenarios)
    ax.legend(loc='upper left', ncols=1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f"{RESULTS_DIR}/{filename}")
    plt.close()
    print(f"Generated: {filename}")

def plot_exp05(data):

    if not data: return

    labels = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(labels, values, color=colors[:len(labels)])

    plt.ylabel("Total Throughput (env_steps/s)")
    plt.title("Exp 05: Multi-Level Parallelism Comparison (64 Cores)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.0f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    output_path = f"{RESULTS_DIR}/plot_exp05_multilevel.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Generated: {output_path}")

def print_summary_table(t1, tp1, t2, s2, mpi_csv, h64, exp05_data):
    os.makedirs("results", exist_ok=True)
    output_file = "results/summary_table.txt"

    print(f"Generating summary report to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        
        def log(text):
            print(text)          
            f.write(text + "\n")

        log("\n" + "="*60)
        log(f"{'🏁 FINAL UNIFIED PERFORMANCE REPORT 🏁':^60}")
        log("="*60)
        log(f"{'Experiment':<30} | {'Metric':<15} | {'Best Result':<10}")
        log("-" * 60)

        # 1. Level 1 Summary
        if tp1:
            best_tp = max(tp1)
            log(f"{'Level 1 (OpenMP Agent)':<30} | {'Max Throughput':<15} | {best_tp:.0f} steps/s")
        else:
            log(f"{'Level 1 (OpenMP Agent)':<30} | {'Status':<15} | N/A")

        # 2. Level 2 Summary
        if s2:
            max_speedup = max(s2)
            log(f"{'Level 2 (OpenMP Env)':<30} | {'Max Speedup':<15} | {max_speedup:.2f}x")
        else:
            log(f"{'Level 2 (OpenMP Env)':<30} | {'Status':<15} | N/A")

        # 3. MPI Summary
        if os.path.exists(mpi_csv):
            try:
                df = pd.read_csv(mpi_csv, skiprows=1)
                df.columns = [c.strip() for c in df.columns]
                max_mpi_speedup = df['Speedup'].max()
                log(f"{'Level 3 (MPI Distributed)':<30} | {'Max Speedup':<15} | {max_mpi_speedup:.2f}x")
            except:
                 log(f"{'Level 3 (MPI Distributed)':<30} | {'Status':<15} | Parse Error")
        else:
            log(f"{'Level 3 (MPI Distributed)':<30} | {'Status':<15} | N/A")

        # 4. Hybrid Summary (Scenario A)
        if h64 and "High Batch (A)" in h64:
            strategies = h64["High Batch (A)"]
            best_strat = max(strategies, key=strategies.get)
            best_val = strategies[best_strat]
            log(f"{'Hybrid (Heavy Load, 64c)':<30} | {'Best Strategy':<15} | {best_strat}")
            log(f"{' ':30} | {'Throughput':<15} | {best_val:.0f} env_steps/s")

        # 5. Exp 05 Multilevel Summary
        if exp05_data:
            best_strat_05 = max(exp05_data, key=exp05_data.get)
            best_val_05 = exp05_data[best_strat_05]
            log(f"{'Exp 05 (Multilevel Arch)':<30} | {'Best Strategy':<15} | {best_strat_05}")
            log(f"{' ':30} | {'Throughput':<15} | {best_val_05:.0f} env_steps/s")
        else:
            log(f"{'Exp 05 (Multilevel Arch)':<30} | {'Status':<15} | N/A")

def main():
    print("--- Parsing Logs & Generating Plots ---")
    
    exp1_data = parse_level1_log(f"{RESULTS_DIR}/exp_01_log.txt")
    t2, s2 = parse_level2_log(f"{RESULTS_DIR}/exp_02_log.txt") 
    h64 = parse_hybrid_log(f"{RESULTS_DIR}/exp_04_log.txt")     
    mpi_csv = f"{RESULTS_DIR}/mpi_scaling_results.csv"
    exp05_data = parse_exp05_log(f"{RESULTS_DIR}/exp_05_log.txt")
    
    plot_level1(exp1_data)
    plot_level2(t2, s2)
    plot_mpi(mpi_csv)
    plot_exp05(exp05_data)
    if h64: plot_hybrid_bar(h64, "plot_hybrid_64c.png", "64-Core Hybrid Comparison")
    
    l1_threads, l1_tps = ([], [])
    if exp1_data:
        _, l1_threads, l1_tps = exp1_data[-1] 

    print_summary_table(l1_threads, l1_tps, t2, s2, mpi_csv, h64, exp05_data)

    print("="*60 + "\n")

if __name__ == "__main__":
    main()