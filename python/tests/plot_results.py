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
    """解析 Level 1 Log: 抓取 Thread 數與 Throughput"""
    threads = []
    throughputs = []
    
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping Plot 1.")
        return None, None

    with open(filepath, 'r') as f:
        content = f.read()
        # Regex 尋找 "Running with X Threads" 和隨後的 "Throughput: Y"
        matches = re.findall(r"Running with (\d+) Thread.*?Throughput: ([\d.]+) Steps/Sec", content, re.DOTALL)
        for t, tp in matches:
            threads.append(int(t))
            throughputs.append(float(tp))
            
    return threads, throughputs

def parse_level2_log(filepath):
    """解析 Level 2 Log Table"""
    threads = []
    speedups = []
    
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping Plot 2.")
        return None, None
        
    with open(filepath, 'r') as f:
        for line in f:
            # 尋找類似 " 1 | 1234.56 | 1.00x" 的行
            parts = line.split('|')
            if len(parts) == 3:
                try:
                    t = int(parts[0].strip())
                    # 取 Speedup (e.g., "1.00x" -> 1.00)
                    s_str = parts[2].strip().replace('x', '').split(' ')[0]
                    if s_str == "N/A": continue
                    s = float(s_str)
                    threads.append(t)
                    speedups.append(s)
                except ValueError:
                    continue
    return threads, speedups

def parse_hybrid_log(filepath):
    """解析 Hybrid 實驗 (16c 或 64c)"""
    if not os.path.exists(filepath):
        print(f"⚠️ {filepath} not found, skipping Hybrid Plot.")
        return None

    data = {} # {Scenario: {Strategy: Throughput}}
    current_scenario = "Unknown"
    
    with open(filepath, 'r') as f:
        for line in f:
            if "Scenario" in line:
                # 抓取 Scenario A 或 B
                if "Scenario A" in line: current_scenario = "High Batch (A)"
                if "Scenario B" in line: current_scenario = "Low Batch (B)"
            
            if "Strategy" in line and "Throughput" in line:
                # 解析策略名稱
                strat_match = re.search(r"Strategy \[(.*?)\]", line)
                tp_match = re.search(r"Throughput: ([\d.]+) env_steps/s", line)
                
                if strat_match and tp_match:
                    strat = strat_match.group(1).replace(" ", "")
                    tp = float(tp_match.group(1))
                    
                    if current_scenario not in data: data[current_scenario] = {}
                    data[current_scenario][strat] = tp
    return data

def plot_level1(threads, throughputs):
    if not threads: return
    plt.figure(figsize=(8, 5))
    plt.plot(threads, throughputs, 'o-', linewidth=2, label="Measured")
    plt.xlabel("Number of Threads (OpenMP)")
    plt.ylabel("Throughput (Steps/Sec)")
    plt.title("Level 1 Scaling: Agent Parallelism")
    plt.grid(True)
    plt.xticks(threads)
    plt.savefig(f"{RESULTS_DIR}/plot_level1_scaling.png")
    plt.close()
    print("Generated: plot_level1_scaling.png")

def plot_level2(threads, speedups):
    if not threads: return
    plt.figure(figsize=(8, 5))
    plt.plot(threads, speedups, 's-', color='orange', linewidth=2, label="Measured")
    # 畫理想線
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
        # [修正] 加入 skiprows=1 跳過檔案第一行的 "MPI Scaling Test Results" 標題
        # [修正] 加入 strip() 去除欄位名稱前後可能多餘的空白
        df = pd.read_csv(csv_path, skiprows=1)
        
        # 確保欄位名稱乾淨 (去除可能的空白)
        df.columns = [c.strip() for c in df.columns]

        plt.figure(figsize=(8, 5))
        plt.plot(df['Ranks'], df['Speedup'], 'D-', color='green', linewidth=2, label="MPI Measured")
        # 畫理想線 (x=y)
        plt.plot(df['Ranks'], df['Ranks'], 'k--', label="Ideal Linear")
        
        plt.xlabel("Number of MPI Ranks")
        plt.ylabel("Speedup Factor")
        plt.title("Level 3 Scaling: Distributed MPI")
        plt.legend()
        plt.grid(True)
        plt.xticks(df['Ranks']) # 確保 X 軸刻度顯示所有 Ranks
        plt.savefig(f"{RESULTS_DIR}/plot_level3_mpi.png")
        plt.close()
        print("Generated: plot_level3_mpi.png")
        
    except Exception as e:
        print(f"Error plotting MPI: {e}")
        # 印出讀到的欄位名稱幫助除錯
        if 'df' in locals():
            print(f"Read columns: {df.columns.tolist()}")

def plot_hybrid_bar(data, filename, title_suffix):
    if not data: return
    
    scenarios = list(data.keys())
    strategies = list(next(iter(data.values())).keys())
    
    # 準備繪圖資料
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

def main():
    print("--- Parsing Logs & Generating Plots ---")
    
    # Plot 1
    t1, tp1 = parse_level1_log(f"{RESULTS_DIR}/level1_log.txt")
    plot_level1(t1, tp1)
    
    # Plot 2
    t2, s2 = parse_level2_log(f"{RESULTS_DIR}/level2_log.txt")
    plot_level2(t2, s2)
    
    # Plot 3
    plot_mpi(f"{RESULTS_DIR}/mpi_scaling.csv")
    
    # Plot 4 (16 Cores)
    h16_data = parse_hybrid_log(f"{RESULTS_DIR}/hybrid_16c_log.txt")
    plot_hybrid_bar(h16_data, "plot_hybrid_16c.png", "16 Cores")
    
    # Plot 5 (64 Cores)
    h64_data = parse_hybrid_log(f"{RESULTS_DIR}/hybrid_64c_log.txt")
    plot_hybrid_bar(h64_data, "plot_hybrid_64c.png", "64 Cores - Heavy Load")

if __name__ == "__main__":
    main()