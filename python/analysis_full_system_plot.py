import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# ==========================================
# 0. Global Style Settings (統一字體與風格)
# ==========================================
# 設定全域字體，優先使用 Arial，若無則退回 sans-serif
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 16

# Parse data
columns = ["Group", "Config", "Nodes", "Ranks", "L2_Threads", "L1_Threads", "Total_Cores", "Throughput"]
try:
    df = pd.read_csv("results/full_grid_search_results.csv", header=None, names=columns)
except FileNotFoundError:
    # Fallback for demonstration if file doesn't exist in current path
    print("Warning: CSV file not found. Please ensure 'results/full_grid_search_results.csv' exists.")
    # Create dummy df structure to avoid crash in demonstration
    df = pd.DataFrame(columns=columns)

# Handle CRASHed
df['Throughput_Val'] = pd.to_numeric(df['Throughput'], errors='coerce').fillna(0)
df['Is_Crash'] = df['Throughput'] == 'CRASHed'

# ==========================================
# Figure 5.1: Single Node Architecture Comparison (G1)
# (使用原本 Plot 1 的樣式與配色)
# ==========================================
g1_data = df[df['Group'] == 'G1_SingleNode'].copy()
# Simplify names for plotting
g1_data['ShortName'] = ['Pure Thread (L2)', 'Nested (16x4)', 'Hybrid (4x16)', 'Pure Process (MPI)']
# 使用 Plot 1 的鮮豔配色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

plt.figure(figsize=(10, 6))
# Fixed: use 'Throughput_Val'
bars = plt.bar(g1_data['ShortName'], g1_data['Throughput_Val'], color=colors)

plt.title('Figure 4.1: Single-Node Architecture Comparison (64 Cores)', fontsize=14, fontweight='bold')
plt.ylabel('Throughput (Steps/s)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
# 儲存為您報告需要的檔名
plt.savefig('results/fig_5_1_single_node_comparison.png')
print("Saved Figure 5.1")
plt.close()


# ==========================================
# Figure 5.2: Distributed Strong Scaling (G4)
# ==========================================
# Filter G4 and sort by cores
g4_data = df[df['Group'].isin(['G4_PureMPI_Opt'])].copy()

# We need the single node Pure MPI baseline from G1 to calculate speedup
# 注意：這裡要抓 G1 中 ShortName 為 'Pure Process (MPI)' 的那筆
baseline_row = g1_data[g1_data['ShortName'] == 'Pure Process (MPI)']
if not baseline_row.empty:
    baseline_throughput = baseline_row['Throughput_Val'].values[0]
else:
    baseline_throughput = 1.0 # Avoid division by zero if data missing

# Add G4 data points
g4_data['Speedup'] = g4_data['Throughput_Val'] / baseline_throughput
g4_data = g4_data.sort_values('Total_Cores')

# Prepare plotting data: Include Single Node (64 cores) as starting point
cores = [64] + g4_data['Total_Cores'].tolist()
speedups = [1.0] + g4_data['Speedup'].tolist()

plt.figure(figsize=(8, 5))
plt.plot(cores, speedups, marker='o', linewidth=2, color='#2ecc71', label='Measured Speedup')

# Add Ideal Linear Scaling line
ideal_speedups = [c / 64.0 for c in cores]
plt.plot(cores, ideal_speedups, linestyle='--', color='gray', label='Ideal Linear')

plt.title('Figure 4.2: Distributed Strong Scaling (Pure MPI)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Cores', fontsize=12)
plt.ylabel('Speedup Factor (Relative to Single Node)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xticks(cores) 

for i, txt in enumerate(speedups):
    if i > 0: # Skip first point
        plt.annotate(f'{txt:.1f}x', (cores[i], speedups[i]), 
                     xytext=(0, 10), textcoords='offset points', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/fig_5_2_distributed_scaling.png')
print("Saved Figure 5.2")
plt.close()


# ==========================================
# Figure 5.3: Hybrid Trade-offs (G3 + Comparison)
# ==========================================
g2_data = df[df['Group'] == 'G2_Cluster_Std']
g3_data = df[df['Group'] == 'G3_Hybrid_Opt']

# Combine relevant rows
hybrid_comparison = pd.concat([g2_data, g3_data])
# Filter for valid numeric throughput using 'Throughput_Val'
hybrid_comparison = hybrid_comparison[hybrid_comparison['Throughput_Val'] > 0]
hybrid_comparison = hybrid_comparison.sort_values('Throughput_Val')

# create readable labels
def get_label(row):
    return f"{int(row['Ranks'])}R x {int(row['L2_Threads'])}T"

hybrid_comparison['Label'] = hybrid_comparison.apply(get_label, axis=1)

plt.figure(figsize=(10, 6))
# 統一配色風格，使用紫色系代表 Hybrid
bars = plt.bar(hybrid_comparison['Label'], hybrid_comparison['Throughput_Val'], color='#9b59b6')

plt.title('Figure 4.3: Hybrid Architecture Trade-offs (Approx. 256 Cores)', fontsize=14, fontweight='bold')
plt.xlabel('Configuration (Ranks x Threads)', fontsize=12)
plt.ylabel('Throughput (Steps/s)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the trend
plt.text(0.5, 0.9, 'Process-Dominant -> Higher Performance', 
         horizontalalignment='center', verticalalignment='center', 
         transform=plt.gca().transAxes, fontsize=12, style='italic', 
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='#dddddd'))

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/fig_5_3_hybrid_tradeoff.png')
print("Saved Figure 5.3")
plt.close()

# ==========================================
# Optional: Plot for "Optimization Evolution" (Can be used as summary)
# ==========================================
# Combine G4 and G5 valid runs for a scaling summary plot if needed
g4_g5 = df[(df['Group'].isin(['G4_PureMPI_Opt', 'G5_Edge_Case']))].copy()
scaling_summary = g4_g5.groupby('Ranks')['Throughput_Val'].max().reset_index()
# Assuming specific order based on ranks
scaling_summary = scaling_summary.sort_values('Ranks')
scaling_summary['ConfigLabel'] = [f'{r} Ranks' for r in scaling_summary['Ranks']]
# Basic check for crashes
# Note: In the original script, G5 had crashes. 
# If throughput is 0, it's a crash.
scaling_summary['Status'] = ['CRASH' if x == 0 else 'Success' for x in scaling_summary['Throughput_Val']]

if not scaling_summary.empty:
    plt.figure(figsize=(10, 6))
    colors_scaling = ['#1f77b4' if s == 'Success' else 'gray' for s in scaling_summary['Status']]
    bars_scale = plt.bar(scaling_summary['ConfigLabel'], scaling_summary['Throughput_Val'], color=colors_scaling)

    plt.title('Pure MPI Scaling & Stability Limits', fontsize=14, fontweight='bold')
    plt.ylabel('Throughput (Steps/s)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars_scale):
        height = bar.get_height()
        status = scaling_summary.iloc[i]['Status']
        if status == 'CRASH':
            plt.text(bar.get_x() + bar.get_width()/2., 10000,
                     'CRASH',
                     ha='center', va='bottom', color='white', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{int(height):,}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/mpi_scaling_crash.png')
    print("Saved scaling crash plot")
    plt.close()

print("All plots generated successfully with unified fonts!")