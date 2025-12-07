# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # List of CSV files for different kernels
# csv_files = [
#     "reduction_benchmark_results_kernel_1.csv",
#     "reduction_benchmark_results_kernel_2.csv",
#     "reduction_benchmark_results_kernel_3.csv",
#     "reduction_benchmark_results_kernel_4.csv",
#     "reduction_benchmark_results_kernel_5.csv"
# ]

# # Kernel names for legend
# kernel_names = [
#     "Kernel 1: Interleaved",
#     "Kernel 2: Sequential",
#     "Kernel 3: Sequential + No Divergence",
#     "Kernel 4: First Add During Load",
#     "Kernel 5: Unroll Last Warp"
# ]

# # Colors for different kernels
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# # Load data from CSV files
# data_frames = []
# available_kernels = []
# available_names = []
# available_colors = []

# for i, file in enumerate(csv_files):
#     if os.path.exists(file):
#         df = pd.read_csv(file)
#         data_frames.append(df)
#         available_kernels.append(i + 1)
#         available_names.append(kernel_names[i])
#         available_colors.append(colors[i])
#         print(f"✓ Loaded: {file}")
#     else:
#         print(f"✗ Not found: {file}")

# if not data_frames:
#     print("\nError: No CSV files found!")
#     exit(1)

# # Create figure with subplots
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('Reduction Kernel Performance Comparison', fontsize=16, fontweight='bold')

# # Plot 1: Throughput vs Array Size
# ax1 = axes[0, 0]
# for i, df in enumerate(data_frames):
#     ax1.plot(df['Array Size'], df['Throughput (GB/s)'], 
#              marker='o', linewidth=2, markersize=6,
#              label=available_names[i], color=available_colors[i])

# ax1.set_xlabel('Array Size', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Throughput (GB/s)', fontsize=12, fontweight='bold')
# ax1.set_title('Throughput vs Array Size', fontsize=13, fontweight='bold')
# ax1.set_xscale('log')
# ax1.grid(True, alpha=0.3, linestyle='--')
# ax1.legend(loc='best', fontsize=9)

# # Add peak bandwidth line if available (you can set this manually)
# peak_bandwidth = None
# if len(data_frames) > 0 and 'Efficiency (%)' in data_frames[0].columns:
#     # Calculate peak bandwidth from efficiency and throughput
#     max_throughput = data_frames[0]['Throughput (GB/s)'].max()
#     max_efficiency = data_frames[0]['Efficiency (%)'].max()
#     if max_efficiency > 0:
#         peak_bandwidth = (max_throughput / max_efficiency) * 100
#         ax1.axhline(y=peak_bandwidth, color='red', linestyle='--', 
#                    linewidth=2, label=f'Peak BW: {peak_bandwidth:.1f} GB/s', alpha=0.7)
#         ax1.legend(loc='best', fontsize=9)

# # Plot 2: Execution Time vs Array Size
# ax2 = axes[0, 1]
# for i, df in enumerate(data_frames):
#     ax2.plot(df['Array Size'], df['Time (ms)'], 
#              marker='s', linewidth=2, markersize=6,
#              label=available_names[i], color=available_colors[i])

# ax2.set_xlabel('Array Size', fontsize=12, fontweight='bold')
# ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
# ax2.set_title('Execution Time vs Array Size', fontsize=13, fontweight='bold')
# ax2. set_xscale('log')
# ax2.set_yscale('log')
# ax2.grid(True, alpha=0.3, linestyle='--')
# ax2.legend(loc='best', fontsize=9)

# # Plot 3: Efficiency vs Array Size (if available)
# ax3 = axes[1, 0]
# has_efficiency = False
# for i, df in enumerate(data_frames):
#     if 'Efficiency (%)' in df.columns:
#         has_efficiency = True
#         ax3.plot(df['Array Size'], df['Efficiency (%)'], 
#                  marker='^', linewidth=2, markersize=6,
#                  label=available_names[i], color=available_colors[i])

# if has_efficiency:
#     ax3.set_xlabel('Array Size', fontsize=12, fontweight='bold')
#     ax3.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
#     ax3.set_title('Memory Bandwidth Efficiency vs Array Size', fontsize=13, fontweight='bold')
#     ax3.set_xscale('log')
#     ax3.grid(True, alpha=0.3, linestyle='--')
#     ax3.legend(loc='best', fontsize=9)
#     ax3.set_ylim(0, 105)
#     ax3.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
# else:
#     ax3.text(0.5, 0.5, 'Efficiency data not available', 
#              ha='center', va='center', transform=ax3.transAxes, fontsize=12)
#     ax3.axis('off')

# # Plot 4: Speedup comparison (relative to kernel 1)
# ax4 = axes[1, 1]
# if len(data_frames) > 1:
#     baseline_df = data_frames[0]
#     for i in range(1, len(data_frames)):
#         df = data_frames[i]
#         # Calculate speedup (lower time = higher speedup)
#         speedup = baseline_df['Time (ms)'] / df['Time (ms)']
#         ax4.plot(df['Array Size'], speedup, 
#                  marker='D', linewidth=2, markersize=6,
#                  label=f"{available_names[i]} vs {available_names[0]}", 
#                  color=available_colors[i])
    
#     ax4.set_xlabel('Array Size', fontsize=12, fontweight='bold')
#     ax4.set_ylabel('Speedup', fontsize=12, fontweight='bold')
#     ax4.set_title(f'Speedup Relative to {available_names[0]}', fontsize=13, fontweight='bold')
#     ax4.set_xscale('log')
#     ax4.grid(True, alpha=0.3, linestyle='--')
#     ax4.legend(loc='best', fontsize=9)
#     ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
# else:
#     ax4.text(0.5, 0.5, 'Need multiple kernels for speedup comparison', 
#              ha='center', va='center', transform=ax4.transAxes, fontsize=12)
#     ax4.axis('off')

# plt. tight_layout(rect=[0, 0.03, 1, 0.97])
# plt.savefig('reduction_kernel_performance.png', dpi=300, bbox_inches='tight')
# print("\n✓ Graph saved as 'reduction_kernel_performance.png'")
# plt.show()

# # Print summary statistics
# print("\n" + "="*80)
# print("PERFORMANCE SUMMARY")
# print("="*80)

# for i, df in enumerate(data_frames):
#     print(f"\n{available_names[i]}:")
#     print(f"  Max Throughput: {df['Throughput (GB/s)']. max():.2f} GB/s @ size {df. loc[df['Throughput (GB/s)']. idxmax(), 'Array Size']}")
#     print(f"  Min Time: {df['Time (ms)'].min():.4f} ms @ size {df. loc[df['Time (ms)'].idxmin(), 'Array Size']}")
#     if 'Efficiency (%)' in df.columns:
#         print(f"  Max Efficiency: {df['Efficiency (%)'].max():.2f}% @ size {df.loc[df['Efficiency (%)'].idxmax(), 'Array Size']}")
    
#     # Check for errors
#     if 'Error' in df.columns:
#         max_error = df['Error'].max()
#         if max_error > 1e-3:
#             print(f"  ⚠ WARNING: Max Error = {max_error:.2e}")
#         else:
#             print(f"  ✓ All results verified (max error: {max_error:.2e})")

# print("\n" + "="*80)

# # Create a second figure for detailed comparison at specific sizes
# fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
# fig2.suptitle('Kernel Comparison at Different Array Sizes', fontsize=14, fontweight='bold')

# # Select a few representative sizes for bar chart comparison
# if len(data_frames) > 0:
#     all_sizes = data_frames[0]['Array Size'].values
#     # Select ~6 representative sizes (powers of 2)
#     selected_indices = np.linspace(0, len(all_sizes)-1, min(6, len(all_sizes)), dtype=int)
#     selected_sizes = all_sizes[selected_indices]
    
#     # Bar chart: Throughput comparison
#     ax_bar1 = axes2[0]
#     bar_width = 0.15
#     x_pos = np.arange(len(selected_sizes))
    
#     for i, df in enumerate(data_frames):
#         throughputs = [df[df['Array Size'] == size]['Throughput (GB/s)'].values[0] 
#                       for size in selected_sizes]
#         ax_bar1.bar(x_pos + i * bar_width, throughputs, bar_width, 
#                    label=available_names[i], color=available_colors[i], alpha=0.8)
    
#     ax_bar1.set_xlabel('Array Size', fontsize=11, fontweight='bold')
#     ax_bar1.set_ylabel('Throughput (GB/s)', fontsize=11, fontweight='bold')
#     ax_bar1.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
#     ax_bar1.set_xticks(x_pos + bar_width * (len(data_frames) - 1) / 2)
#     ax_bar1.set_xticklabels([f'{int(size)}' if size < 1024 else f'{int(size//1024)}K' 
#                              if size < 1024*1024 else f'{int(size//(1024*1024))}M' 
#                              for size in selected_sizes], rotation=45)
#     ax_bar1.legend(loc='best', fontsize=8)
#     ax_bar1. grid(True, alpha=0.3, axis='y', linestyle='--')
    
#     # Bar chart: Time comparison
#     ax_bar2 = axes2[1]
#     for i, df in enumerate(data_frames):
#         times = [df[df['Array Size'] == size]['Time (ms)'].values[0] 
#                 for size in selected_sizes]
#         ax_bar2.bar(x_pos + i * bar_width, times, bar_width, 
#                    label=available_names[i], color=available_colors[i], alpha=0.8)
    
#     ax_bar2. set_xlabel('Array Size', fontsize=11, fontweight='bold')
#     ax_bar2. set_ylabel('Execution Time (ms)', fontsize=11, fontweight='bold')
#     ax_bar2.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
#     ax_bar2.set_xticks(x_pos + bar_width * (len(data_frames) - 1) / 2)
#     ax_bar2.set_xticklabels([f'{int(size)}' if size < 1024 else f'{int(size//1024)}K' 
#                              if size < 1024*1024 else f'{int(size//(1024*1024))}M' 
#                              for size in selected_sizes], rotation=45)
#     ax_bar2.legend(loc='best', fontsize=8)
#     ax_bar2.grid(True, alpha=0.3, axis='y', linestyle='--')

# plt.tight_layout(rect=[0, 0.03, 1, 0.96])
# plt.savefig('reduction_kernel_comparison_bars.png', dpi=300, bbox_inches='tight')
# print("✓ Bar chart saved as 'reduction_kernel_comparison_bars.png'")
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # List of CSV files for different kernels
# csv_files = [
#     "reduction_benchmark_results_kernel_1.csv",
#     "reduction_benchmark_results_kernel_2.csv",
#     "reduction_benchmark_results_kernel_3.csv",
#     "reduction_benchmark_results_kernel_4.csv",
#     "reduction_benchmark_results_kernel_5.csv"
# ]

# # Kernel names for legend
# kernel_names = [
#     "Kernel 1: Interleaved",
#     "Kernel 2: Sequential",
#     "Kernel 3: Sequential + No Divergence",
#     "Kernel 4: First Add During Load",
#     "Kernel 5: Unroll Last Warp"
# ]

# # Colors for different kernels
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# # Load data
# data_frames = []
# available_names = []
# available_colors = []

# for i, file in enumerate(csv_files):
#     if os.path.exists(file):
#         df = pd.read_csv(file)
#         data_frames.append(df)
#         available_names.append(kernel_names[i])
#         available_colors.append(colors[i])
#         print(f"✓ Loaded: {file}")
#     else:
#         print(f"✗ Not found: {file}")

# if not data_frames:
#     print("Error: No CSV files found!")
#     exit(1)

# # Create figure with 2 subplots
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# fig.suptitle('Reduction Kernel Performance', fontsize=16, fontweight='bold')

# # Throughput vs Array Size
# ax1 = axes[0]
# for i, df in enumerate(data_frames):
#     ax1.plot(df['Array Size'], df['Throughput (GB/s)'], 
#              marker='o', linewidth=2, markersize=6,
#              label=available_names[i], color=available_colors[i])

# ax1.set_xlabel('Array Size', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Throughput (GB/s)', fontsize=12, fontweight='bold')
# ax1.set_title('Throughput vs Array Size', fontsize=13, fontweight='bold')
# ax1.set_xscale('log')
# ax1.grid(True, alpha=0.3, linestyle='--')
# ax1.legend(loc='best', fontsize=9)

# # Execution Time vs Array Size
# ax2 = axes[1]
# for i, df in enumerate(data_frames):
#     ax2.plot(df['Array Size'], df['Time (ms)'], 
#              marker='s', linewidth=2, markersize=6,
#              label=available_names[i], color=available_colors[i])

# ax2.set_xlabel('Array Size', fontsize=12, fontweight='bold')
# ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
# ax2.set_title('Execution Time vs Array Size', fontsize=13, fontweight='bold')
# ax2.set_xscale('log')
# ax2.set_yscale('log')
# ax2.grid(True, alpha=0.3, linestyle='--')
# ax2.legend(loc='best', fontsize=9)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('reduction_kernel_simple_plots.png', dpi=300, bbox_inches='tight')
# print("✓ Graph saved as 'reduction_kernel_simple_plots.png'")
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import os

# List of CSV files for different kernels
csv_files = [
    "reduction_benchmark_results_kernel_1.csv",
    "reduction_benchmark_results_kernel_2.csv",
    "reduction_benchmark_results_kernel_3.csv",
    "reduction_benchmark_results_kernel_4.csv",
    "reduction_benchmark_results_kernel_5.csv",
    "reduction_benchmark_results_kernel_6.csv"
]

# Kernel names for legend
kernel_names = [
    "Kernel 1: Interleaved with divergent branching",
    "Kernel 2: Interleaved with bank conflicts",
    "Kernel 3: Sequential",
    "Kernel 4: First Add During Load",
    "Kernel 5: Unroll Last Warp",
    "Kernel 6: Sequential + warp unrolling"
]

# Colors for different kernels
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
# Load data
data_frames = []
available_names = []
available_colors = []

for i, file in enumerate(csv_files):
    if os.path.exists(file):
        df = pd.read_csv(file)
        data_frames.append(df)
        available_names.append(kernel_names[i])
        available_colors.append(colors[i])
        print(f"✓ Loaded: {file}")
    else:
        print(f"✗ Not found: {file}")

if not data_frames:
    print("Error: No CSV files found!")
    exit(1)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 1, figsize=(24, 12))
fig.suptitle('Reduction Kernel Performance', fontsize=16, fontweight='bold')

# Throughput vs Array Size
ax1 = axes
for i, df in enumerate(data_frames):
    ax1.plot(df['Array Size'], df['Throughput (GB/s)'], 
             marker='o', linewidth=2, markersize=6,
             label=available_names[i], color=available_colors[i])

ax1.set_xlabel('Array Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Throughput (GB/s)', fontsize=12, fontweight='bold')
ax1.set_title('Throughput vs Array Size', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='best', fontsize=9)

# Speedup vs Array Size (relative to Kernel 1)
# ax2 = axes[1]
# baseline_df = data_frames[0]  # Kernel 1 as baseline

# for i in range(1, len(data_frames)):
#     df = data_frames[i]
#     # Compute speedup
#     speedup = baseline_df['Time (ms)'] / df['Time (ms)']
#     ax2.plot(df['Array Size'], speedup, 
#              marker='s', linewidth=2, markersize=6,
#              label=f"{available_names[i]} vs {available_names[0]}", 
#              color=available_colors[i])

# ax2.set_xlabel('Array Size', fontsize=12, fontweight='bold')
# ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
# ax2.set_title(f'Speedup vs {available_names[0]}', fontsize=13, fontweight='bold')
# ax2.set_xscale('log')
# ax2.grid(True, alpha=0.3, linestyle='--')
# ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
# ax2.legend(loc='best', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('reduction_kernel_throughput_speedup.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved as 'reduction_kernel_throughput_speedup.png'")
plt.show()
