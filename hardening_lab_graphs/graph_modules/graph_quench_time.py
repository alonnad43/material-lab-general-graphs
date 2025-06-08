"""
Quench Time Analysis and Advanced Plotting Module

This module provides functions for advanced analysis including quench time effects,
hardness profiling, and comparative analysis for the hardening lab system.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize_column

# Set matplotlib to support default font (remove explicit font.family)

def plot_hardness_profile_analysis(models_data, output_dir):
    """
    Plot comprehensive hardness profile analysis across all models.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Hardness vs Distance for all models (detailed)
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    for i, (model_num, df) in enumerate(models_data.items()):
        if df.empty:
            continue
            
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
        
        if hrc_col and dist_col:
            x = df[dist_col].astype(float)
            y = df[hrc_col].astype(float)
            ax1.plot(x, y, marker='o', linestyle='-', color=colors[i], 
                    label=f"Model {model_num}", linewidth=2, markersize=4)
    
    ax1.set_xlabel('Distance [mm]')
    ax1.set_ylabel('HRC')
    ax1.set_title('Hardness Profile Analysis')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Average hardness per model
    ax2 = axes[0, 1]
    model_nums = []
    avg_hardness = []
    std_hardness = []
    
    for model_num, df in models_data.items():
        if df.empty:
            continue
            
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        
        if hrc_col:
            hardness_values = df[hrc_col].astype(float)
            model_nums.append(model_num)
            avg_hardness.append(hardness_values.mean())
            std_hardness.append(hardness_values.std())
    
    bars = ax2.bar([f"Model {m}" for m in model_nums], avg_hardness, 
                   yerr=std_hardness, capsize=5, alpha=0.7, color=colors[:len(model_nums)])
    ax2.set_ylabel('Average HRC')
    ax2.set_title('Average Hardness by Model')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, avg_hardness, std_hardness)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{val:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Hardness range analysis
    ax3 = axes[1, 0]
    hardness_ranges = []
    min_hardness = []
    max_hardness = []
    
    for model_num in model_nums:
        df = models_data[model_num]
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        
        if hrc_col:
            hardness_values = df[hrc_col].astype(float)
            min_val = hardness_values.min()
            max_val = hardness_values.max()
            min_hardness.append(min_val)
            max_hardness.append(max_val)
            hardness_ranges.append(max_val - min_val)
    
    x_pos = range(len(model_nums))
    ax3.bar(x_pos, hardness_ranges, alpha=0.7, color='lightcoral')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"Model {m}" for m in model_nums], rotation=45)
    ax3.set_ylabel('Hardness Range [HRC]')
    ax3.set_title('Hardness Range Analysis')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (pos, range_val) in enumerate(zip(x_pos, hardness_ranges)):
        ax3.text(pos, range_val + 0.5, f'{range_val:.1f}', ha='center', va='bottom')
    
    # Plot 4: Distance distribution analysis
    ax4 = axes[1, 1]
    all_distances = []
    distance_labels = []
    
    for model_num, df in models_data.items():
        if df.empty:
            continue
            
        col_map = {normalize_column(c): c for c in df.columns}
        dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
        
        if dist_col:
            distances = df[dist_col].astype(float)
            all_distances.extend(distances)
            distance_labels.extend([f"Model {model_num}"] * len(distances))
    
    # Create box plot for distance distribution
    distance_groups = {}
    for label, dist in zip(distance_labels, all_distances):
        if label not in distance_groups:
            distance_groups[label] = []
        distance_groups[label].append(dist)
    
    if distance_groups:
        box_data = [distance_groups[label] for label in distance_groups.keys()]
        bp = ax4.boxplot(box_data, labels=list(distance_groups.keys()), patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Distance [mm]')
        ax4.set_title('Distance Distribution Analysis')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45)
    
    plt.suptitle('Comprehensive Hardness Profile Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graphs', 'quench_time', 'comprehensive_hardness_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")

def plot_cooling_rate_estimation(models_data, output_dir):
    """
    Plot estimated cooling rate analysis based on distance and hardness data.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models_data)))
    
    for i, (model_num, df) in enumerate(models_data.items()):
        if df.empty:
            continue
            
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
        
        if hrc_col and dist_col:
            x = df[dist_col].astype(float)
            y = df[hrc_col].astype(float)
            
            # Estimate cooling rate (simplified model)
            # Assumes cooling rate is inversely proportional to distance
            cooling_rate = 100 / (x + 1)  # Arbitrary units
            
            plt.scatter(cooling_rate, y, alpha=0.7, s=60, color=colors[i], 
                       label=f"Model {model_num}")
    
    plt.xlabel('Estimated Cooling Rate [Arbitrary Units]')
    plt.ylabel('HRC')
    plt.title('Hardness by Estimated Cooling Rate', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graphs', 'quench_time', 'cooling_rate_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")

def plot_statistical_summary(models_data, output_dir):
    """
    Generate statistical summary plots for all models.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    # Prepare statistical data
    stats_data = []
    
    for model_num, df in models_data.items():
        if df.empty:
            continue
            
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
        
        if hrc_col and dist_col:
            hardness = df[hrc_col].astype(float)
            distance = df[dist_col].astype(float)
            
            stats_data.append({
                'model': f"Model {model_num}",
                'model_num': model_num,
                'mean_hardness': hardness.mean(),
                'std_hardness': hardness.std(),
                'min_hardness': hardness.min(),
                'max_hardness': hardness.max(),
                'mean_distance': distance.mean(),
                'max_distance': distance.max(),
                'num_points': len(hardness)
            })
    
    if not stats_data:
        return
    
    # Create summary table plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Summary statistics table
    ax1.axis('tight')
    ax1.axis('off')
    
    table_data = []
    headers = ['Model', 'Mean HRC', 'Std Dev', 'Min', 'Max', 'Points']
    
    for stats in stats_data:
        row = [
            stats['model'],
            f"{stats['mean_hardness']:.1f}",
            f"{stats['std_hardness']:.1f}",
            f"{stats['min_hardness']:.1f}",
            f"{stats['max_hardness']:.1f}",
            f"{stats['num_points']}"
        ]
        table_data.append(row)
    
    table = ax1.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax1.set_title('Statistical Summary by Model', fontsize=14, fontweight='bold', pad=20)
    
    # Plot 2: Performance comparison radar chart
    ax2 = plt.subplot(122, projection='polar')
    
    # Normalize metrics for radar chart
    categories = ['Mean Hardness', 'Stability', 'Measurement Range', 'Data Points']
    
    for stats in stats_data[:3]:  # Show first 3 models to avoid clutter
        values = [
            stats['mean_hardness'] / 60,  # Normalize to 0-1
            (1 - stats['std_hardness'] / 10),  # Lower std is better
            stats['max_distance'] / 25,  # Normalize distance range
            min(stats['num_points'] / 10, 1)  # Normalize number of points
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values += values[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax2.plot(angles, values, marker='o', linewidth=2, label=stats['model'])
        ax2.fill(angles, values, alpha=0.25)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 1)
    ax2.set_title('Performance Comparison', fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'graphs', 'quench_time', 'statistical_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")
