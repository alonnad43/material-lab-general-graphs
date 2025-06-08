"""
Individual Model Plotting Module for Jominy Test Analysis

This module provides functions to create individual graphs for each model in the
hardening lab analysis, showing detailed Jominy test results with English labels
for BGU Materials Lab reports.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize_column, get_model_description

def plot_individual_jominy_curve(df, model_num, output_dir):
    """
    Plot and save the main Jominy curve for a specific model in its own folder.

    Parameters:
        df (pd.DataFrame): DataFrame containing distance and HRC data for the model
        model_num (int): Model number
        output_dir (str): Base output directory
    """
    if df.empty:
        print(f"No data available for Model {model_num}")
        return
    
    # Find HRC and distance columns
    col_map = {normalize_column(c): c for c in df.columns}
    hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
    dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
    
    if not hrc_col or not dist_col:
        print(f"Required columns not found for Model {model_num}")
        return
    
    x = df[dist_col].astype(float)
    y = df[hrc_col].astype(float)
    
    model_dir = os.path.join(output_dir, 'graphs', 'jominy', str(model_num))
    os.makedirs(model_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Main curve (no trend line)
    description = get_model_description(model_num)
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2, markersize=8, 
             color='#1f77b4', label=description)
    
    # Add data point annotations
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.annotate(f'{yi:.1f}', (xi, yi), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.xlabel('Distance from Cooling Edge [mm]', fontsize=12)
    plt.ylabel('Hardness HRC', fontsize=12)
    plt.title(f'Jominy Curve - {description}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    stats_text = f'Number of Measurements: {len(x)}\nAverage Hardness: {y.mean():.1f} HRC\nHardness Range: {y.min():.1f} - {y.max():.1f} HRC'
    if model_num not in [1, 4]:
        # Bottom left
        plt.text(0.02, 0.02, stats_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                 verticalalignment='bottom', fontsize=10)
    else:
        # Top left (default)
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                 verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(model_dir, 'main_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")

def plot_individual_detailed_analysis(df, model_num, output_dir):
    """
    Save each detailed analysis subplot as its own image in the model's folder, only if it contains data.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing data for the model
        model_num (int): Model number  
        output_dir (str): Base output directory
    """
    if df.empty:
        return
    
    col_map = {normalize_column(c): c for c in df.columns}
    hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
    dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
    
    if not hrc_col or not dist_col:
        return
    
    x = df[dist_col].astype(float)
    y = df[hrc_col].astype(float)
    
    model_dir = os.path.join(output_dir, 'graphs', 'jominy', str(model_num))
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Error analysis (moving average)
    if len(x) > 2:
        plt.figure(figsize=(7, 5))
        window_size = min(3, len(x))
        y_smooth = y.rolling(window=window_size, center=True).mean()
        plt.plot(x, y, 'o', alpha=0.6, label='Measurements')
        plt.plot(x, y_smooth, '-', linewidth=2, label='Moving Average')
        plt.xlabel('Distance [mm]')
        plt.ylabel('HRC')
        plt.title('Measurement Error Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(model_dir, 'error_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Image created: {output_path}")
    
    # 2. Hardness distribution histogram
    if len(y) > 0:
        plt.figure(figsize=(7, 5))
        plt.hist(y, bins=min(10, len(y)), alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('HRC')
        plt.ylabel('Frequency')
        plt.title('Hardness Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(model_dir, 'distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Image created: {output_path}")
    
    # 3. Cooling rate analysis (approximation)
    if len(x) > 1:
        plt.figure(figsize=(7, 5))
        cooling_rate = 1 / (x + 1)
        plt.plot(x, cooling_rate, marker='s', linestyle='-', color='red')
        plt.xlabel('Distance [mm]')
        plt.ylabel('Relative Cooling Rate')
        plt.title('Cooling Rate by Distance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_path = os.path.join(model_dir, 'cooling_rate.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Image created: {output_path}")

def plot_all_individual_models(models_data, output_dir):
    """
    Generate individual plots for all models.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    for model_num, df in models_data.items():
        plot_individual_jominy_curve(df, model_num, output_dir)
        plot_individual_detailed_analysis(df, model_num, output_dir)

def plot_summary_hardness_vs_time_4340(models_data, output_dir):
    """
    Generate and save a summary graph of average hardness vs. soaking time for 4340 steel (models 2, 3, 4).
    Uses hardcoded soaking times: 2=0 min, 3=30 min, 4=90 min.
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Hardcoded soaking times for 4340 models
    soak_info = {
        2: (0,   '4340 water (0 min)'),
        3: (30,  '4340 water (30 min)'),
        4: (90,  '4340 water (90 min)'),
    }
    soak_times = []
    avg_hrcs = []
    labels = []
    for model_num, (soak_time, label) in soak_info.items():
        df = models_data.get(model_num)
        if df is not None and not df.empty:
            col_map = {normalize_column(c): c for c in df.columns}
            hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
            if hrc_col:
                avg_hrc = df[hrc_col].astype(float).mean()
                soak_times.append(soak_time)
                avg_hrcs.append(avg_hrc)
                labels.append(label)

    if not soak_times or not avg_hrcs:
        print("No valid data for summary hardness vs. time graph.")
        return

    # Sort by soaking time
    sorted_indices = np.argsort(soak_times)
    soak_times = np.array(soak_times)[sorted_indices]
    avg_hrcs = np.array(avg_hrcs)[sorted_indices]
    labels = np.array(labels)[sorted_indices]

    summary_dir = os.path.join(output_dir, 'graphs', 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(soak_times, avg_hrcs, marker='o', linestyle='-', color='#2ca02c', linewidth=2)
    for x, y, label in zip(soak_times, avg_hrcs, labels):
        plt.annotate(f'{label}\n{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    plt.xlabel('Soaking Time [min]', fontsize=12)
    plt.ylabel('Average Hardness HRC', fontsize=12)
    plt.title('4340 Steel: Average Hardness vs. Soaking Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(summary_dir, 'hardness_vs_time_4340.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")
