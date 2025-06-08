"""
Comparison Plotting Module for Hardening Lab Analysis

This module provides comprehensive plotting functions for comparing multiple models
in hardening lab analysis. It generates various comparison plots including:
- All models comparison
- Model pairs comparison  
- Distance vs HRC analysis
- English labeled plots for BGU Materials Lab reports
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize_column, get_model_description

# Set matplotlib to support default font (remove explicit font.family)

def plot_all_models_comparison(models_data, output_dir):
    """
    Plot comparison of all models on a single graph.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (model_num, df) in enumerate(models_data.items()):
        if df.empty:
            continue
            
        # Find HRC and distance columns
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
        
        if hrc_col and dist_col:
            x = df[dist_col].astype(float)
            y = df[hrc_col].astype(float)
            color = colors[i % len(colors)]
            
            plt.plot(x, y, marker='o', linestyle='-', linewidth=2, 
                    markersize=6, color=color, label=get_model_description(model_num))
    
    plt.xlabel('Distance [mm]', fontsize=12)
    plt.ylabel('HRC', fontsize=12)
    plt.title('All Models Comparison - Hardness by Distance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'graphs', 'comparisons', 'all_models_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")

def plot_model_pairs_comparison(models_data, output_dir):
    """
    Plot pairwise comparisons between models with custom titles and legends including model numbers.
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    # Custom comparison configuration
    comparison_graph_titles = [
        {"filename": "compare_model_1_2.png", "models": (1, 2), "title": "4340 Grossman – Effect of Quenching Medium (Oil vs. Water)"},
        {"filename": "compare_model_3_4.png", "models": (3, 4), "title": "4340 Grossman – Effect of Soaking Time on Hardness (30 min vs. 90 min @ 670°C)"},
        {"filename": "compare_model_5_8.png", "models": (5, 8), "title": "4340 vs. 1040 – Jominy Curve Comparison (Direct Quench in Water)"},
        {"filename": "compare_model_6_7.png", "models": (6, 7), "title": "1040 Grossman – Effect of Quenching Medium (Oil vs. Water)"},
    ]
    for comp in comparison_graph_titles:
        model1, model2 = comp["models"]
        title = comp["title"]
        filename = comp["filename"]
        if model1 not in models_data or model2 not in models_data:
            continue
        df1, df2 = models_data[model1], models_data[model2]
        if df1.empty or df2.empty:
            continue
        plt.figure(figsize=(10, 6))
        # Legend: always include model number and description
        desc1 = get_model_description(model1)
        desc2 = get_model_description(model2)
        leg1 = f"Model {model1}: {desc1}"
        leg2 = f"Model {model2}: {desc2}"
        # Plot both models
        for model_num, df, color, leg in [
            (model1, df1, '#1f77b4', leg1),
            (model2, df2, '#ff7f0e', leg2)
        ]:
            col_map = {normalize_column(c): c for c in df.columns}
            hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
            dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
            if hrc_col and dist_col:
                x = df[dist_col].astype(float)
                y = df[hrc_col].astype(float)
                plt.plot(x, y, marker='o', linestyle='-', linewidth=2, 
                        markersize=6, color=color, label=leg)
        plt.xlabel('Distance [mm]', fontsize=12)
        plt.ylabel('HRC', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'graphs', 'comparisons', filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Image created: {output_path}")

def plot_hardness_distribution(models_data, output_dir):
    """
    Plot hardness distribution analysis across all models.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    plt.figure(figsize=(12, 8))
    
    all_hardness = []
    model_labels = []
    
    for model_num, df in models_data.items():
        if df.empty:
            continue
            
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        
        if hrc_col:
            hardness_values = df[hrc_col].astype(float)
            all_hardness.extend(hardness_values)
            model_labels.extend([get_model_description(model_num)] * len(hardness_values))
    
    # Create box plot
    plt.figure(figsize=(12, 6))
    model_groups = {}
    for i, label in enumerate(model_labels):
        if label not in model_groups:
            model_groups[label] = []
        model_groups[label].append(all_hardness[i])
    
    plt.boxplot(model_groups.values(), labels=model_groups.keys())
    plt.ylabel('HRC', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title('Hardness Distribution by Model', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'graphs', 'comparisons', 'hardness_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")

def plot_distance_analysis(models_data, output_dir):
    """
    Plot distance analysis showing hardness change over distance.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    plt.figure(figsize=(12, 8))
    
    for model_num, df in models_data.items():
        if df.empty:
            continue
            
        col_map = {normalize_column(c): c for c in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
        
        if hrc_col and dist_col:
            x = df[dist_col].astype(float)
            y = df[hrc_col].astype(float)
            
            # Calculate hardness gradient (change in hardness per mm)
            if len(x) > 1:
                sorted_indices = np.argsort(x)
                x_sorted = x.iloc[sorted_indices]
                y_sorted = y.iloc[sorted_indices]
                
                # Smooth the data and calculate gradient
                if len(x_sorted) > 2:
                    gradients = np.gradient(y_sorted, x_sorted)
                    avg_gradient = np.mean(np.abs(gradients))
                    
                    plt.scatter(model_num, avg_gradient, s=100, alpha=0.7, 
                              label=f"Model {model_num}")
    
    plt.xlabel('Model Number', fontsize=12)
    plt.ylabel('Average Hardness Change [HRC/mm]', fontsize=12)
    plt.title('Hardness Gradient Analysis by Distance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'graphs', 'comparisons', 'distance_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")
