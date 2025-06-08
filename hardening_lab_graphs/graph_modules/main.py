"""
Comprehensive Hardening Lab Graphing System - Main Module

This is the main orchestrator for a 2-part graphing system designed for BGU Materials Lab
hardening analysis reports. It generates:

PART 1 - COMPARISON GRAPHS (8 total):
- All models comparison
- Model pairs comparison (7 pairs)

PART 2 - INDIVIDUAL MODEL GRAPHS (8 total):  
- Individual Jominy curves for each model
- Detailed analysis for each model

Features:
- Model 4 data override from 4.csv
- Data cleaning and column normalization
- Professional graph styling
- Comprehensive statistical analysis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    prepare_comparison_data, ensure_output_directories, 
    clean_data, load_model_4_data, normalize_column
)
from graph_grossman import (
    plot_all_models_comparison, plot_model_pairs_comparison,
    plot_hardness_distribution, plot_distance_analysis
)
from graph_jominy import plot_all_individual_models
from graph_quench_time import (
    plot_hardness_profile_analysis, plot_cooling_rate_estimation, 
    plot_statistical_summary
)
from display_micrographs import process_micrographs

# Set matplotlib defaults for professional appearance
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_hardness_data():
    """
    Load and prepare hardness measurement data from CSV files.
    
    Returns:
        dict: Dictionary with model numbers as keys and cleaned DataFrames as values
    """
    data_dir = os.path.join(BASE_DIR, '..', 'data')
    main_csv_path = os.path.join(data_dir, 'hardnes_messermts_new.csv')
    
    if not os.path.exists(main_csv_path):
        print(f"Error: Main data file not found at {main_csv_path}")
        return {}
    
    try:
        # Try reading as utf-8, fallback to latin1 if UnicodeDecodeError
        try:
            df = pd.read_csv(main_csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(main_csv_path, encoding='latin1')
            
        print(f"Loaded main data file: {main_csv_path}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Prepare comparison data (includes model 4 override)
        models_data = prepare_comparison_data(df)
        
        print(f"Processed {len(models_data)} models: {list(models_data.keys())}")
        for model_num, model_df in models_data.items():
            print(f"  Model {model_num}: {len(model_df)} data points")
            
        return models_data
        
    except Exception as e:
        print(f"Error loading hardness data: {e}")
        return {}

def generate_part1_comparison_graphs(models_data, output_dir):
    """
    Generate Part 1: Comparison Graphs (6 total as per specification)
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    print("\n=== PART 1: GENERATING COMPARISON GRAPHS ===")
    
    # 1. All models comparison
    print("1. Generating all models comparison...")
    plot_all_models_comparison(models_data, output_dir)
    
    # 2-6. Model pairs comparison (5 graphs to make total of 6)
    print("2-6. Generating model pairs comparisons...")
    plot_model_pairs_comparison(models_data, output_dir)
    
    print("✓ Part 1 complete: 6 comparison graphs generated")

def generate_part2_individual_graphs(models_data, output_dir):
    """
    Generate Part 2: Individual Model Graphs (for all models)
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    print("\n=== PART 2: GENERATING INDIVIDUAL MODEL GRAPHS ===")
    from graph_jominy import plot_individual_jominy_curve, plot_individual_detailed_analysis
    all_models = sorted(models_data.keys())
    print(f"Generating Jominy curves for models: {all_models}")
    for model_num in all_models:
        df = models_data[model_num]
        plot_individual_jominy_curve(df, model_num, output_dir)
        plot_individual_detailed_analysis(df, model_num, output_dir)
    print("✓ Part 2 complete: Jominy graphs for all models generated")

def generate_summary_report(models_data, output_dir):
    """
    Generate a summary report with key findings and statistics.
    
    Parameters:
        models_data (dict): Dictionary with model numbers as keys and DataFrames as values
        output_dir (str): Base output directory
    """
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    report_lines = []
    report_lines.append("Hardening Laboratory Analysis Report - Ben-Gurion University")
    report_lines.append("=" * 50)
    report_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    report_lines.append(f"Number of models analyzed: {len(models_data)}")
    report_lines.append("")
    # Statistical summary for each model
    report_lines.append("Statistical summary by model:")
    report_lines.append("-" * 30)
    for model_num, df in models_data.items():
        if df.empty:
            continue
        col_map = {col.lower().strip(): col for col in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        if not hrc_col:
            continue
        y = df[hrc_col].astype(float)
        report_lines.append(f"Model {model_num}: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}, n={len(y)}")
    # Overall analysis
    all_hardness = []
    for df in models_data.values():
        col_map = {col.lower().strip(): col for col in df.columns}
        hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
        if hrc_col:
            all_hardness.extend(df[hrc_col].astype(float).tolist())
    if all_hardness:
        all_hardness = np.array(all_hardness)
        report_lines.append("")
        report_lines.append(f"Overall summary: mean={all_hardness.mean():.2f}, std={all_hardness.std():.2f}, min={all_hardness.min():.2f}, max={all_hardness.max():.2f}, n={len(all_hardness)}")
    report_lines.append("")
    report_lines.append(f"Generated graph files:")
    report_lines.append(f"  Comparison graphs: graphs/comparisons/")
    report_lines.append(f"  Jominy graphs: graphs/jominy/")
    report_lines.append(f"  Quench time analysis: graphs/quench_time/")
    report_lines.append(f"  Micrograph images: graphs/images/")
    
    # Save report
    report_path = os.path.join(output_dir, 'graphs', 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        for line in report_lines:
            f.write(line + '\n')
    
    print(f"✓ Summary report saved: {report_path}")

def plot_hardness_vs_time_4340(models_data, output_dir):
    """
    Generate a summary graph: average hardness vs. soaking time for models 2, 3, 4 (4340 steel).
    Saves to graphs/summary/hardness_vs_time_4340.png
    """
    # Model: (soak_time_minutes, label)
    soak_info = {
        2: (0,   '4340 water (0 min)'),
        3: (30,  '4340 water (30 min)'),
        4: (90,  '4340 water (90 min)'),
    }
    soak_times = []
    avg_hardness = []
    labels = []
    for model_num, (soak_time, label) in soak_info.items():
        df = models_data.get(model_num)
        if df is not None and not df.empty:
            col_map = {col.lower().strip(): col for col in df.columns}
            hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
            if hrc_col:
                y = df[hrc_col].astype(float)
                soak_times.append(soak_time)
                avg_hardness.append(y.mean())
                labels.append(label)
    if not soak_times:
        print('No data for models 2, 3, 4 to plot hardness vs. time.')
        return
    # Plot
    summary_dir = os.path.join(output_dir, 'graphs', 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(soak_times, avg_hardness, marker='o', linestyle='-', color='#d62728')
    for x, y, label in zip(soak_times, avg_hardness, labels):
        plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
    plt.xticks(soak_times, labels, rotation=20)
    plt.xlabel('Soaking Time at 670°C (minutes)')
    plt.ylabel('Average Hardness (HRC)')
    plt.title('4340 Steel: Average Hardness vs. Soaking Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(summary_dir, 'hardness_vs_time_4340.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Image created: {output_path}")

def main():
    # Suppress matplotlib font warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
    
    print("\n=== BGU Materials Lab Hardening Analysis System ===")
    # Ensure output directories exist
    base_dir = ensure_output_directories()
    output_dir = base_dir
    
    # Load and prepare data
    models_data = load_hardness_data()
    if not models_data:
        print("No data loaded. Exiting.")
        return
    
    # PART 1: Generate comparison graphs
    generate_part1_comparison_graphs(models_data, output_dir)
    
    # PART 2: Generate individual model graphs
    generate_part2_individual_graphs(models_data, output_dir)
    
    # Process micrograph images
    print("\n=== COPYING MICROGRAPH IMAGES ===")
    try:
        process_micrographs()
    except Exception as e:
        print(f"Error processing micrograph images: {e}")
    
    # Generate summary report
    generate_summary_report(models_data, output_dir)
    # Generate summary graph for 4340
    plot_hardness_vs_time_4340(models_data, output_dir)
    
    print("\n=== ALL TASKS COMPLETE ===")
    print("All graphs, images, and reports are available in the 'graphs' directory.")

if __name__ == '__main__':
    main()
