"""
Graphs for Three-Point and Four-Point Bending Tests
"""
from utils import load_csv, ensure_graphs_dir
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import linregress
import numpy as np

def linear_fit_elastic_region(x, y, max_strain=0.5):
    # Fit only initial range of extension (e.g., up to 0.5 mm or 10% of max x)
    mask = x <= min(max_strain, 0.1 * x.max())
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
        return slope, intercept, r_value, x_fit, y_fit
    return None, None, None, x_fit, y_fit

def plot_bending_graphs():
    """Generate bending test graphs for all materials and methods."""
    graphs_dir = ensure_graphs_dir()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    title_fontsize = 18
    label_fontsize = 12
    tick_fontsize = 10
    base = 'Three_Point_and_Four_Point_Bending_Tests'
    # --- Glass only 3PB ---
    glass_file = 'glass 3pb.csv'
    glass_path = os.path.join(os.path.dirname(__file__), '..', 'data', base, glass_file)
    df = pd.read_csv(glass_path, skiprows=5)
    fig, ax = plt.subplots()
    # There are 3 models, each with its own columns: (0,1), (3,4), (6,7)
    col_sets = [(0,1), (3,4), (6,7)]
    for i, (xcol, ycol) in enumerate(col_sets):
        x = df.iloc[:, xcol].dropna().to_numpy()
        y = df.iloc[:, ycol].dropna().to_numpy()
        if not x.size or not y.size:
            continue
        slope, intercept, r_value, x_fit, y_fit = linear_fit_elastic_region(x, y)
        # Use line only for dense data, marker for sparse
        if len(x) > 50:
            ax.plot(x, y, linestyle='-', linewidth=1.2, label=f'Specimen {i+1}')
            # Optionally mark start/end
            ax.plot([x[0], x[-1]], [y[0], y[-1]], 'o', markersize=4, alpha=0.7, color=ax.get_lines()[-1].get_color())
        else:
            ax.plot(x, y, marker='o', linestyle='-', markersize=2, alpha=0.4, label=f'Specimen {i+1}')
        if slope is not None and r_value**2 >= 0.5:
            ax.plot(x_fit, slope * x_fit + intercept, '--', label=f'Specimen {i+1} elastic fit (E={slope:.1f}, $R^2$={r_value**2:.2f})')
        elif slope is not None:
            ax.plot(x_fit, slope * x_fit + intercept, '--', label=f'Specimen {i+1} elastic fit (E={slope:.1f})')
    ax.set_xlabel('Extension (mm)', fontsize=label_fontsize)
    ax.set_ylabel('Load (N)', fontsize=label_fontsize)
    ax.set_title('Glass 3PB', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'glass_only_3pb.png'))
    print(f"Graph saved: {os.path.abspath(os.path.join(graphs_dir, 'glass_only_3pb.png'))}")
    plt.close()

    # --- All other materials overlay ---
    files = [
        ('Akulon 3pb.csv', 'Akulon', '3PB'),
        ('Akulon 4pb.csv', 'Akulon', '4PB'),
        ('Black akulon 4pb.csv', 'Black Akulon', '4PB'),
        ('Bkack akulom 3pb.csv', 'Black Akulon', '3PB'),
        ('Bakalit 4pb.csv', 'Bakelite', '4PB'),
        ('bakalit 3pm.csv', 'Bakelite', '3PB'),
    ]
    fig, ax = plt.subplots()
    for fname, mat, method in files:
        path = os.path.join(os.path.dirname(__file__), '..', 'data', base, fname)
        df = pd.read_csv(path, skiprows=5)
        x = df.iloc[:,0].dropna().to_numpy()
        y = df.iloc[:,1].dropna().to_numpy()
        label = f'{mat} {method}'
        # Use line only for all other materials (no markers)
        ax.plot(x, y, linestyle='-', linewidth=1.2, label=label)
        slope, intercept, r_value, x_fit, y_fit = linear_fit_elastic_region(x, y)
        if slope is not None and r_value**2 >= 0.5:
            ax.plot(x_fit, slope * x_fit + intercept, '--', label=f'{label} elastic fit (E={slope:.1f}, $R^2$={r_value**2:.2f})')
        elif slope is not None:
            ax.plot(x_fit, slope * x_fit + intercept, '--', label=f'{label} elastic fit (E={slope:.1f})')
    ax.set_xlabel('Extension (mm)', fontsize=label_fontsize)
    ax.set_ylabel('Load (N)', fontsize=label_fontsize)
    ax.set_title('Bending Curves: All Other Materials', fontsize=title_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'bending_all_other_materials.png'))
    print(f"Graph saved: {os.path.abspath(os.path.join(graphs_dir, 'bending_all_other_materials.png'))}")
    plt.close()
