"""
Graphs for Charpy Impact Test - Effect of Annealing Temperature
"""
from utils import load_csv, ensure_graphs_dir
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress
from scipy import optimize

def piecewise_linear(x, x0, y0, k1, k2):
    return np.where(x < x0, k1*(x-x0) + y0, k2*(x-x0) + y0)

def plot_annealing_impact_graphs():
    """Generate graphs for annealing temp vs hardness and fracture energy."""
    graphs_dir = ensure_graphs_dir()
    # Set font globally
    plt.rcParams['font.family'] = 'DejaVu Sans'
    title_fontsize = 18
    label_fontsize = 12
    tick_fontsize = 10

    # --- Hardness vs Annealing Temp ---
    df_hardness = load_csv('Charpy_Impact_Test_Effect_of_Annealing_Temperature', 'Table 1 – Hardness Test Results for Heat-Treated Samples (Rockwell C).csv')
    # Extract temperature from sample type
    def extract_temp(label):
        if 'No Annealing' in label:
            return 25  # Assume room temp
        import re
        m = re.search(r'(\d+)', label)
        return int(m.group(1)) if m else None
    temps = df_hardness['Sample Type'].apply(extract_temp).to_numpy()
    hardness = df_hardness[['Measurement 1', 'Measurement 2', 'Measurement 3']].astype(float)
    hardness_mean = hardness.mean(axis=1).to_numpy()
    hardness_std = hardness.std(axis=1).to_numpy()
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(temps, hardness_mean)
    fit_line = slope * temps + intercept
    # Piecewise fit if drop at 670°C is abrupt
    abrupt_drop = (hardness_mean[-2] - hardness_mean[-1]) > 10
    plt.figure()
    plt.errorbar(temps, hardness_mean, yerr=hardness_std, fmt='o', label='Mean ± SD')
    if abrupt_drop:
        # Fit piecewise linear: split at 670°C
        p , _ = optimize.curve_fit(piecewise_linear, temps, hardness_mean, [temps[-2], hardness_mean[-2], slope, -10])
        plt.plot(temps, piecewise_linear(temps, *p), 'g--', label='Piecewise fit')
    else:
        plt.plot(temps, fit_line, 'r--', label=f'Linear fit ($R^2$={r_value**2:.2f})')
    plt.xlabel('Temperature [°C]', fontsize=label_fontsize)
    plt.ylabel('Hardness [HRC]', fontsize=label_fontsize)
    plt.title('Hardness vs Annealing Temperature', fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'hardness_vs_annealing_temp.png'))
    print(f"Graph saved: {os.path.abspath(os.path.join(graphs_dir, 'hardness_vs_annealing_temp.png'))}")
    plt.close()

    # --- Fracture Energy vs Annealing Temp ---
    df_energy = load_csv('Charpy_Impact_Test_Effect_of_Annealing_Temperature', 'fracter_for_aneel_smp.csv')
    temps2 = df_energy['Sample Type'].apply(extract_temp).to_numpy()
    energy_kpm = df_energy['Fracture Energy [kpm]'].astype(float).to_numpy()
    energy_j = energy_kpm * 9.8
    # Identify 'not broken' points (machine limit)
    not_broken_mask = energy_kpm == 5.0
    fit_x = temps2[~not_broken_mask]
    fit_y = energy_j[~not_broken_mask]
    # 2nd-degree polynomial fit (exclude not broken)
    poly = np.polyfit(fit_x, fit_y, 2)
    poly_fn = np.poly1d(poly)
    plt.figure()
    plt.plot(fit_x, fit_y, 'o', label='Broken')
    plt.plot(fit_x, poly_fn(fit_x), 'b--', label='2nd-degree fit')
    # Plot 'not broken' points
    if np.any(not_broken_mask):
        plt.scatter(temps2[not_broken_mask], energy_j[not_broken_mask], color='red', marker='>', label='Not broken')
        for t, e in zip(temps2[not_broken_mask], energy_j[not_broken_mask]):
            plt.annotate('Not broken', (t, e), textcoords="offset points", xytext=(10,0), ha='left', color='red', fontsize=12)
    plt.xlabel('Temperature [°C]', fontsize=label_fontsize)
    plt.ylabel('Fracture Energy [J]', fontsize=label_fontsize)
    plt.title('Fracture Energy vs Annealing Temperature', fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'fracture_energy_vs_annealing_temp.png'))
    print(f"Graph saved: {os.path.abspath(os.path.join(graphs_dir, 'fracture_energy_vs_annealing_temp.png'))}")
    plt.close()
