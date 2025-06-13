"""Graphs for Charpy Impact Test - Effect of Annealing Temperature."""

from utils import load_csv, ensure_graphs_dir
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import linregress
from scipy import optimize


def piecewise_linear(x, x0, y0, k1, k2):
    """Piecewise linear function used if there is a sharp drop."""
    return np.where(x < x0, k1 * (x - x0) + y0, k2 * (x - x0) + y0)


def plot_annealing_impact_graphs() -> None:
    """Generate hardness vs temperature and fracture energy plots."""
    graphs_dir = ensure_graphs_dir()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    title_fontsize = 18
    label_fontsize = 12
    tick_fontsize = 10

    # Hardness vs annealing temperature
    df_hardness = load_csv('Charpy_Impact_Test_Effect_of_Annealing_Temperature',
                           'Table 1 – Hardness Test Results for Heat-Treated Samples (Rockwell C).csv')

    def extract_temp(label: str) -> int:
        if 'No Annealing' in label:
            return 25
        import re
        m = re.search(r'(\d+)', label)
        return int(m.group(1)) if m else 0

    temps = df_hardness['Sample Type'].apply(extract_temp).to_numpy()
    hardness = df_hardness[[
        'Measurement 1', 'Measurement 2', 'Measurement 3'
    ]].astype(float)
    hardness_mean = hardness.mean(axis=1).to_numpy()
    hardness_std = hardness.std(axis=1).to_numpy()

    slope, intercept, r_value, _p, _std = linregress(temps, hardness_mean)
    fit_line = slope * temps + intercept
    abrupt_drop = (hardness_mean[-2] - hardness_mean[-1]) > 10

    plt.figure()
    plt.errorbar(temps, hardness_mean, yerr=hardness_std, fmt='o', label='Mean ± SD')
    if abrupt_drop:
        p, _ = optimize.curve_fit(piecewise_linear, temps, hardness_mean,
                                  [temps[-2], hardness_mean[-2], slope, -10])
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
    plt.close()

    # Fracture energy vs annealing temperature
    df_energy = load_csv('Charpy_Impact_Test_Effect_of_Annealing_Temperature',
                         'fracter_for_aneel_smp.csv')
    temps2 = df_energy['Sample Type'].apply(extract_temp).to_numpy()
    energy_kpm = df_energy['Fracture Energy [kpm]'].astype(float).to_numpy()
    energy_j = energy_kpm * 9.8

    not_broken_mask = energy_kpm == 5.0
    fit_x = temps2[~not_broken_mask]
    fit_y = energy_j[~not_broken_mask]

    poly = np.polyfit(fit_x, fit_y, 2)
    poly_fn = np.poly1d(poly)

    plt.figure()
    plt.plot(fit_x, fit_y, 'o', label='Broken')
    plt.plot(fit_x, poly_fn(fit_x), 'b--', label='2nd-degree fit')
    if np.any(not_broken_mask):
        plt.scatter(temps2[not_broken_mask], energy_j[not_broken_mask],
                    color='red', marker='>', label='Not broken')
        for t, e in zip(temps2[not_broken_mask], energy_j[not_broken_mask]):
            plt.annotate('Not broken', (t, e), textcoords="offset points",
                         xytext=(10, 0), ha='left', color='red', fontsize=12)
    plt.xlabel('Temperature [°C]', fontsize=label_fontsize)
    plt.ylabel('Fracture Energy [J]', fontsize=label_fontsize)
    plt.title('Fracture Energy vs Annealing Temperature', fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'fracture_energy_vs_annealing_temp.png'))
    plt.close()
