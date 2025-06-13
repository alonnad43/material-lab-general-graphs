"""
Graphs for Charpy Impact Test - Ductile to Brittle Transition
"""
from utils import load_csv, ensure_graphs_dir
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit

def sigmoid(T, A, k, T0, E_min):
    return A / (1 + np.exp(-k * (T - T0))) + E_min

def plot_dbtt_graph():
    """Generate DBTT energy vs temperature graph with transition marker and sigmoid fit."""
    graphs_dir = ensure_graphs_dir()
    plt.rcParams['font.family'] = 'DejaVu Sans'
    title_fontsize = 18
    label_fontsize = 12
    tick_fontsize = 10
    df = load_csv('Charpy_Impact_Test_Ductile_to_Brittle_Transition', 'Fracture_Energy_Temperature_for_Ductile_britle_transtion.csv')
    temps = df['Temperature [°C]'].astype(float).to_numpy()
    energy_kpm = df['Fracture Energy [kpm]'].astype(float).to_numpy()
    energy_j = energy_kpm * 9.8
    # Identify 'not broken' points (machine limit)
    not_broken_mask = energy_kpm == 5.0
    fit_temps = temps[~not_broken_mask]
    fit_energy = energy_j[~not_broken_mask]
    # Fit sigmoid (exclude not broken)
    p0 = [max(fit_energy)-min(fit_energy), 0.1, np.median(fit_temps), min(fit_energy)]
    popt, _ = curve_fit(sigmoid, fit_temps, fit_energy, p0, maxfev=10000)
    fit_y = sigmoid(temps, *popt)
    T0 = popt[2]
    plt.figure()
    plt.plot(fit_temps, fit_energy, 'o', label='Broken')
    plt.plot(temps, fit_y, 'b--', label='Sigmoid fit')
    # Plot 'not broken' points
    if np.any(not_broken_mask):
        plt.scatter(temps[not_broken_mask], energy_j[not_broken_mask], color='red', marker='>', label='Not broken')
        for t, e in zip(temps[not_broken_mask], energy_j[not_broken_mask]):
            plt.annotate('Not broken', (t, e), textcoords="offset points", xytext=(10,0), ha='left', color='red', fontsize=12)
    plt.axvline(T0, color='red', linestyle='--', label=f'DBTT (T0={T0:.1f}°C)')
    plt.xlabel('Temperature [°C]', fontsize=label_fontsize)
    plt.ylabel('Fracture Energy [J]', fontsize=label_fontsize)
    plt.title('Ductile-to-Brittle Transition (DBTT)', fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, 'dbtt_energy_vs_temp.png'))
    print(f"Graph saved: {os.path.abspath(os.path.join(graphs_dir, 'dbtt_energy_vs_temp.png'))}")
    plt.close()
