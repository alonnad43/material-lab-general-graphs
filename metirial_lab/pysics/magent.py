import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import sys
sys.stdout.reconfigure(encoding='utf-8')

# === Experimental data ===
current = np.array([0.02, 0.52, 1.02, 1.52, 2.02, 2.52, 3.02, 3.52, 4.02, 4.52, 5.02])
mass_g = np.array([0.00, 0.10, 0.17, 0.32, 0.44, 0.54, 0.62, 0.65, 0.76, 0.88, 0.97])

# === Constants ===
g = 9.8  # gravity (m/s^2)
L = 0.042  # wire length in meters

# === Force calculation ===
force_n = (mass_g / 1000) * g  # Convert grams to kg, then compute F = m·g

# === Linear fit ===
slope, intercept = np.polyfit(current, force_n, 1)
fit_line = slope * current + intercept
B = slope / L  # Magnetic field

# === Metrics Calculation ===
# R-squared
residuals = force_n - fit_line
ss_res = np.sum(residuals**2)  # Sum of squared errors (SSE)
ss_tot = np.sum((force_n - np.mean(force_n))**2)  # Total sum of squares
r_squared = 1 - (ss_res / ss_tot)

# RMSE
rmse = np.sqrt(ss_res / len(force_n))

# 95% Confidence Intervals for slope and intercept
n = len(current)
alpha = 0.05  # 95% confidence level
t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
std_err_slope = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((current - np.mean(current))**2))
std_err_intercept = std_err_slope * np.sqrt(np.sum(current**2) / n)
conf_int_slope = (slope - t_crit * std_err_slope, slope + t_crit * std_err_slope)
conf_int_intercept = (intercept - t_crit * std_err_intercept, intercept + t_crit * std_err_intercept)

# Adjusted R-squared (optional for multiple regressors)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(current, force_n, 'o', label='Measured Force')
plt.plot(current, fit_line, '-', label=f'Fit: F = {slope:.5f}·I + {intercept:.2e}')
plt.xlabel('Current I (A)')
plt.ylabel('Force F (N)')
plt.title('Magnetic Force vs. Current')
plt.grid(True)
plt.legend()
plt.tight_layout()

# === Annotate Plot with Metrics ===
textstr = (f"Slope = {slope:.5f} N/A\n"
           f"Intercept = {intercept:.5f} N\n"
           f"95% CI for Slope: [{conf_int_slope[0]:.5f}, {conf_int_slope[1]:.5f}]\n"
           f"95% CI for Intercept: [{conf_int_intercept[0]:.5f}, {conf_int_intercept[1]:.5f}]\n"
           f"R² = {r_squared:.5f}\n"
           f"Adjusted R² = {adjusted_r_squared:.5f}\n"
           f"RMSE = {rmse:.5f} N\n"
           f"SSE = {ss_res:.5f} N²")

# Add the text box to the plot
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

# === Save ===
output_dir = r"C:\Users\ramaa\OneDrive\Nadav\לימודים\בן גוריון\לימודים\הנסדת חורמים\סמסטר ו\מעבדת פיזיקה 2\מגנתויות"
output_path = os.path.join(output_dir, "Magnetic_Force_vs_Current.png")
plt.savefig(output_path)

# === Print results ===
print(f"Saved graph to: {output_path}")
print(f"Slope = {slope:.5f} N/A")
print(f"Calculated Magnetic Field B = {B:.5f} T")
print(f"Slope = {slope:.5f} N/A")
print(f"Intercept = {intercept:.5f} N")
print(f"95% Confidence Interval for Slope: {conf_int_slope}")
print(f"95% Confidence Interval for Intercept: {conf_int_intercept}")
print(f"R² = {r_squared:.5f}")
print(f"Adjusted R² = {adjusted_r_squared:.5f}")
print(f"RMSE = {rmse:.5f} N")
print(f"SSE = {ss_res:.5f} N²")
