import numpy as np
import matplotlib.pyplot as plt
import os

# === Output Directory (Same as Previous) ===
output_dir = r"C:\Users\ramaa\OneDrive\Nadav\לימודים\בן גוריון\לימודים\הנסדת חורמים\סמסטר ו\מעבדת פיזיקה 2\גרעין"
os.makedirs(output_dir, exist_ok=True)

# === Data: Gaussian Distribution (n > 10) ===
n_gauss = np.array([
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36
])
freq_gauss = np.array([
    1, 5, 7, 6, 11, 18, 25, 39, 41, 36,
    38, 47, 35, 48, 35, 29, 24, 13, 10, 8,
    7, 7, 4, 3, 2, 0, 1
])
prev_gauss = freq_gauss / np.sum(freq_gauss)

# === Data: Poisson Distribution (n < 10) ===
n_poisson = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
freq_poisson = np.array([36, 111, 123, 106, 74, 33, 12, 4, 0, 1])
prev_poisson = freq_poisson / np.sum(freq_poisson)

# === Plot Gaussian Graph ===
plt.figure(figsize=(8, 5))
plt.bar(n_gauss, prev_gauss, color='steelblue', label='Experimental')
plt.xlabel("Counts (n)")
plt.ylabel("Prevalence")
plt.title("Gaussian Distribution of Radiation Counts")
plt.grid(axis='y', alpha=0.7)

# Fit a Gaussian curve to the data
from scipy.stats import norm
mean_gauss, std_gauss = np.mean(n_gauss), np.std(n_gauss)
x_gauss = np.linspace(min(n_gauss), max(n_gauss), 500)
y_gauss = norm.pdf(x_gauss, mean_gauss, std_gauss)

from scipy.optimize import curve_fit

def gaussian(x, a, b, c):
    return a * np.exp(-((x - b)**2) / (2 * c**2))

# Fit the Gaussian curve to the data
params, covariance = curve_fit(gaussian, n_gauss, prev_gauss, p0=[0.1, np.mean(n_gauss), np.std(n_gauss)])

# Extract the parameters
amplitude, mean, stddev = params

# Generate the Gaussian line
x_fit = np.linspace(min(n_gauss), max(n_gauss), 500)
y_fit = gaussian(x_fit, amplitude, mean, stddev)

# Plot the Gaussian line
plt.plot(x_fit, y_fit, color='red', label=f'Gaussian Fit\nμ={mean:.2f}, σ={stddev:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Graph_Gaussian_Distribution.png"))
plt.close()

# === Plot Poisson Graph ===
plt.figure(figsize=(8, 5))
plt.bar(n_poisson, prev_poisson, color='darkorange', label='Experimental')
plt.xlabel("Counts (n)")
plt.ylabel("Prevalence")
plt.title("Poisson Distribution of Radiation Counts")
plt.grid(axis='y', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Graph_Poisson_Distribution.png"))
plt.close()
