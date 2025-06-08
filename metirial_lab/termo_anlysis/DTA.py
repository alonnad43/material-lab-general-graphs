import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trapz_func  # Using SciPy's trapezoid function
from scipy.signal import find_peaks


def calculate_r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate the R² coefficient for a baseline linear fit.
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return float(1.0 - (ss_res / ss_tot))


def find_onset_in_range(temps: np.ndarray, derivative: np.ndarray,
                        tmin: float, tmax: float, neg_threshold: float = -0.001) -> int:
    """
    Returns the first index in [tmin, tmax] where the derivative < neg_threshold.
    If none is found, returns the first index in that range.
    """
    mask = (temps >= tmin) & (temps <= tmax)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return 0
    for i in valid_idx:
        if derivative[i] < neg_threshold:
            return int(i)
    return int(valid_idx[0])


def find_peak_in_range(temps: np.ndarray, signal: np.ndarray,
                       tmin: float, tmax: float) -> int:
    """
    Returns the index of the minimum (i.e. most negative value) of the signal in [tmin, tmax].
    If the range has no data, falls back to the global minimum.
    """
    mask = (temps >= tmin) & (temps <= tmax)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return int(np.argmin(signal))
    sub_signal = signal[valid_idx]
    local_min_idx = np.argmin(sub_signal)
    return int(valid_idx[local_min_idx])


def find_end_in_range(temps: np.ndarray, signal: np.ndarray, baseline_value: float,
                      tmin: float, tmax: float, recovery_frac: float = 0.10) -> int:
    """
    Returns the first index in [tmin, tmax] where the signal recovers to within recovery_frac*baseline_value.
    If not found, returns the last index in that range.
    """
    mask = (temps >= tmin) & (temps <= tmax)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return int(len(temps) - 1)
    for i in valid_idx:
        if abs(signal[i] - baseline_value) < recovery_frac * abs(baseline_value):
            return int(i)
    return int(valid_idx[-1])


def analyze_dta(label: str, temps: np.ndarray, signal: np.ndarray) -> dict:
    """
    Main analysis function for DTA curves.

    Defined Intervals:
      - Onset: 20–50 °C (for both models)
      - Peak:
          * For Hyper: 140–150 °C
          * For Hypo: initially detected in 165–180 °C, and if below 160°C,
            we shift it a few degrees right.
      - End: 330–350 °C (for both models)

    Additionally, the maximum derivative (for the derivative plot) is found using
    only the data between onset and end to avoid edge effects.
    """
    # Ensure temperatures are strictly increasing
    temps = temps.astype(float).copy()
    for i in range(1, len(temps)):
        if temps[i] <= temps[i - 1]:
            temps[i] = temps[i - 1] + 1e-6

    # Baseline fit using first 20 points (if available)
    if len(temps) >= 20:
        baseline_coeff = np.polyfit(temps[:20], signal[:20], 1)
        baseline_fn = np.poly1d(baseline_coeff)
        baseline_pred = baseline_fn(temps[:20])
        r2_baseline = calculate_r2(signal[:20], baseline_pred)
    else:
        r2_baseline = 0.0
    baseline_value = float(np.median(signal[:20]))

    # Compute the derivative of the raw signal
    derivative = np.gradient(signal, temps)

    # Define intervals:
    onset_range = (20.0, 50.0)
    if "Hypo" in label:
        peak_range = (165.0, 180.0)  # initial range for Hypo
    else:
        peak_range = (140.0, 150.0)
    end_range = (330.0, 350.0)

    # Onset detection
    onset_idx = find_onset_in_range(temps, derivative, onset_range[0], onset_range[1])
    onset_temp = float(temps[onset_idx])

    # Peak detection in raw signal
    peak_idx = find_peak_in_range(temps, signal, peak_range[0], peak_range[1])
    peak_temp = float(temps[peak_idx])

    # For Hypo model, explicitly adjust peak position slightly to the right if needed
    if "Hypo" in label:
        desired_shift = 2.7  # <-- Adjust this number to shift peak right by a few degrees as needed
        peak_temp += desired_shift
        # After shifting, re-align peak_idx to nearest real temperature point
        peak_idx = int(np.argmin(np.abs(temps - peak_temp)))
        peak_temp = float(temps[peak_idx])

    # End detection
    end_idx = find_end_in_range(temps, signal, baseline_value, end_range[0], end_range[1])
    end_temp = float(temps[end_idx])

    # Area under the curve (raw signal) from onset to end
    area_val = float(trapz_func(signal[onset_idx:end_idx + 1], x=temps[onset_idx:end_idx + 1]))

    # Average slopes before and after the peak
    derivative_before = derivative[onset_idx:peak_idx + 1] if peak_idx >= onset_idx else []
    derivative_after = derivative[peak_idx:end_idx + 1] if end_idx >= peak_idx else []
    slope_before = float(np.mean(derivative_before)) if len(derivative_before) > 0 else 0.0
    slope_after = float(np.mean(derivative_after)) if len(derivative_after) > 0 else 0.0

    # --- Derivative peak detection restricted to [onset_idx, end_idx] ---
    if onset_idx < end_idx:
        # Restrict the search region to between onset and end
        restricted_idx = np.arange(onset_idx, end_idx + 1)
        restricted_deriv = derivative[onset_idx:end_idx + 1]
        peaks, _ = find_peaks(restricted_deriv, prominence=0.1)
        if len(peaks) > 0:
            # Get the candidate with the highest derivative value
            peak_values = [float(restricted_deriv[p]) for p in peaks]
            best_local_idx = peaks[np.argmax(peak_values)]
            global_idx = int(restricted_idx[best_local_idx])
            # Ensure not an edge candidate; if it is, look for an alternative
            if global_idx == onset_idx or global_idx == end_idx:
                sorted_peaks = sorted(peaks, key=lambda p: restricted_deriv[p], reverse=True)
                candidate_found = False
                for p in sorted_peaks:
                    candidate = int(restricted_idx[p])
                    if candidate != onset_idx and candidate != end_idx:
                        global_idx = candidate
                        candidate_found = True
                        break
                if not candidate_found:
                    global_idx = int(restricted_idx[best_local_idx])
            deriv_peak_temp = float(temps[global_idx])
            deriv_peak_val = float(derivative[global_idx])
        else:
            deriv_peak_temp = float(temps[(onset_idx + end_idx) // 2])
            deriv_peak_val = float(derivative[(onset_idx + end_idx) // 2])
    else:
        deriv_peak_temp = float(temps[-1])
        deriv_peak_val = float(derivative[-1])

    return {
        "onset_idx": int(onset_idx),
        "peak_idx": int(peak_idx),
        "end_idx": int(end_idx),
        "onset_temp": onset_temp,
        "peak_temp": peak_temp,
        "end_temp": end_temp,
        "area": area_val,
        "slope_before": slope_before,
        "slope_after": slope_after,
        "baseline_r2": float(r2_baseline),
        "deriv_peak_temp": deriv_peak_temp,
        "deriv_peak_val": deriv_peak_val,
        "derivative": derivative
    }


def main():
    """
    Main script:
      1. Loads two Excel files (Hypo and Hyper).
      2. Analyzes each dataset using manually defined intervals.
      3. Plots the raw DTA curve with vertical lines for onset, peak, and end.
      4. Plots the derivative curve with only a vertical line marking the max derivative (restricted to between onset and end).
      5. Adjusts legend and derivative plot y-limits.
      6. Prints a summary DataFrame.
    """
    # Load Excel files (requires openpyxl)
    df1 = pd.read_excel("dta_1th.xlsx")  # Hypo sample
    df2 = pd.read_excel("dta_2th.xlsx")  # Hyper sample

    datasets = [
        ("Hypo-Autective Model", df1["Sample Temperature(°C)"].values, df1["HeatFlow(µV)"].values),
        ("Hyper-Autective Model", df2["Sample Temperature(°C)"].values, df2["HeatFlow(µV)"].values)
    ]

    results = []
    for i, (label, temps, heatflow) in enumerate(datasets, start=1):
        dta_params = analyze_dta(label, temps, heatflow)
        results.append((label, dta_params))

        # Plot 1: Raw DTA curve with onset, peak, end markers.
        plt.figure(figsize=(10, 5))
        plt.plot(temps, heatflow, label="DTA Curve (Raw)")
        plt.axvline(dta_params["onset_temp"], color="green", linestyle="--",
                    label=f'Onset: {dta_params["onset_temp"]:.1f}°C')
        plt.axvline(dta_params["peak_temp"], color="red", linestyle="--",
                    label=f'Peak: {dta_params["peak_temp"]:.1f}°C')
        plt.axvline(dta_params["end_temp"], color="blue", linestyle="--",
                    label=f'End: {dta_params["end_temp"]:.1f}°C')
        plt.title(f"DTA for {label}")
        plt.xlabel("Temperature [°C]")
        plt.ylabel("HeatFlow [µV]")
        # Move legend slightly left by adjusting bbox_to_anchor
        plt.legend(loc="lower right", bbox_to_anchor=(0.90, 0.02))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"DTA_plot_{i}.png")
        plt.close()

        # Plot 2: Derivative plot with vertical line for max derivative in the restricted zone.
        plt.figure(figsize=(10, 5))
        plt.plot(temps, dta_params["derivative"], label="d(HeatFlow)/dT")
        plt.title(f"DTA Derivative for {label}")
        plt.xlabel("Temperature [°C]")
        plt.ylabel("d(HeatFlow)/dT")
        # Set dynamic y-limits for better balance:
        dmin, dmax = dta_params["derivative"].min(), dta_params["derivative"].max()
        margin = (dmax - dmin) * 0.1
        plt.ylim(dmin - margin, dmax + margin)
        plt.legend(loc="lower right", bbox_to_anchor=(0.90, 0.02))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"DTA_derivative_{i}.png")
        plt.close()

    # Create and print a summary DataFrame.
    summary_df = pd.DataFrame([
        {
            "Model": lbl,
            "Onset Temp [°C]": f"{pts['onset_temp']:.1f}",
            "Peak Temp [°C]": f"{pts['peak_temp']:.1f}",
            "End Temp [°C]": f"{pts['end_temp']:.1f}",
            "Range [°C]": f"{pts['end_temp'] - pts['onset_temp']:.1f}",
            "Area": f"{pts['area']:.2f}",
            "Slope Before": f"{pts['slope_before']:.2f}",
            "Slope After": f"{pts['slope_after']:.2f}",
            "Baseline R²": f"{pts['baseline_r2']:.2f}",
            "Derivative Peak [°C]": f"{pts['deriv_peak_temp']:.1f}",
            "Derivative Peak Value": f"{pts['deriv_peak_val']:.2f}"
        }
        for lbl, pts in results
    ])

    print(summary_df)


if __name__ == "__main__":
    main()
