"""
File: derivative_plots_only.py

מפיק אך ורק את גרף הנגזרת עבור כל אחד מהדגמים (Hypo/Hyper).
פותר את בעיות האזהרה של PyCharm ע"י שימוש ב-numpy.typing.
מוודא שניתן להחזיר None בפונקציה לאיתור פיק, לפי הצורך.
"""

# מייבא ספריות דרושות
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from typing import Optional, Tuple
from numpy.typing import NDArray

# 1) פונקציית חישוב הנגזרת
def compute_derivative(
    temp_arr: NDArray[np.float64],
    heat_arr: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    מחשב d(HeatFlow)/dT באמצעות np.gradient
    """
    return np.gradient(heat_arr, temp_arr)

# 2) פונקציית איתור פיק בנגזרת
def find_derivative_peak(
    temp_arr: NDArray[np.float64],
    deriv_arr: NDArray[np.float64],
    prominence: float = 0.1
) -> Tuple[Optional[float], Optional[float]]:
    """
    מאתר את הפיק המשמעותי ביותר בנגזרת, מחזיר (peak_temp, peak_value).
    אם לא נמצא פיק, מחזיר (None, None).
    """
    peaks, _ = find_peaks(deriv_arr, prominence=prominence)
    if len(peaks) == 0:
        return None, None
    # בוחר את הפיק עם ערך הנגזרת המקסימלי
    max_idx = peaks[np.argmax(deriv_arr[peaks])]
    return float(temp_arr[max_idx]), float(deriv_arr[max_idx])

# 3) פונקציה לשרטוט גרף הנגזרת בלבד
def plot_derivative_curve(
    temp_arr: NDArray[np.float64],
    deriv_arr: NDArray[np.float64],
    peak_temp: Optional[float],
    peak_value: Optional[float],
    label: str,
    output_path: str
) -> None:
    plt.figure(figsize=(8, 5))

    # Plot the derivative
    plt.plot(temp_arr, deriv_arr, label="d(HeatFlow)/dT", color="blue")

    # If a peak was found, mark it
    if peak_temp is not None and peak_value is not None:
        plt.axvline(peak_temp, color="red", linestyle="--",
                    label=f"Peak: {peak_temp:.1f}°C")
        plt.plot(peak_temp, peak_value, 'o', color='red')

    plt.xlabel("Temperature [°C]")
    plt.ylabel("d(HeatFlow)/dT")
    plt.title(f"DTA Derivative Curve – {label}")
    plt.legend()
    plt.grid(True)

    # **Limit the y-axis** to [-100, 1000]
    plt.ylim(-100, 100)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 4) פונקציית main
def main():
    # מיקום הקבצים
    script_dir = r"C:\Users\ramaa\PycharmProjects\metirial_lab\termo_anlysis"

    # רשימת דגמים
    samples = [
        {"label": "Hypo-Eutectic",  "excel_file": "dta_1th.xlsx"},
        {"label": "Hyper-Eutectic", "excel_file": "dta_2th.xlsx"}
    ]

    # מגדיר טווח טמפרטורה אופציונלי (לפי בחירתך, או בטל)
    t_min, t_max = 100, 250

    for sample in samples:
        excel_path = os.path.join(script_dir, sample["excel_file"])
        try:
            df = pd.read_excel(excel_path, skiprows=33)

            # שליפת המידע כ-NDArray[np.float64]
            temperature_col = df.iloc[:, 1].to_numpy(dtype=np.float64)
            heatflow_col    = df.iloc[:, 2].to_numpy(dtype=np.float64)

            # ניקוי מערכים לא חוקיים
            valid_mask = (
                ~np.isnan(temperature_col) & ~np.isnan(heatflow_col) &
                np.isfinite(temperature_col) & np.isfinite(heatflow_col)
            )
            temp_data = temperature_col[valid_mask]
            hf_data   = heatflow_col[valid_mask]

            # אופציונלי: חיתוך טווח 100–250
            in_range_mask = (temp_data >= t_min) & (temp_data <= t_max)
            temp_data = temp_data[in_range_mask]
            hf_data   = hf_data[in_range_mask]

            if len(temp_data) < 2:
                print(f"Not enough data points for {sample['label']}")
                continue

            # חישוב הנגזרת
            deriv_data = compute_derivative(temp_data, hf_data)
            # החלפת ∞ או NaN
            deriv_data[~np.isfinite(deriv_data)] = np.nan

            # מציאת פיק בנגזרת
            peak_t, peak_val = find_derivative_peak(temp_data, deriv_data, prominence=0.1)

            # שם קובץ פלט
            out_png = os.path.join(
                script_dir,
                f"DTA_derivative_{sample['label'].replace(' ', '_')}.png"
            )

            # שרטוט ושמירה
            plot_derivative_curve(temp_data, deriv_data, peak_t, peak_val,
                                  label=sample["label"],
                                  output_path=out_png)

            print(f"Successfully created derivative plot for {sample['label']} at: {out_png}")

        except Exception as e:
            print(f"Error processing {sample['label']}: {e}")

# 5) קריאה לפונקציית main
if __name__ == "__main__":
    main()
