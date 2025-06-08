#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Thermal Map Analysis
1) Graph 1: Max/Min Temperature vs. Distance
2) Graph 2: Red Fraction (%) and Blue+Purple Fraction (%) vs. Distance

We still do temperature mapping for the first graph using calibration.
For the second graph, we ignore temperature and directly measure how many
pixels are "red" vs. "blue+purple" in each image (in HSV space).
"""

import cv2
import numpy as np
import pytesseract
import re
import matplotlib.pyplot as plt
import sys

# ----------------------------------------------------------------------
# 1) Configuration
# ----------------------------------------------------------------------

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CALIBRATION_VALUES = {
    "0cm.png": (355.5, 338.6),
    "5cm.png": (359.3, 345.1),
    "10cm.png": (363.4, 338.2),
    "15cm.png": (373.6, 348.2),
    "20cm.png": (375.0, 350.7),
    "25cm.png": (367.7, 323.8)
}

IMAGE_FILES = ["0cm.png", "5cm.png", "10cm.png", "15cm.png", "20cm.png", "25cm.png"]
DISTANCES_CM = [0, 5, 10, 15, 20, 25]


# ----------------------------------------------------------------------
# 2) Helper Functions
# ----------------------------------------------------------------------

def extract_colorbar(image, width_percent=0.05):
    """
    Extract the rightmost 'width_percent' portion as the colorbar,
    leaving the rest as the 'thermal_map'.
    """
    h, w = image.shape[:2]
    bar_w = int(w * width_percent)
    colorbar = image[:, w - bar_w:]
    thermal_map = image[:, :w - bar_w]
    return colorbar, thermal_map


def ocr_temperature_values(colorbar):
    """
    Fallback for OCR if no calibration is found.
    """
    gray = cv2.cvtColor(colorbar, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    config_str = "--psm 6 --oem 3 -c tessedit_char_whitelist=-.0123456789"
    text = pytesseract.image_to_string(thresh, config=config_str, lang='eng')
    print("Debug OCR text:", text)

    numbers = re.findall(r'-?\d+\.\d+|-?\d+', text)
    if len(numbers) < 2:
        print("Warning: Less than two numbers detected; returning NaN.")
        return np.nan, np.nan
    num1, num2 = float(numbers[0]), float(numbers[1])
    return max(num1, num2), min(num1, num2)


def build_gradient_mapping(colorbar, max_temp, min_temp):
    """
    Convert each row in the colorbar to an average BGR color,
    linearly mapped from max_temp (top) to min_temp (bottom).
    """
    h = colorbar.shape[0]
    gradient_colors = np.mean(colorbar, axis=1)  # shape: (h, 3)
    gradient_temps = np.linspace(max_temp, min_temp, num=h)
    return gradient_colors, gradient_temps


def map_pixel_to_temperature(pixel_bgr, gradient_colors, gradient_temps):
    """
    Find closest row color in gradient_colors by Euclidean distance.
    """
    distances = np.linalg.norm(gradient_colors - pixel_bgr, axis=1)
    idx = np.argmin(distances)
    return gradient_temps[idx]


def convert_thermal_map_to_temperature(thermal_map, gradient_colors, gradient_temps):
    """
    Map each pixel in 'thermal_map' to a temperature via 'map_pixel_to_temperature'.
    """
    h, w = thermal_map.shape[:2]
    temp_array = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            temp_array[i, j] = map_pixel_to_temperature(thermal_map[i, j, :],
                                                        gradient_colors,
                                                        gradient_temps)
    return temp_array


def compute_image_metrics(temp_array):
    """
    Temperature-based metrics:
      - max, min
      - top 10% "high temp" threshold => high_temp_percentage
    """
    t_max = np.nanmax(temp_array)
    t_min = np.nanmin(temp_array)
    # We won't use std_temp if we don't need error bars, but let's keep it.
    t_std = np.nanstd(temp_array)

    threshold = t_min + 0.9 * (t_max - t_min)
    high_pixels = np.sum(temp_array >= threshold)
    total_px = temp_array.size
    if total_px < 1:
        high_pct = 0.0
    else:
        high_pct = 100.0 * high_pixels / total_px

    return {
        'max_temp_image': t_max,
        'min_temp_image': t_min,
        'std_temp': t_std,
        'high_temp_percentage': high_pct
    }


def compute_red_fraction(thermal_map):
    """
    Detect "red" in the thermal_map by HSV thresholding.
    Typically hue ranges for red are split:
      1) [0..10]
      2) [170..180]
    Adjust as needed for your images.
    """
    hsv = cv2.cvtColor(thermal_map, cv2.COLOR_BGR2HSV)
    # Lower red range
    lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    # Upper red range
    lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    total_px = thermal_map.size // 3  # since shape is (h,w,3)
    if total_px < 1:
        return 0.0
    red_pixels = np.sum(red_mask > 0)
    return 100.0 * red_pixels / total_px


def compute_bluepurple_fraction(thermal_map):
    """
    Detect "blue+purple" in HSV.  This is approximate;
    you might want to refine the hue range based on your images.

    Here, let's pick hue ~ [100..160], saturation >=70, value >=50.
    """
    hsv = cv2.cvtColor(thermal_map, cv2.COLOR_BGR2HSV)
    lower_bp = np.array([100, 70, 50], dtype=np.uint8)
    upper_bp = np.array([160, 255, 255], dtype=np.uint8)

    mask_bp = cv2.inRange(hsv, lower_bp, upper_bp)

    total_px = thermal_map.size // 3
    if total_px < 1:
        return 0.0
    bp_pixels = np.sum(mask_bp > 0)
    return 100.0 * bp_pixels / total_px


# ----------------------------------------------------------------------
# 3) Main Processing
# ----------------------------------------------------------------------
def process_image(image_path):
    """
    1) Load image, separate colorbar from thermal_map
    2) Build temperature array from calibration or OCR
    3) Compute temperature-based metrics
    4) Also compute color-based fractions (red + blue/purple)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    colorbar, thermal_map = extract_colorbar(img, 0.05)

    if image_path in CALIBRATION_VALUES:
        max_temp, min_temp = CALIBRATION_VALUES[image_path]
        print(f"Using calibration for {image_path}: max={max_temp}, min={min_temp}")
    else:
        max_temp, min_temp = ocr_temperature_values(colorbar)

    gradient_colors, gradient_temps = build_gradient_mapping(colorbar, max_temp, min_temp)
    temp_array = convert_thermal_map_to_temperature(thermal_map, gradient_colors, gradient_temps)
    temp_metrics = compute_image_metrics(temp_array)

    red_pct = compute_red_fraction(thermal_map)
    blue_pct = compute_bluepurple_fraction(thermal_map)

    return {
        'mapped_max_temp': temp_metrics['max_temp_image'],
        'mapped_min_temp': temp_metrics['min_temp_image'],
        'high_temp_percentage': temp_metrics['high_temp_percentage'],
        'red_fraction': red_pct,
        'bluepurple_fraction': blue_pct
    }


def main():
    mapped_max_temps = []
    mapped_min_temps = []
    high_temp_percents = []
    red_percentages = []
    bluepurple_percentages = []

    # Process images
    for img_file, dist in zip(IMAGE_FILES, DISTANCES_CM):
        results = process_image(img_file)
        mapped_max_temps.append(results['mapped_max_temp'])
        mapped_min_temps.append(results['mapped_min_temp'])
        high_temp_percents.append(results['high_temp_percentage'])
        red_percentages.append(results['red_fraction'])
        bluepurple_percentages.append(results['bluepurple_fraction'])

    # ----- Graph 1: Temperature vs. Distance -----
    plt.figure()
    plt.plot(DISTANCES_CM, mapped_max_temps, '-o', color='red', label='Max Temp')
    plt.plot(DISTANCES_CM, mapped_min_temps, '-s',color='blue', label='Min Temp')
    plt.title('Max & Min Temperature vs. Distance (cm)')
    plt.xlabel('Distance (cm)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.legend()
    plt.savefig("temperature_graph.png")
    plt.close()

    # ----- Graph 2: Color Fractions vs. Distance -----
    plt.figure()
    plt.plot(DISTANCES_CM, red_percentages, '-o', color='red', label='Hot area (%)')
    plt.plot(DISTANCES_CM, bluepurple_percentages, '-s', color='blue', label='Cold area (%)')
    plt.title('Hot & Cold Fractions vs. Distance (cm)')
    plt.xlabel('Distance (cm)')
    plt.ylabel('Pixels (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig("color_fractions_graph.png")
    plt.close()

    # End
    sys.exit(0)


if __name__ == "__main__":
    main()
