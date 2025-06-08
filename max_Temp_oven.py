import re
import cv2  # OpenCV for image processing (install: pip install opencv-python)
print(cv2.__version__)
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # For statistical analysis and regression
import pytesseract

# Set the path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print(pytesseract.get_tesseract_version())


# --------------------------------------------------------------------------------
# Function: extract_heatmap_metrics
# --------------------------------------------------------------------------------
def extract_heatmap_metrics(image_path):
    """
    Extracts temperature metrics from a heat map image using OCR and image processing.
    The function returns a dictionary with:
      - max_temp: Maximum temperature read from the image.
      - min_temp: Minimum temperature read from the image.
      - avg_center_temp: Average temperature in the center region.
      - difference: Difference between max_temp and min_temp.
      - percentage_high, percentage_medium, percentage_low: Percentages of areas classified as high, medium, or low.

    Note: This implementation is a prototype. Adjust preprocessing, ROI segmentation,
          and threshold criteria according to your image specifications.
    """
    # Read the image from file
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} not found.")

    # Preprocess the image for better OCR performance:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Optional: Apply a binary threshold or blur to remove noise
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Use OCR to extract text; ideally the image has printed numbers for max and min
    ocr_result = pytesseract.image_to_string(thresh, lang='eng')

    # Use regular expressions to extract floating point numbers from OCR result
    numbers = re.findall(r'\d+\.\d+', ocr_result)
    # Assume the first two numbers represent max and min temperatures.
    if len(numbers) >= 2:
        max_temp = float(numbers[0])
        min_temp = float(numbers[1])
    else:
        # Fallback values if OCR fails; adjust accordingly
        max_temp = np.nan
        min_temp = np.nan

    # For the average temperature in the center, define a region of interest (ROI)
    h, w = thresh.shape
    center_roi = thresh[h // 4:3 * h // 4, w // 4:3 * w // 4]  # example: center 50% region
    # In a real scenario, you might calibrate pixel values to temperature.
    # Here, we simulate by assuming that OCR provided values in the image relate directly.
    # For demonstration, we calculate an average from the pixel intensities.
    avg_center_temp = np.mean(center_roi)  # This is a placeholder!

    # Calculate the difference between max and min temperatures
    difference = max_temp - min_temp if not (np.isnan(max_temp) or np.isnan(min_temp)) else np.nan

    # --------------------------------------------------------------------------------
    # Calculate percentages of high, medium, and low areas:
    # Define thresholds for segmentation (these values need calibration)
    # For example, using the intensity range (0-255), suppose:
    # Low: 0-85, Medium: 86-170, High: 171-255
    # In practice, map these to temperature ranges.
    total_pixels = thresh.size
    percentage_low = 100 * np.sum(thresh <= 85) / total_pixels
    percentage_medium = 100 * np.sum((thresh > 85) & (thresh <= 170)) / total_pixels
    percentage_high = 100 * np.sum(thresh > 170) / total_pixels

    return {
        'max_temp': max_temp,
        'min_temp': min_temp,
        'avg_center_temp': avg_center_temp,
        'difference': difference,
        'percentage_high': percentage_high,
        'percentage_medium': percentage_medium,
        'percentage_low': percentage_low
    }


# --------------------------------------------------------------------------------
# Main Processing: Load all images and extract metrics for each depth (X)
# --------------------------------------------------------------------------------
# Image file paths corresponding to each measurement depth
# (Make sure these filenames match your actual files)
image_paths = ["0cm.png", "5cm.png", "10cm.png", "15cm.png", "20cm.png", "25cm.png"]
depths_cm = [0, 5, 10, 15, 20, 25]

# Initialize lists to hold metrics
max_temps = []
min_temps = []
differences = []
avg_center_temps = []
perc_high = []
perc_medium = []
perc_low = []

# Loop over each image to extract data
for path in image_paths:
    metrics = extract_heatmap_metrics(path)
    max_temps.append(metrics['max_temp'])
    min_temps.append(metrics['min_temp'])
    differences.append(metrics['difference'])
    avg_center_temps.append(metrics['avg_center_temp'])
    perc_high.append(metrics['percentage_high'])
    perc_medium.append(metrics['percentage_medium'])
    perc_low.append(metrics['percentage_low'])

# Convert lists to numpy arrays for further processing
max_temps = np.array(max_temps)
min_temps = np.array(min_temps)
differences = np.array(differences)
avg_center_temps = np.array(avg_center_temps)
perc_high = np.array(perc_high)
perc_medium = np.array(perc_medium)
perc_low = np.array(perc_low)

# --------------------------------------------------------------------------------
# Error Estimation and Inaccuracy Calculations
# --------------------------------------------------------------------------------
# Example: If you have many pixel readings per ROI, you can compute the standard deviation.
# Here we simulate an error based on assumed uncertainties (replace with actual calculations).
# For instance, if each measurement has a known sensor error:
sensor_error = 0.5  # °C, example constant uncertainty
# For derived differences, propagate uncertainties:
error_diff = np.sqrt(sensor_error ** 2 + sensor_error ** 2)

# If multiple measurements per depth are available, use:
# error = np.std(pixel_values) / np.sqrt(N)
# and for regression use scipy.stats.linregress which yields standard errors.

# --------------------------------------------------------------------------------
# Linear Regression on the average center temperatures (as an example)
# --------------------------------------------------------------------------------
slope, intercept, r_value, p_value, std_err = stats.linregress(depths_cm, avg_center_temps)
predicted_avg = slope * np.array(depths_cm) + intercept


# --------------------------------------------------------------------------------
# Plotting the Enhanced Graph (in Hebrew)
# --------------------------------------------------------------------------------
def plot_temperature_data(depths, max_temps, min_temps, differences, avg_temps,
                          perc_high, perc_medium, perc_low,
                          slope, intercept, r_value, std_err, error_diff):
    """
    Plots various temperature data metrics for the oven in Hebrew.
    Displays:
      1. The change of max temperature per X.
      2. The change of min temperature per X.
      3. The difference between max and min per X.
      4. The percentage of high, medium, and low areas per map.
      5. The average center temperature per X.
    Also annotates the regression line and error estimates.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot max and min temperatures
    ax1.errorbar(depths, max_temps, yerr=sensor_error, fmt='-o', capsize=5, label='טמפרטורה מקסימלית')
    ax1.errorbar(depths, min_temps, yerr=sensor_error, fmt='-s', capsize=5, label='טמפרטורה מינימלית')
    # Plot the difference (with propagated error)
    ax1.errorbar(depths, differences, yerr=error_diff, fmt='-^', capsize=5, label='הפרש (מקסימום-מינימום)')
    # Plot the average center temperature
    ax1.errorbar(depths, avg_temps, fmt='-d', capsize=5, label='טמפרטורה ממוצעת בליבה')

    # Plot regression line for the average center temperatures
    ax1.plot(depths, predicted_avg, '--', label='התאמה ליניארית לממוצע')

    ax1.set_title("פילוג טמפרטורה בתנור", fontsize=14)
    ax1.set_xlabel("עומק בתנור [ס\"מ]", fontsize=12)
    ax1.set_ylabel("טמפרטורה [°C]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')

    # Create a second y-axis to show the percentages (high, medium, low)
    ax2 = ax1.twinx()
    ax2.plot(depths, perc_high, '-o', color='red', label='אחוז - גבוה')
    ax2.plot(depths, perc_medium, '-s', color='green', label='אחוז - בינוני')
    ax2.plot(depths, perc_low, '-^', color='blue', label='אחוז - נמוך')
    ax2.set_ylabel("אחוז", fontsize=12)
    ax2.legend(loc='lower right')

    # Annotate the regression line parameters on the plot
    annot_text = (f"קו ליניארי (ממוצע): y = {slope:.2f}x + {intercept:.2f}\n" +
                  f"R\u00b2 = {r_value ** 2:.3f}\n" +
                  f"Std Err = {std_err:.2f}")
    ax1.text(0.05, 0.95, annot_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# Call the plotting function with the extracted and computed data
plot_temperature_data(depths_cm, max_temps, min_temps, differences, avg_center_temps,
                      perc_high, perc_medium, perc_low,
                      slope, intercept, r_value, std_err, error_diff)
# Save the figure if needed