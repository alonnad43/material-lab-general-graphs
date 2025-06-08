# lengh_full.py
#
# This script reads data from an Excel file and produces two separate plots:
# 1. A continuous plot displaying all measurement data (Temperature vs. ΔL).
# 2. A plot for data up to 250°C in which linear regression is performed.
#
# In the second plot, the code calculates the regression parameters (slope, intercept),
# R², and the thermal expansion coefficient (α) using the formula:
#       α = slope / L0,
# where L0 is the initial length, here defined as 9.69 µm.
#
# Both plots are saved as separate image files in the same folder as the Excel file.
#
# Author: [Your Name]
# Date: [Current Date]

import pandas as pd  # For reading Excel file and data manipulation
import numpy as np  # For numerical calculations
import matplotlib.pyplot as plt  # For plotting graphs
import os  # For handling file paths


def read_data(file_path):
    """
    Reads experimental data from an Excel file.

    Parameters:
        file_path (str): The full path to the Excel file.

    Returns:
        DataFrame: The data from the file.
    """
    data = pd.read_excel(file_path)
    return data


def filter_data(data, lower_bound, upper_bound):
    """
    Filters the data to include only temperatures within a specified range.

    Parameters:
        data (DataFrame): The complete dataset.
        lower_bound (float): The lower temperature (inclusive).
        upper_bound (float): The upper temperature (inclusive).

    Returns:
        DataFrame: Filtered data.
    """
    mask = (data['Temperature(°C)'] >= lower_bound) & (data['Temperature(°C)'] <= upper_bound)
    filtered_data = data.loc[mask]
    return filtered_data


def perform_linear_regression(x_values, y_values):
    """
    Performs linear regression on the provided data.

    Parameters:
        x_values (array-like): Temperature values.
        y_values (array-like): ΔL values.

    Returns:
        tuple: (slope, intercept)
    """
    fit_coeffs = np.polyfit(x_values, y_values, 1)
    slope = fit_coeffs[0]
    intercept = fit_coeffs[1]
    return slope, intercept


def compute_r2(x_values, y_values, slope, intercept):
    """
    Computes the coefficient of determination (R²) for the linear fit.

    Parameters:
        x_values, y_values: The original data.
        slope, intercept: Regression parameters.

    Returns:
        float: R² value.
    """
    y_pred = slope * x_values + intercept
    ss_res = np.sum((y_values - y_pred) ** 2)
    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def calculate_alpha(slope, initial_length):
    """
    Calculates the thermal expansion coefficient (α).

    Parameters:
        slope (float): The slope from the linear regression.
        initial_length (float): The initial length (L₀) of the sample.

    Returns:
        float: Thermal expansion coefficient (α).
    """
    alpha = slope / initial_length
    return alpha


def plot_all_data(data, save_path):
    """
    Plots a continuous graph of all measurement data.

    Parameters:
        data (DataFrame): The complete dataset.
        save_path (str): Full path where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    # Plot continuous line with markers for each data point
    plt.plot(data['Temperature(°C)'], data['Delta L(µm)'], '-o', label='Measurement Data')
    plt.title('Continuous Plot of All Data Points')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('ΔL (µm)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_linear_regression_and_stats(filtered_data, lower_bound, upper_bound, initial_length, save_path):
    """
    Plots data up to a specified upper temperature (e.g., 250°C) with linear regression.
    The plot displays the regression line, R², regression equation, and the thermal
    expansion coefficient (α) in the plot annotation.

    Parameters:
        filtered_data (DataFrame): Data within the desired temperature range.
        lower_bound (float): Lower temperature bound (e.g., 50).
        upper_bound (float): Upper temperature bound (e.g., 250).
        initial_length (float): The initial length (L₀) of the sample.
        save_path (str): Full path where the plot image will be saved.
    """
    x = filtered_data['Temperature(°C)']
    y = filtered_data['Delta L(µm)']

    # Perform linear regression on the filtered data
    slope, intercept = perform_linear_regression(x, y)
    r2 = compute_r2(x, y, slope, intercept)
    alpha = calculate_alpha(slope, initial_length)

    # Prepare data for the regression line over the specified range
    temp_fit = np.linspace(lower_bound, upper_bound, 100)
    delta_l_fit = slope * temp_fit + intercept

    plt.figure(figsize=(8, 6))

    # Plot measurement data
    plt.plot(x, y, '-o', label='Measurement Data')
    # Plot the linear regression line
    plt.plot(temp_fit, delta_l_fit, '-', color='red', label='Linear Fit')

    plt.title('Linear Regression Plot up to 250°C')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('ΔL (µm)')
    plt.legend()
    plt.grid(True)

    # Create annotation text with regression data
    stats_text = (
        f"Fit Equation: ΔL = {slope:.3f}·T + {intercept:.3f}\n"
        f"R² = {r2:.3f}\n"
        f"Thermal Expansion Coefficient (α) = {alpha * 1e6:.3f} x10⁻⁶ 1/°C\n"
        f"L₀ = 9.69 µm"
    )
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, backgroundcolor='white', verticalalignment='top')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Main function execution:
        - Reads Excel data.
        - Plots continuous graph of all data.
        - Filters data up to 250°C, performs linear regression, and plots the regression results.
    """
    # Full path to the Excel file
    file_path = r"C:\Users\ramaa\PycharmProjects\metirial_lab\lengh\lengh.xlsx"

    # The folder in which the Excel file is located (also used for saving plots)
    save_folder = os.path.dirname(file_path)

    # Read data from the Excel file
    data = read_data(file_path)

    # Define the initial sample length (L₀); here, L₀ = 9.69 µm (average value)
    initial_length = 9.69

    # First plot: Continuous plot of all measurement data
    save_path_all = os.path.join(save_folder, "all_data_plot.png")
    plot_all_data(data, save_path_all)

    # Define the temperature range for linear regression (e.g., 50 to 250°C)
    temp_lower = 50
    temp_upper = 250

    # Filter the data for the specified temperature range
    filtered_data = filter_data(data, temp_lower, temp_upper)

    # Second plot: Plot for data up to 250°C with linear regression and statistical data
    save_path_regression = os.path.join(save_folder, "linear_regression_plot.png")
    plot_linear_regression_and_stats(filtered_data, temp_lower, temp_upper, initial_length, save_path_regression)

    print("Plots saved successfully in folder:", save_folder)


if __name__ == "__main__":
    main()
