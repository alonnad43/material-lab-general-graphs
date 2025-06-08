"""
Project Purpose:
This script demonstrates a multi-experiment framework for analyzing various
material science data sets. It detects the experiment type from CSV data
(e.g., Stress-Strain, TTT Diagram, TGA Analysis, Hardness Profile) and
applies specialized logic and plotting for each experiment.

Structure:
1) Constants & Configuration:
   - BASE_DIR: The base directory for input/output.
   - GRAPHS_DIR: A subdirectory for saving plots.
   - EXPERIMENT_CONFIGS: A dictionary mapping experiment types to relevant parameters.
   - HANDLERS: A dictionary mapping experiment types to their specialized plotting function.

2) Utility Functions:
   - read_csv_file(): Safely reads a CSV file, returning a pandas DataFrame or None on error.
   - detect_experiment_type(): Inspects DataFrame columns to identify the experiment type.

3) Specialized Plot Functions:
   - plot_stress_strain(): Demonstrates advanced logic for stress-strain (e.g., offset yield).
   - plot_ttt_diagram(): Plots time-temperature-transformation data.
   - plot_tga_analysis(): Plots TGA data (temperature vs. mass).
   - plot_hardness_profile(): Plots hardness vs. depth.

4) Main Flow:
   - Collects CSV filenames from user input (or from a predefined list).
   - Reads each CSV into a DataFrame, detects the experiment type,
     and calls the corresponding specialized function.
   - Each plot is saved in the "graphs" directory.

Key Components:
- detect_experiment_type() ensures the code can decide which specialized logic to apply.
- The specialized plotting functions tailor analysis to each experiment type.
- The code is modular and easily extended with new experiments or specialized logic.

Usage:
1) Place your CSV files in the same directory or adjust BASE_DIR accordingly.
2) Run the script, and when prompted, enter the CSV filenames.
3) The script detects the experiment type and produces the corresponding plots
   in the "graphs" folder.
"""

# -------------------------------------------------------
# Import Statements
# -------------------------------------------------------
import os  # Interacting with the file system
import pandas as pd  # Handling CSV data
import matplotlib.pyplot as plt  # Plotting library

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

# 2 blank lines above each top-level function or class to comply with PEP 8


BASE_DIR = os.getcwd()  # Base directory for input/output
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")  # Directory for saving plots
os.makedirs(GRAPHS_DIR, exist_ok=True)  # Ensure the graphs directory exists

EXPERIMENT_CONFIGS = {
    "Stress-Strain": {
        "thickness": 2.0,
        "width": 6.0
        # Add more parameters if needed (e.g., gauge length, smoothing window)
    },
    "TTT Diagram": {},
    "TGA Analysis": {},
    "Hardness Profile": {}
}


def read_csv_file(filename, encoding='utf-8'):
    """
    Safely reads a CSV file into a pandas DataFrame using the specified encoding.
    If the file does not exist or an error occurs, returns None.
    """
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None

    try:
        df = pd.read_csv(filename, encoding=encoding)
        return df
    except Exception as exc:
        print(f"❌ Error reading {filename}: {exc}")
        return None


def detect_experiment_type(df):
    """
    Inspects the DataFrame columns (in lowercase) to identify the experiment type.
    Returns a string indicating the experiment type, or 'Unknown' if no match is found.
    """
    lower_cols = [col.lower() for col in df.columns]

    if "loadcell n" in lower_cols and any("strain" in c for c in lower_cols):
        return "Stress-Strain"
    if "time (s)" in lower_cols and "temperature (c)" in lower_cols:
        return "TTT Diagram"
    if "temperature (c)" in lower_cols and "mass (%)" in lower_cols:
        return "TGA Analysis"
    if "depth (mm)" in lower_cols and "hardness (hv)" in lower_cols:
        return "Hardness Profile"

    return "Unknown"


def plot_stress_strain(df, config, filename):
    """
    Specialized logic for Stress-Strain plots.
    Uses thickness and width from config (if present) to compute stress.
    """
    thickness = config.get("thickness", 2.0)
    width = config.get("width", 6.0)

    # Identify columns for strain and load
    strain_cols = [c for c in df.columns if "strain" in c.lower()]
    load_cols = [c for c in df.columns if "loadcell n" in c.lower()]

    if not strain_cols or not load_cols:
        print("⚠ Required columns for Stress-Strain not found.")
        return

    strain_col = strain_cols[0]
    load_col = load_cols[0]

    df[strain_col] = pd.to_numeric(df[strain_col], errors='coerce')
    df[load_col] = pd.to_numeric(df[load_col], errors='coerce')
    df.dropna(subset=[strain_col, load_col], inplace=True)

    area = thickness * width
    df["Engineering Stress"] = df[load_col] / area
    df["True Stress"] = df["Engineering Stress"] * (1 + df[strain_col])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[strain_col], df["Engineering Stress"], label="Engineering Stress", color="blue")
    ax.plot(df[strain_col], df["True Stress"], label="True Stress", color="red")

    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (MPa)")
    ax.set_title(f"Stress-Strain: {filename}")
    ax.legend()
    ax.grid(True)

    output_path = os.path.join(GRAPHS_DIR, f"stress_strain_{filename}.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✔ Saved Stress-Strain plot: {output_path}")


def plot_ttt_diagram(df, filename):
    """
    Specialized logic for TTT Diagram plots (Time vs. Temperature).
    """
    time_col = "Time (s)"
    temp_col = "Temperature (C)"

    if time_col not in df.columns or temp_col not in df.columns:
        print("⚠ Required columns for TTT Diagram not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[time_col], df[temp_col], marker="o", linestyle="--", color="blue")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"TTT Diagram: {filename}")
    ax.grid(True)

    output_path = os.path.join(GRAPHS_DIR, f"ttt_diagram_{filename}.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✔ Saved TTT Diagram: {output_path}")


def plot_tga_analysis(df, filename):
    """
    Specialized logic for TGA Analysis (Temperature vs. Mass).
    """
    temp_col = "Temperature (C)"
    mass_col = "Mass (%)"

    if temp_col not in df.columns or mass_col not in df.columns:
        print("⚠ Required columns for TGA Analysis not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[temp_col], df[mass_col], color="green")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Mass (%)")
    ax.set_title(f"TGA Analysis: {filename}")
    ax.grid(True)

    output_path = os.path.join(GRAPHS_DIR, f"tga_analysis_{filename}.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✔ Saved TGA Analysis: {output_path}")


def plot_hardness_profile(df, filename):
    """
    Specialized logic for Hardness Profile (Depth vs. Hardness).
    """
    depth_col = "Depth (mm)"
    hardness_col = "Hardness (HV)"

    if depth_col not in df.columns or hardness_col not in df.columns:
        print("⚠ Required columns for Hardness Profile not found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[depth_col], df[hardness_col], marker="o", color="purple")

    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("Hardness (HV)")
    ax.set_title(f"Hardness Profile: {filename}")
    ax.grid(True)

    output_path = os.path.join(GRAPHS_DIR, f"hardness_profile_{filename}.png")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"✔ Saved Hardness Profile: {output_path}")


HANDLERS = {
    "Stress-Strain": plot_stress_strain,
    "TTT Diagram": plot_ttt_diagram,
    "TGA Analysis": plot_tga_analysis,
    "Hardness Profile": plot_hardness_profile
}


def main():
    """
    Main function that coordinates user input for CSV filenames,
    reads each file, detects the experiment type, and calls the
    corresponding handler function.
    """
    print("Enter the CSV filenames (separated by commas):")
    files = input().split(",")
    files = [f.strip() for f in files if f.strip()]

    for csv_file in files:
        df = read_csv_file(csv_file)
        if df is None:
            continue

        exp_type = detect_experiment_type(df)

        if exp_type in HANDLERS:
            # Retrieve config if it exists; some handlers might ignore it if not needed
            config = EXPERIMENT_CONFIGS.get(exp_type, {})
            # Stress-Strain requires config, others do not
            if exp_type == "Stress-Strain":
                HANDLERS[exp_type](df, config, csv_file)
            else:
                # TTT Diagram, TGA Analysis, Hardness Profile do not use config
                HANDLERS[exp_type](df, csv_file)
        else:
            print(f"⚠ Unknown experiment type for file: {csv_file}")


if __name__ == "__main__":
    main()
