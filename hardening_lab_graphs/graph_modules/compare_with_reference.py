"""
Reference Curve Loader Module

This module provides a utility function to load reference curves from CSV files for comparison with experimental data.
It is used in the hardening_lab_graphs project to enable overlaying standard or literature curves on experimental plots, such as the Grossman curve. The function handles file existence checks and error reporting for missing or problematic files.
"""

import pandas as pd
import os

def load_reference_curve(ref_path):
    """
    Load a reference curve CSV for comparison.

    This function attempts to load a reference curve from a specified CSV file path. It is used to provide standard or literature data for overlaying on experimental plots, enabling direct comparison between measured and reference results. The function checks if the file exists, loads it as a pandas DataFrame, and returns it. If the file is missing or cannot be loaded, it prints an error message and returns None.

    Parameters:
        ref_path (str): Path to the reference CSV file.
    Returns:
        pd.DataFrame or None: The loaded reference data, or None if loading fails.
    """
    if not os.path.exists(ref_path):
        print(f'Reference file not found: {ref_path}')
        return None
    try:
        return pd.read_csv(ref_path)
    except Exception as e:
        print(f'Error loading reference: {e}')
        return None
