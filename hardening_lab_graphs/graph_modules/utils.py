"""
Utility Functions for Hardening Lab Analysis

This module provides utility functions for the hardening lab graphing system,
including data processing and special handling for model 4 data from 4.csv file.
"""

import pandas as pd
import os
import numpy as np

def normalize_column(column_name):
    """Normalize column name to lowercase for easier matching."""
    return str(column_name).lower().strip()

def load_model_4_data():
    """
    Load special model 4 data from 4.csv file using the new HRC column.
    Returns DataFrame with standardized columns for compatibility.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_4_path = os.path.join(base_dir, '..', 'data', '4.csv')
    
    if not os.path.exists(model_4_path):
        print(f"Warning: Model 4 data file not found at {model_4_path}")
        return None
    
    try:
        df = pd.read_csv(model_4_path)
        # Use D [mm] directly, rename for compatibility
        df['distance [mm ± 0.01]'] = df['D [mm]']
        df['number of Sample'] = 4
        df_converted = df[['distance [mm ± 0.01]', 'HRC', 'number of Sample']]
        return df_converted
    except Exception as e:
        print(f"Error loading model 4 data: {e}")
        return None

def clean_data(df):
    """
    Clean and prepare data for analysis.
    Remove invalid values and ensure proper data types.
    """
    # Find HRC and distance columns with flexible naming
    col_map = {normalize_column(c): c for c in df.columns}
    hrc_col = next((col_map[c] for c in col_map if 'hrc' in c), None)
    dist_col = next((col_map[c] for c in col_map if 'distance' in c), None)
    sample_col = next((col_map[c] for c in col_map if 'sample' in c or 'number' in c), None)
    
    if not hrc_col or not dist_col:
        return df
    
    # Convert to numeric and remove invalid entries
    df[hrc_col] = pd.to_numeric(df[hrc_col], errors='coerce')
    df[dist_col] = pd.to_numeric(df[dist_col], errors='coerce')
    
    # Remove rows with NaN values
    df = df.dropna(subset=[hrc_col, dist_col])
    
    return df

def get_unique_models(df):
    """Get list of unique model numbers from the dataset."""
    col_map = {normalize_column(c): c for c in df.columns}
    sample_col = next((col_map[c] for c in col_map if 'sample' in c or 'number' in c), None)
    
    if sample_col:
        return sorted(df[sample_col].unique())
    return []

def filter_by_model(df, model_num):
    """Filter DataFrame to only include data for specified model number."""
    col_map = {normalize_column(c): c for c in df.columns}
    sample_col = next((col_map[c] for c in col_map if 'sample' in c or 'number' in c), None)
    
    if sample_col:
        return df[df[sample_col] == model_num].copy()
    return df

def prepare_comparison_data(df):
    """
    Prepare data for comparison plots by organizing by model number.
    Returns dictionary with model numbers as keys and data as values.
    """
    models_data = {}
    models = get_unique_models(df)
    
    for model in models:
        model_df = filter_by_model(df, model)
        if not model_df.empty:
            models_data[model] = clean_data(model_df)
    
    # Add model 4 data if available
    model_4_data = load_model_4_data()
    if model_4_data is not None and not model_4_data.empty:
        models_data[4] = clean_data(model_4_data)
    
    return models_data

def ensure_output_directories():
    """Ensure all required output directories exist."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = [
        os.path.join(base_dir, 'graphs'),
        os.path.join(base_dir, 'graphs', 'comparisons'),
        os.path.join(base_dir, 'graphs', 'jominy'),
        os.path.join(base_dir, 'graphs', 'quench_time'),
        os.path.join(base_dir, 'graphs', 'images')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return base_dir

# Model number to descriptive name mapping for legends and titles
MODEL_DESCRIPTIONS = {
    1: "Grossman 4340 – quenched in oil",
    2: "Grossman 4340 – quenched in water",
    3: "Grossman 4340 – soak 30min @670°C → quenched in water",
    4: "Grossman 4340 – soak 90min @670°C → quenched in water",
    5: "Jominy 4340 – direct quench in water",
    6: "Grossman 1040 – quenched in oil",
    7: "Grossman 1040 – quenched in water",
    8: "Jominy 1040 – direct quench in water",
}

def get_model_description(model_num):
    return MODEL_DESCRIPTIONS.get(model_num, f"Model {model_num}")
