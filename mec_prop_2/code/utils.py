"""
Utility functions for loading data and common calculations.
"""
import pandas as pd
import os

def load_csv(folder, filename):
    """Load a CSV from the data directory."""
    base = os.path.join(os.path.dirname(__file__), '..', 'data', folder)
    path = os.path.join(base, filename)
    return pd.read_csv(path, encoding='latin1')

def ensure_graphs_dir():
    """Ensure the graphs directory exists."""
    graphs_dir = r'C:\Users\ramaa\Documents\metiral_lab_general _graphs\mec_prop_2\graphs'
    os.makedirs(graphs_dir, exist_ok=True)
    return graphs_dir
