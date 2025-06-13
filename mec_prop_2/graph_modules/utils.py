"""Utility functions for Mechanical Properties 2 graphs."""
import os
import pandas as pd


def load_csv(folder: str, filename: str) -> pd.DataFrame:
    """Load a CSV file from the data directory."""
    base = os.path.join(os.path.dirname(__file__), '..', 'data', folder)
    path = os.path.join(base, filename)
    return pd.read_csv(path, encoding='latin1')


def ensure_graphs_dir() -> str:
    """Ensure the graphs output directory exists and return its path."""
    graphs_dir = os.path.join(os.path.dirname(__file__), '..', 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    return graphs_dir
