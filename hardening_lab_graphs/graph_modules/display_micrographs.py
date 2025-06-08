"""
Display Micrographs Utility Module

This module provides a utility function to process microscope images for the hardening_lab_graphs project. Its main purpose is to copy all microscope image files (JPG, JPEG, PNG) from the project's images directory to the output graphs/images directory. This ensures that all relevant microstructure images are available alongside the generated graphs for reporting and analysis. The function handles directory creation and error reporting for missing or problematic files.
"""

import os
from shutil import copy2

def process_micrographs():
    """
    Copy microscope images from images/ to graphs/images/.

    This function scans the project's images directory for all files with image extensions (JPG, JPEG, PNG). For each image found, it copies the file to the graphs/images output directory, creating the destination directory if it does not exist. This is useful for consolidating all output images in a single location for easy access and reporting. The function prints a message for each file copied and reports any errors encountered during the process, such as missing source files or permission issues.
    """
    src = 'c:/Users/ramaa/Documents/metiral lab general graphs/hardening_lab_graphs/images'
    dst = 'c:/Users/ramaa/Documents/metiral lab general graphs/hardening_lab_graphs/graphs/images'
    os.makedirs(dst, exist_ok=True)
    for fname in os.listdir(src):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(src, fname)
            dst_path = os.path.join(dst, fname)
            try:
                copy2(src_path, dst_path)
                print(f'Copied {fname} to graphs/images/')
            except Exception as e:
                print(f'Error copying {fname}: {e}')
