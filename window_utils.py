"""
window_functions.py
~~~~~~~~~~~~~~~~~~~

Provides utilities for generating and saving common window functions.
"""

import os
import numpy as np

# Define constants in uppercase
WINDOW_TYPES = ['triang', 'hann', 'hamming', 'blackman']

def get_triangle_window(size: int) -> np.ndarray:
    """
    Return a triangular (Bartlett) window of the given size.
    
    Args:
        size (int): The length of the window.
    
    Returns:
        np.ndarray: A 1-D array containing the triangular window values.
    """
    return np.bartlett(size)

def get_hann_window(size: int) -> np.ndarray:
    """
    Return a Hann window of the given size.
    
    Args:
        size (int): The length of the window.
    
    Returns:
        np.ndarray: A 1-D array containing the Hann window values.
    """
    return np.hanning(size)

def get_hamming_window(size: int) -> np.ndarray:
    """
    Return a Hamming window of the given size.
    
    Args:
        size (int): The length of the window.
    
    Returns:
        np.ndarray: A 1-D array containing the Hamming window values.
    """
    return np.hamming(size)

def get_blackman_window(size: int) -> np.ndarray:
    """
    Return a Blackman window of the given size.
    
    Args:
        size (int): The length of the window.
    
    Returns:
        np.ndarray: A 1-D array containing the Blackman window values.
    """
    return np.blackman(size)

# Map window name to the corresponding function
WINDOW_MAP = {
    'triang': get_triangle_window,
    'hann': get_hann_window,
    'hamming': get_hamming_window,
    'blackman': get_blackman_window
}

def export_windows(fft_size: int, output_dir: str = "Windows") -> None:
    """
    Export all defined window functions to text files.

    Args:
        fft_size (int): Size of each window (e.g., FFT size).
        output_dir (str, optional): Directory to save the window text files. Defaults to "Windows".

    Raises:
        ValueError: If fft_size is None or not a positive integer.
    """
    if fft_size is None or fft_size <= 0:
        raise ValueError("Parameter 'fft_size' must be a positive integer.")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save each window
    for window_type in WINDOW_TYPES:
        window_array = WINDOW_MAP[window_type](fft_size)
        file_path = os.path.join(output_dir, f"{window_type}.txt")
        np.savetxt(file_path, window_array, delimiter=' ')
    
    print(f"Windows successfully exported to: {os.path.abspath(output_dir)}")
