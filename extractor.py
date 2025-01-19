"""
extractor.py
~~~~~~~~~~~~

Module for extracting MFCC features from .wav files, clustering them, and then
converting them back into STFT tables for further processing or export.
"""

import os
import sys
import numpy as np
import librosa
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.preprocessing import StandardScaler



###############################################################################
# Global objects / utilities
###############################################################################
scaler = StandardScaler()

def check_none_args(**kwargs) -> None:
    """
    Check if any of the given keyword arguments are None.
    If so, raises a ValueError listing which arguments are missing.

    Args:
        **kwargs: Arbitrary keyword arguments to check.

    Raises:
        ValueError: If any of the provided keyword arguments is None.
    """
    none_args = [name for name, value in kwargs.items() if value is None]
    if none_args:
        caller_name = sys._getframe(1).f_code.co_name
        raise ValueError(
            f"Missing arguments in '{caller_name}': {', '.join(none_args)}"
        )


###############################################################################
# Standardization functions
###############################################################################
def standardize(data: np.ndarray) -> np.ndarray:
    return scaler.fit_transform(data)

def unstandardize(data: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(data)


###############################################################################
# Feature extraction
###############################################################################
def extract_mfcc(
    file_path: str,
    sr: int,
    n_fft: int,
    n_mels: int,
    n_mfcc: int,
    hop_length: int
) -> np.ndarray:
    """
    Load a .wav file and extract its MFCC features.

    Args:
        file_path (str): Path to the .wav audio file.
        sr (int): Sample rate to load the audio at.
        n_fft (int): FFT window size for the MFCC computation.
        n_mels (int): Number of Mel bands to generate.
        n_mfcc (int): Number of MFCC coefficients to extract.
        hop_length (int): Hop length between frames.

    Returns:
        np.ndarray: A 2D array of shape (n_mfcc, T) representing the MFCC frames.
    """
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        hop_length=hop_length
    )
    return mfcc

def random_sample_frames(mfcc: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Shuffle and randomly select 'num_samples' frames from the given MFCC array.

    Args:
        mfcc (np.ndarray): MFCC array of shape (n_mfcc, T).
        num_samples (int): Number of frames to select.

    Returns:
        np.ndarray: A 2D array of shape (num_samples, n_mfcc) representing
                    the randomly selected MFCC frames, transposed so each row
                    is a single frame of MFCC coefficients.

    Raises:
        ValueError: If 'mfcc' or 'num_samples' is None (handled by check_none_args).
    """
    check_none_args(
        mfcc=mfcc,
        num_samples=num_samples
    )

    # Transpose so frames are along axis 0.
    frames = mfcc.T  # shape: (T, n_mfcc)
    np.random.shuffle(frames)
    return frames[:num_samples, :]  # shape: (num_samples, n_mfcc)


###############################################################################
# Data processing
###############################################################################
def process_data(
    data: str = None,
    sr: int = None,
    n_fft: int = None,
    n_mels: int = None,
    n_mfcc: int = None,
    num_samples: int = None,
    silence_level: float = 0.7
) -> np.ndarray:
    """
    Traverse a directory (recursively), find all .wav files, extract MFCC for each,
    remove frames considered 'silence', and randomly sample frames from each file.

    Args:
        data (str): Directory path containing .wav files.
        sr (int): Desired sample rate to load audio.
        n_fft (int): FFT window size for the MFCC computation.
        n_mels (int): Number of Mel bands to generate.
        n_mfcc (int): Number of MFCC coefficients to extract.
        num_samples (int): Total number of frames to sample across all files.

    Returns:
        np.ndarray: A 2D array of shape (num_samples, n_mfcc) combining frames
                    from all .wav files.

    Raises:
        ValueError: If any of the required arguments are None (handled by check_none_args).
    """
    check_none_args(
        data=data,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        num_samples=num_samples
    )

    mfcc_list = []
    hop_length = n_fft // 4

    # Count how many .wav files exist to compute how many frames per file.
    # (This ensures we don't oversample or get an indexing error.)
    all_wav_files = []
    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith('.wav'):
                all_wav_files.append(os.path.join(root, file))

    # Avoid division by zero if no .wav files are found
    if not all_wav_files:
        raise FileNotFoundError(f"No .wav files found in directory: {data}")

    frames_per_file = num_samples // len(all_wav_files)

    for file_path in all_wav_files:
        mfcc = extract_mfcc(file_path, sr, n_fft, n_mels, n_mfcc, hop_length)

        # Consider frames "silent" if the DC of MFCC (energy) is below
        # 70% of its minimum value:
        silence = np.where(mfcc[0, :] < (np.min(mfcc[0, :]) * silence_level))
        new_mfcc = np.delete(mfcc, silence, axis=1)

        sampled = random_sample_frames(new_mfcc, num_samples=frames_per_file)
        mfcc_list.append(sampled)

    print("Dataset was processed.")
    # Combine all sampled frames vertically:
    return np.vstack(mfcc_list)


###############################################################################
# Clustering functions
###############################################################################
def hdbscan_clustering(data: np.ndarray) -> tuple[int, np.ndarray]:
    """
    By analysing the data using HDBSCAN, estimate number of clusters.

    Args:
        data (np.ndarray): 2D array (n_samples, n_features).

    Returns:
        (num_clusters, labels): A tuple where 'num_clusters' is the
        estimated number of clusters, and 'labels' is an array of cluster
        assignments for each sample.
    """
    hdb = HDBSCAN()
    hdb.fit(data)
    num_clusters = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)
    return num_clusters

def kmeans_clustering(data: np.ndarray, num_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster the data using KMeans, returning cluster centers and labels.

    Args:
        data (np.ndarray): 2D array (n_samples, n_features).
        num_clusters (int): Number of clusters to form.

    Returns:
        (cluster_centers, labels): A tuple of cluster centers and an
        array of cluster assignments for each sample.
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=50)
    kmeans.fit(data)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers


###############################################################################
# MFCC -> Mel, Mel -> STFT
###############################################################################
def mfcc_to_mel(mfcc: np.ndarray, n_mels: int) -> np.ndarray:
    """
    Convert MFCC back into a Mel spectrogram.

    Args:
        mfcc (np.ndarray): 2D array of shape (T, n_mfcc).
        n_mels (int): Number of Mel bands.

    Returns:
        np.ndarray: 2D array of shape (n_mels, T) containing the Mel spectrogram.
    """
    return librosa.feature.inverse.mfcc_to_mel(mfcc, n_mels=n_mels)

def process_mfcc(
    mfcc: np.ndarray = None,
    n_mels: int = None,
    n_fft: int = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert MFCC to a Mel spectrogram, then invert to approximate STFT.

    Args:
        mfcc (np.ndarray): 2D array of shape (n_samples, n_mfcc) or (T, n_mfcc).
        n_mels (int): Number of Mel bands.
        n_fft (int): FFT size to use for the inverse operation.

    Returns:
        (mel_spectrogram, stft): A tuple containing the Mel spectrogram
        and the approximate STFT (as a complex array).

    Raises:
        ValueError: If any arguments are None (handled by check_none_args).
    """
    check_none_args(
        mfcc=mfcc,
        n_mels=n_mels,
        n_fft=n_fft
    )

    if mfcc.shape[0] != n_mels:
        mfcc = mfcc.T

    mel_spectrogram = mfcc_to_mel(mfcc, n_mels=n_mels)
    stft_result = librosa.feature.inverse.mel_to_stft(mel_spectrogram, n_fft=n_fft)

    print("MFCC was processed.")
    return mel_spectrogram, stft_result


###############################################################################
# Export / Master function
###############################################################################
def export_stft_tables(stft: np.ndarray, n_fft: int, n_mels: int, n_mfcc: int, num_clusters: int, num_samples: int) -> None:
    """
    Export the columns of the given STFT array into text files.

    Args:
        stft (np.ndarray): 2D array, shape (n_bins, num_clusters).
        num_clusters (int): Number of clusters (columns in 'stft' to export).

    Side Effects:
        - Creates a 'STFTs' directory if not already present.
        - Saves each cluster's STFT column in a separate .txt file.
    """
    os.makedirs(f'STFTs', exist_ok=True)

    for i in range(num_clusters):
        # stft[:, i] is the i-th cluster's STFT (across n_bins).
        col_data = np.asarray(stft[:, i])
        filename = os.path.join(f'STFTs', f"stft_{i}.txt")
        np.savetxt(filename, col_data, delimiter=' ')

def extract_stft(
    data: str,
    sr: int,
    n_fft: int,
    n_mels: int,
    n_mfcc: int,
    num_samples: int = 30000,
    silence_level: float = 0.7,
    use_hdb: bool = True,
    set_clusters: int = 10
) -> np.ndarray:
    """
    Master function to:
      1) Walk through a dataset directory, extract MFCC, and sample frames.
      2) Standardize and cluster the data using HDBSCAN to find # of clusters.
      3) Force the number of clusters to max(10, found_clusters).
      4) Use KMeans with that cluster count to get centroids.
      5) Convert those centroids from MFCC -> STFT and normalize.
      6) Export each cluster's STFT.

    Args:
        data (str): Directory containing .wav files.
        sr (int): Sample rate to load audio.
        n_fft (int): FFT size for MFCC extraction and STFT reconstruction.
        n_mels (int): Number of Mel bands for MFCC.
        n_mfcc (int): Number of MFCC coefficients to extract.
        num_samples (int, optional): Total frames to sample from all .wav files. Defaults to 20000.

    Returns:
        np.ndarray: Normalized STFT array of shape (n_bins, num_clusters).

    Side Effects:
        - Prints messages during each major step.
        - Exports STFT text files to a 'STFTs' directory.
    """
    # Process the data
    mfcc_data = process_data(data, sr, n_fft, n_mels, n_mfcc, num_samples, silence_level)

    # Standardize
    s_mfcc_data = standardize(mfcc_data)

    if use_hdb == True:
        # Get number of clusters with HDBSCAN 
        num_clusters = hdbscan_clustering(s_mfcc_data)
        print(f"Estimated number of clusters: {num_clusters}")
        # Ensure at most 10 clusters
        if num_clusters >= 10:
            num_clusters = 10
        else:
            num_cluster = num_clusters
    else:
        num_clusters = set_clusters
    

    

    # Use KMeans
    centroids = kmeans_clustering(s_mfcc_data, num_clusters) 

    # Convert centroids back to the original scale
    centroids_original_scale = unstandardize(centroids)

    # Convert MFCC -> STFT
    mel_spec, stft_result = process_mfcc(centroids_original_scale, n_mels, n_fft)

    # Normalize STFT magnitude
    stft_magnitude = np.abs(stft_result)
    max_value = np.max(stft_magnitude)
    stft_magnitude_normalized = stft_magnitude / max_value if max_value != 0 else stft_magnitude

    # Export each cluster's STFT
    export_stft_tables(stft_magnitude_normalized, n_fft, n_mels, n_mfcc, num_clusters, num_samples)

    print(f"STFT was extracted. Tables were exported to /STFTs/") #stft_{n_fft}_{n_mels}_{n_mfcc}_{num_samples}
    return stft_magnitude_normalized


