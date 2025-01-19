"""
@author: markm
"""
import os
import numpy as np
import librosa
import soundfile as sf
import window_utils as wu


## Here Spectral filtering takes place

def load_external_stft(file_path):
    """
    Loads external STFT data from a .txt file.
    """
    return np.loadtxt(file_path, dtype=np.float32)  


def get_stft(data, n_fft, hop_length):
    return librosa.stft(data, n_fft = n_fft, hop_length = hop_length, center=True, window=wu.get_hann_window(n_fft))

def stft_to_audio(data, n_fft, hop_length):
    return librosa.istft(data, n_fft=n_fft, hop_length = hop_length, center=True, window=wu.get_hann_window(n_fft))

def save_audio_sample(sample, sample_name, n_fft, sr, i, num_samples):
    os.makedirs(f'filtered_samples_{n_fft}_{num_samples}', exist_ok=True)
    name, ext = os.path.splitext(sample_name)
    output_path = os.path.join(f"filtered_samples_{n_fft}_{num_samples}/{name}_filtered_{i}_{n_fft}_{num_samples}{ext}")
    sf.write(output_path, sample, sr)

def process_audio_samples(data, e_stft, n_fft, num_samples):
    hop_length = n_fft // 4
    
    for root, dirs, files in os.walk(data):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path)
                
                sample_stft = get_stft(y, n_fft, hop_length)
                
                for i in range(0, len(e_stft.T)):
                    filtered_stft = sample_stft * e_stft.T[i][:, np.newaxis]
                    
                    filtered_sample = stft_to_audio(filtered_stft, n_fft, hop_length)
                    
                    save_audio_sample(filtered_sample, file, n_fft, sr, i, num_samples)

                
                
    print(f"Audio samples were processed and saved to: /filtered_samples_{n_fft}_{num_samples}")    # preferably from the name of the dataset folder         
    return 