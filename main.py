## This script runs STFT table extraction, 
## applies spectral shaping,
## generates sounds by running them through model and
## saves them. FAD evaluation is performed in a separate 
## environment due to conflicting dependencies.

import extractor as e
import window_utils as wf
import gui

# 1. Run the setup GUI to get user inputs:
user_config = gui.run_setup_gui()

# 2. Map them to your local variables (or just use user_config dict directly)
DATA_F        = user_config['DATA_F']
SAMPLE_F      = user_config['SAMPLE_F']
FFT_N         = int(user_config['FFT_N'])
MEL_N         = int(user_config['MEL_N'])
MFCC_N        = int(user_config['MFCC_N'])
SAMPLE_N      = int(user_config['SAMPLE_N'])
SILENCE_LEVEL = int(user_config['SILENCE_LEVEL'])
USE_HDB       = bool(user_config['USE_HDB'])
CLUSTERS_N    = int(user_config['CLUSTERS_N'])


print("DATA_F:", DATA_F)
print("SAMPLE_F:", SAMPLE_F)
print("FFT_N:", FFT_N)
print("MEL_N:", MEL_N)
print("MFCC_N:", MFCC_N)
print("SAMPLE_N:", SAMPLE_N)
print("SILENCE_LEVEL:", SILENCE_LEVEL)
print("USE_HDB:", USE_HDB)
print("CLUSTERS_N:", CLUSTERS_N)

sr = 44100

# extract stft out of the dataset
stft = e.extract_stft(DATA_F, sr, FFT_N, MEL_N, MFCC_N, SAMPLE_N, SILENCE_LEVEL, USE_HDB, CLUSTERS_N)

# export windows for puredata
wf.export_windows(FFT_N, "Windows")

