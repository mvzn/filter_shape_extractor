## This script runs STFT table extraction, 
## applies spectral shaping,
## generates sounds by running them through model and
## saves them. FAD evaluation is performed in a separate 
## environment due to conflicting dependencies.

import extractor as e
import stft_filter as stf
import window_utils as wf
import loudness_uniform as lu
import gui



sr = 44100 

FFT_N = 4096

MEL_N = 128

MFCC_N = 40

SIL_TH = 0.7

SAMPLE_N = 30000


CHOIR_F = r'C:/Users/markm/Desktop/New_Choir_Dataset/'
ARABIC_F = r'C:/Users/markm/Desktop/arabic_speech_corpus/'
MUSICNET_F = r'C:/Users/markm/Desktop/musicnet dataset'
SAMPLE_F = r'C:/Users/markm/.conda/envs/Dataset_STFT_Filtering_conda/samples'


# extract stft out of the dataset
stft = e.extract_stft(MUSICNET_F, sr, 4096, MEL_N, 20, SAMPLE_N, SIL_TH, True)

# run folder of audio files through FFT filtering
sample_stft_list = stf.process_audio_samples(SAMPLE_F, stft, 4096, SAMPLE_N)


# export windows for puredata
#wf.export_windows(FFT_N, "Windows")

