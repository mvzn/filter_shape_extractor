# gui_setup.py

import tkinter as tk
from tkinter import filedialog, ttk

def run_setup_gui():
    """
    Opens a Tkinter GUI, asks user for input, 
    and returns all variables as a dictionary.
    """
    # This dictionary will store the final results:
    config_data = {
        'DATA_F': None,
        'SAMPLE_F': None,
        'FFT_N': None,
        'MEL_N': None,
        'MFCC_N': None,
        'SAMPLE_N': None,
        'SILENCE_LEVEL': None,
        'USE_HDB': None,
        'CLUSTER_N': None
    }

    def browse_directory(var):
        path = filedialog.askdirectory(title="Select Folder")
        var.set(path)

    def browse_file(var):
        path = filedialog.askopenfilename(title="Select File")
        var.set(path)

    def submit():
        # Store the values in config_data
        config_data['DATA_F'] = data_f.get()
        config_data['FFT_N'] = fft_n.get()
        config_data['MEL_N'] = mel_n.get()
        config_data['MFCC_N'] = mfcc_n.get()
        config_data['SAMPLE_N'] = sample_n.get()
        config_data['SILENCE_LEVEL'] = silence_level.get()
        config_data['USE_HDB'] = use_hdb.get()
        config_data['CLUSTERS_N'] = clusters_n.get()
        
        # Close the GUI after submission
        root.destroy()
    
    # Create main window
    root = tk.Tk()
    root.title("STFT Masters Setup")

    # Variables
    data_f         = tk.StringVar(value="C:/Users/user/dataset_folder")
    fft_n          = tk.IntVar(value=4096)
    mel_n          = tk.IntVar(value=128)
    mfcc_n         = tk.IntVar(value=20)
    sample_n       = tk.IntVar(value=30000)
    silence_level  = tk.DoubleVar(value=.7)
    use_hdb       = tk.BooleanVar(value=True)
    clusters_n     = tk.IntVar(value=10)


    # ----- DATA_F -----
    frame_data_f = ttk.Frame(root)
    frame_data_f.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_data_f, text="Data Folder Path:").pack(side='left', padx=(0,5))
    ttk.Entry(frame_data_f, textvariable=data_f, width=60).pack(side='left', fill='x', expand=True)
    ttk.Button(frame_data_f, text="Browse", 
               command=lambda: browse_directory(data_f)).pack(side='left', padx=(5,0))


    # ----- FFT_N -----
    frame_fft_n = ttk.Frame(root)
    frame_fft_n.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_fft_n, text="Size of the FFT:").pack(side='left', padx=(0,5))
    ttk.Entry(frame_fft_n, textvariable=fft_n, width=10).pack(side='left', padx=(0,5))

    # ----- MEL_N -----
    frame_mel_n = ttk.Frame(root)
    frame_mel_n.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_mel_n, text="Size of Mel:").pack(side='left', padx=(0,5))
    ttk.Entry(frame_mel_n, textvariable=mel_n, width=10).pack(side='left', padx=(0,5))

    # ----- MFCC_N -----
    frame_mfcc_n = ttk.Frame(root)
    frame_mfcc_n.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_mfcc_n, text="Size of MFCC:").pack(side='left', padx=(0,5))
    ttk.Entry(frame_mfcc_n, textvariable=mfcc_n, width=10).pack(side='left', padx=(0,5))

    # ----- SAMPLE_N -----
    frame_sample_n = ttk.Frame(root)
    frame_sample_n.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_sample_n, text="Number of Random Samples:").pack(side='left', padx=(0,5))
    ttk.Entry(frame_sample_n, textvariable=sample_n, width=10).pack(side='left', padx=(0,5))

    # ----- SILENCE_LEVEL -----
    frame_silence_level = ttk.Frame(root)
    frame_silence_level.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_silence_level, text="Silence Threshold:").pack(side='left', padx=(0,5))
    ttk.Entry(frame_silence_level, textvariable=silence_level, width=10).pack(side='left', padx=(0,5))

    # ----- USE_HDB -----
    frame_use_hdb = ttk.Frame(root)
    frame_use_hdb.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_use_hdb, text="Use HDB?:").pack(side='left', padx=(0,5))
    ttk.Checkbutton(frame_use_hdb, variable=use_hdb, width=10).pack(side='left', padx=(0,5))

    # ----- CLUSTERS_N -----
    frame_clusters_n = ttk.Frame(root)
    frame_clusters_n.pack(padx=10, pady=5, fill='x')
    ttk.Label(frame_clusters_n, text="Number of Clusters:").pack(side='left', padx=(0,5))
    ttk.Entry(frame_clusters_n, textvariable=clusters_n, width=10).pack(side='left', padx=(0,5))

    # ----- SUBMIT BUTTON -----
    ttk.Button(root, text="Submit", command=submit).pack(padx=10, pady=10)

    # Start the event loop
    root.mainloop()

    return config_data  # Return the collected configuration
