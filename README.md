SETUP
1.	Download and install python 3.10
2.	Download and Install PureData
3.	Download and install nn~ object for PureData from https://github.com/acids-ircam/nn_tilde
4.	Open cmd and run C:/path/to/venv/Scripts/activate
5.	pip install -r /path/to/requirements.txt
6.	If you have any IDE able to run python script, open it
   - 	if not, python /path/to/main_script.py in cmd
7.	You’ll be presented with this GUI, select appropriate values, for best results it is recommended to go with the ones as in the image below. 
![image](https://github.com/user-attachments/assets/b37c36cd-1cf4-440e-8ab7-118f9de83273)

Note: Size of FFT PureData default FFT size is 4096, if you choose to change the value, change it in PureData as well. Check what is best for the dataset at hand.
Note: № of clusters Sometimes estimation goes out of bounds, you can select custom number of clusters, which will result in corresponding number of presets generated.
8.	After submitting, allow the spectral shape extraction for appx 1-3 minutes, depending on the size of your data set.
9.	Open PureData
10.	Run noise NoiseFilteringPrototype.pd
11.	Add path to your .ts file to  , it will gain an output as a result.
12.	Connect the new output of   to the left input of  
CONTROLS AND UI
![image](https://github.com/user-attachments/assets/1a8c6bb1-3094-490a-8c5f-f8c3f1c3e377)

1.	STFTs – STFT selection
2.	Window Types – Window Type selection
3.	Noise Input Gain – control Noise Input Gain
4.	Filtered Output Level – control Filtered Output Level
5.	STFT Table Level – control STFT Table Level
6.	Model Output Level – control Output Level
7.	Window - visualiser
8.	dSTFT - visualiser
9.	Spectrum – visualiser

For FAD evaluation you can go to [LINK](https://github.com/mvzn/FAD) and follow instructions
