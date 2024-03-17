import mne
import numpy as np

def load_openbci_txt(file_path):
    # TODO: unhardcode the sampling frequency and channel names
    sfreq = 250  # Sampling frequency of the OpenBCI data
    ch_names = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3',
                'EXG Channel 4', 'EXG Channel 5', 'EXG Channel 6', 'EXG Channel 7']

    # Read data from the text file
    data = np.loadtxt(file_path, delimiter=',', skiprows=5, usecols=range(1, 9))
    
    # Create MNE RawArray object
    info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data.T / 1e6, info, verbose=False)  # Convert to volts

    return raw