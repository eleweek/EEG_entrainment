import mne
import numpy as np
import pyxdf

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


def load_raw_xdf(file_path):
    streams, _ = pyxdf.load_xdf(file_path)
    eeg_stream = None
    for stream in streams:
        if stream['info']['name'][0] == 'openbci_eeg':
            eeg_stream = stream
            break

    if eeg_stream is None:
        raise ValueError('No EEG stream found in the XDF file')

    channel_descs = eeg_stream['info']['desc'][0]['channels'][0]["channel"]
    assert all(ch_desc['type'][0] == 'EEG' for ch_desc in channel_descs)

    ch_names = [ch_desc["label"][0] for ch_desc in channel_descs]
    
    sfreq = eeg_stream['info']['nominal_srate'][0]
    data = 1e-6 * eeg_stream['time_series'].T
    print(data.shape)

    print(len(ch_names), ch_names)
    info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)

    return raw