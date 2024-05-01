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
    markers_stream = None

    for stream in streams:
        stream_type = stream['info']['type'][0]
        if stream_type.upper() == 'EEG':
            eeg_stream = stream
        elif stream_type == 'Markers':
            markers_stream = stream        

    if eeg_stream is None:
        raise ValueError('No EEG stream found in the XDF file')
    
    channel_descs = eeg_stream['info']['desc'][0]['channels'][0]["channel"]
    assert all(ch_desc['type'][0].upper() == 'EEG' for ch_desc in channel_descs)

    ch_names = [ch_desc["label"][0] for ch_desc in channel_descs]
    
    sfreq = eeg_stream['info']['nominal_srate'][0]
    data = 1e-6 * eeg_stream['time_series'].T

    info = mne.create_info(ch_names, sfreq, ch_types='eeg', verbose=False)
    raw = mne.io.RawArray(data, info, verbose=False)

    if markers_stream is not None:
        eeg_start_time = eeg_stream['time_stamps'][0]

        onset = markers_stream['time_stamps'][:-1] - eeg_start_time
        duration = markers_stream['time_stamps'][1:] - onset - eeg_start_time
        description = [ts[0].split(' ')[2].rstrip('.wav') for ts in markers_stream['time_series'][:-1]]
        annotations = mne.Annotations(onset, duration, description)

        raw.set_annotations(annotations)

    return raw


def load_recording(file_path):
    if file_path.endswith('.txt'):
        return load_openbci_txt(file_path)
    elif file_path.endswith('.xdf'):
        return load_raw_xdf(file_path)
    elif file_path.endswith('.vhdr'):
        return mne.io.read_raw(file_path, preload=True)