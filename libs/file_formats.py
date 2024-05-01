import math

import mne
import numpy as np
import pandas as pd
import pyxdf

# A function to load data in Daniel Ingram's format here: https://osf.io/srfnz/
# Might not work for Muse data coming from other sources
def load_muse_csv(file_path):
    def estimate_sfreq(timestamps):
        """Estimate sampling frequency by bucketing data for each second."""
        bucket_counts = timestamps.dt.floor('s').value_counts()

        avg_count = bucket_counts.mean()
        sfreq = math.ceil(avg_count)

        return sfreq

    # Define the schema for the CSV file
    schema = {
        'TimeStamp': str,
        'RAW_TP9': float,
        'RAW_AF7': float,
        'RAW_AF8': float,
        'RAW_TP10': float
    }

    # Read the CSV file with the specified schema
    df = pd.read_csv(file_path, dtype=schema)

    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    data = df[['RAW_' + name for name in ch_names]].values.T

    # Estimate sampling frequency from timestamps
    timestamps = df['TimeStamp'].str.slice(0, 24)  # Truncate the timestamps
    timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M:%S.%f')


    sfreq = estimate_sfreq(timestamps)
    assert sfreq == 256, f'Unexpected sampling frequency: {sfreq}'

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=['eeg'] * len(ch_names), verbose=False)
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)

    return raw


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
    elif file_path.endswith('.csv'):
        return load_muse_csv(file_path)
    elif file_path.endswith('.xdf'):
        return load_raw_xdf(file_path)
    elif file_path.endswith('.vhdr'):
        return mne.io.read_raw(file_path, preload=True)