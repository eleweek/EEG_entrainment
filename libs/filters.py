import mne

from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

import numpy as np

def filter_and_drop_dead_channels(raw, picks):
    data = raw.get_data()
    for channel in range(raw.info['nchan']):
        if np.all(data[channel] == data[channel][0]):
            print(f"Channel {channel} is dead, dropping it")
            raw.drop_channels([raw.ch_names[channel]])
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    raw.notch_filter(50, notch_widths=4, verbose=False)

    raw.pick(picks)

    raw.set_montage('standard_1020')


def create_new_raw_with_brainflow_filters_applied(raw):
    data = raw.get_data()
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    num_channels = raw.info['nchan']

    for channel in range(num_channels):
        DataFilter.perform_bandpass(data[channel], sampling_rate, 1.0, 45.0, 4, FilterTypes.BUTTERWORTH.value, 1)
        DataFilter.remove_environmental_noise(data[channel], sampling_rate, NoiseTypes.FIFTY.value)
        DataFilter.remove_environmental_noise(data[channel], sampling_rate, NoiseTypes.SIXTY.value)

    filtered_raw = mne.io.RawArray(data, raw.info)
    return filtered_raw