import mne

from brainflow.board_shim import BoardShim, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes

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