import argparse
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes

import mne


def real_uvrms(data):
    """Compute the real root mean square."""
    return np.sqrt(np.mean(data ** 2))


def fake_uvrms(data):
    """Compute the 'fake RMS', which is the standard deviation."""
    return np.std(data)

def apply_mne_operations(data, sampling_rate, eeg_channels):
    # Convert the data to an MNE RawArray for more convenient processing
    ch_names = [f'EEG {ch}' for ch in eeg_channels]
    ch_types = ['eeg'] * len(eeg_channels)
    info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
    raw = mne.io.RawArray(data[eeg_channels, :] / 1e6, info)

    # Apply band-pass filter
    raw.filter(1.0, 45.0, method="iir", iir_params=None)

    # Remove environmental noise using notch filters at 50 Hz and 60 Hz
    raw.notch_filter(np.array([50, 60]), notch_widths=4, filter_length='auto', phase='zero', fir_design='firwin')

    # Return the filtered data
    return raw.get_data()


BUF_SIZE_SECONDS = 22  # the same as the OpenBCI GUI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='Serial port', required=True)
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.serial_port = args.serial_port
    board = BoardShim(BoardIds.CYTON_BOARD, params)

    board.prepare_session()
    board.start_stream()

    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"EEG channels: {eeg_channels}")
    print("Waiting for 3 seconds before starting streaming data...")
    time.sleep(3)

    try:
        while True:
            data = board.get_current_board_data(BUF_SIZE_SECONDS * sampling_rate)
            print("\n\n")
            print(f"Data shape: {data.shape}")

            mne_raw = apply_mne_operations(data.copy(), sampling_rate, eeg_channels)

            for channel_idx, channel in enumerate(eeg_channels):
                DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value), 1.0, 45.0, 4, FilterTypes.BUTTERWORTH.value, 1)
                DataFilter.remove_environmental_noise(data[channel], BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value), NoiseTypes.FIFTY.value)
                DataFilter.remove_environmental_noise(data[channel], BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value), NoiseTypes.SIXTY.value)

                # Get the last second of data
                last_second_data = data[channel][-sampling_rate:]
                fake_value = fake_uvrms(last_second_data)
                real_value = real_uvrms(last_second_data)

                last_second_mne_data = mne_raw[channel_idx, -sampling_rate:]

                print(f"Channel {channel}: fake = {fake_value:.2f} uV  real = {real_value:.2f} uV    MNE fake = {1e6 * real_uvrms(last_second_mne_data):.2f} uV real = {1e6 * fake_uvrms(last_second_mne_data):.2f} uV")

            time.sleep(1)

    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()