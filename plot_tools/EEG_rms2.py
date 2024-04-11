import time

import numpy as np
import mne
from mne_lsl.lsl import resolve_streams, StreamInlet

from libs.parse import get_channels_from_xml_desc, print_xml_element

streams = resolve_streams()
print(streams)

eeg_streams = [stream for stream in streams if stream.stype == 'EEG']
if not eeg_streams:
    raise ValueError('No EEG streams found')

if len(eeg_streams) > 1:
    raise ValueError('Multiple EEG streams found, TODO: implement selection')


inlet = StreamInlet(eeg_streams[0])
inlet.open_stream()

stream_info = inlet.get_sinfo()
sampling_rate = stream_info.sfreq
n_channels = stream_info.n_channels
names = get_channels_from_xml_desc(stream_info.desc)

print(f"Found {n_channels} channels", names)
print("Sampling rate:", sampling_rate)
print("Units:", stream_info.get_channel_units())


window_samples = 250


while True:
    # Pull data from the LSL stream
    data, timestamps = inlet.pull_chunk(max_samples=window_samples)

    if len(data) > 0:
        # Convert data to a NumPy array
        data_array = np.array(data)

        

        uvrms = np.sqrt(np.mean(data_array ** 2, axis=0))

        print("uVRMS:", uvrms)

    
    time.sleep(1.0)