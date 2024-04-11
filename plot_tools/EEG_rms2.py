import time

import numpy as np
import mne
import pygame

from mne_lsl.lsl import resolve_streams, StreamInlet

from libs.filters import filter_and_drop_dead_channels
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

all_data = []
max_seconds = 20


pygame.init()
pygame.display.init() 
screen_width, screen_height = 1000, 1000

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('EEG Noise RMS Display')
font = pygame.font.Font(None, 36)

# colors
gry = (128,128,128)
wht = (255,255,255)


while True:
    # Pull data from the LSL stream
    data, timestamps = inlet.pull_chunk()

    all_data.extend(data)[:-int(sampling_rate) * max_seconds]

    if len(data) > 0:
        # Convert data to a NumPy array
        all_data_array = np.array(all_data)
        print(f"All data shape: {all_data_array.shape}")

        raw = mne.io.RawArray(all_data_array.T, mne.create_info(names, sampling_rate, ch_types='eeg'))
        raw.set_montage('standard_1020')
        filter_and_drop_dead_channels(raw, None)

        start_index = len(raw.times) - int(sampling_rate)
        last_second_data = raw.get_data(start=start_index)

        uvrms = np.sqrt(np.mean(last_second_data ** 2, axis=1))

        # Update screen
        pygame.event.get()
        screen.fill(wht)
        trial_text = f"Most recent RMS: {','.join(str(d) for d in uvrms(raw.get_data()).tolist())}"
        text = font.render(trial_text, True, gry) 
        screen.blit(text, text.get_rect(center=(screen_width/2, screen_height/2)))
        pygame.display.flip()

        print("uVRMS:", uvrms)
        

    
    time.sleep(0.3)