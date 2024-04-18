import time
import argparse

import numpy as np
import mne
import pygame

from mne_lsl.lsl import resolve_streams, StreamInlet

from libs.filters import filter_and_drop_dead_channels
from libs.parse import get_channels_from_xml_desc
from libs.plot import plot_psd, plot_to_pygame

import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg

parser = argparse.ArgumentParser(
                    prog='EEG_rms2',
                    description='WIP, for now prints RMS values and the PSD plot')

parser.add_argument('--convert-uv', action='store_true', help='Convert uV to V')
args = parser.parse_args()

scale_factor = 1e-6 if args.convert_uv else 1.0

streams = resolve_streams()
print(streams)

eeg_streams = [stream for stream in streams if stream.stype.upper() == 'EEG']
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

TOP_MARGIN = 20
LEFT_MARGIN = 20

all_data = None
while True:
    # Pull data from the LSL stream
    data, _ = inlet.pull_chunk()

    if all_data is None:
        all_data = data.copy()
    else:
        all_data = np.concatenate((all_data, data), axis=0)
    
    if len(all_data) > int(sampling_rate) * max_seconds:
        all_data = all_data[-int(sampling_rate) * max_seconds:]

    print("All data", len(all_data))
    print("Pulled data", len(data))


    if len(data) > 0:
        # Convert data to a NumPy array
        print(f"All data shape: {all_data.shape}")

        raw = mne.io.RawArray(all_data.T * scale_factor, mne.create_info(names, sampling_rate, ch_types='eeg'))
        filter_and_drop_dead_channels(raw, None)
        # raw.pick(['O1', 'Oz', 'O2'])

        start_index = len(raw.times) - int(sampling_rate)
        last_second_data = raw.get_data(start=start_index)

        uvrms = np.sqrt(np.mean(last_second_data ** 2, axis=1)) * 1e6

        print("uVRMS:", uvrms)


        # Update screen
        pygame.event.get()

        psd = raw.compute_psd(fmin=1.0, fmax=45.0)
        fig, _ = plot_psd(psd, title="PSD", average=True, ylim=(-20, 30))

        

        psd_plot_pygame_image = plot_to_pygame(agg, fig)

        screen.fill(wht)
        screen.blit(psd_plot_pygame_image, (LEFT_MARGIN, TOP_MARGIN))

        fig = raw.plot(duration=5, show=False, show_scrollbars=False, show_scalebars=False, block=False)
        screen.blit(plot_to_pygame(agg, fig), (LEFT_MARGIN, TOP_MARGIN + psd_plot_pygame_image.get_height() + 20))


        trial_text = f"Most recent RMS: {','.join(str(int(d)) for d in uvrms)}"
        text = font.render(trial_text, True, gry) 
        screen.blit(text, text.get_rect(center=(screen_width/2, screen_height/2)))

        pygame.display.flip()
    
    time.sleep(0.3)