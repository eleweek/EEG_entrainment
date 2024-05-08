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
import matplotlib.pyplot as plt

def plot_raw_eeg(raw, duration, start_offset=1.0, end_offset=1.0):
    # Get the data, channel names, and sampling frequency
    data = raw.get_data()
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']

    # Define the time range to plot (first 10 seconds)
    start_time = start_offset
    end_time = (len(data[0]) / sfreq if duration is None else min(duration, len(data[0]) / sfreq)) - end_offset
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)

    # Create a figure and subplots
    n_channels = len(ch_names)
    fig, axes = plt.subplots(n_channels, 1, figsize=(8, n_channels * 0.75), sharex=True)

    ch_indices = [ch_names.index(ch) for ch in ['O1', 'O2', 'Oz'] if ch in ch_names]

    y_min = np.min(data[ch_indices, start_sample:end_sample])
    y_max = np.max(data[ch_indices, start_sample:end_sample])

    # Plot each channel in its own subplot
    time = np.arange(start_sample, end_sample) / sfreq
    for i, ax in enumerate(axes):
        ax.plot(time, data[i, start_sample:end_sample], color='black', linewidth=0.25)
        ax.set_ylabel(ch_names[i], rotation=0, labelpad=5, ha='left')
        ax.set_xlim(start_time, duration - start_offset - end_offset)
        ax.set_ylim(y_min, y_max)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
        ax.set_yticks([])

    # Set the x-label for the last subplot
    axes[-1].set_xlabel('Time (s)')

    # Adjust the spacing between subplots
    plt.tight_layout()

    return fig


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
font = pygame.font.SysFont("monospace", 18)

# colors
black = (0, 0, 0)
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

        second_before_the_last_data = raw.get_data(start=len(raw.times) - int(sampling_rate) * 2, stop=len(raw.times) - int(sampling_rate))
        # Calculate the RMS value for each channel, convert to uV
        # Use the second before the last second of data because the last second might have filter ringing
        uvrms = np.sqrt(np.mean(second_before_the_last_data ** 2, axis=1)) * 1e6

        print("uVRMS:", uvrms)


        # Update screen
        pygame.event.get()

        psd = raw.compute_psd(fmin=1.0, fmax=45.0)
        fig, _ = plot_psd(psd, title="PSD", average=True, ylim=(-20, 30))

        psd_plot_pygame_image = plot_to_pygame(agg, fig)

        screen.fill(wht)
        screen.blit(psd_plot_pygame_image, (LEFT_MARGIN, TOP_MARGIN))

        if len(raw.times) > 3 * sampling_rate:
            # fig = raw.plot(duration=5, show=False, show_scrollbars=False, show_scalebars=False, block=False)
            fig = plot_raw_eeg(raw, max_seconds, start_offset=1.5, end_offset=1.5)
            screen.blit(plot_to_pygame(agg, fig), (LEFT_MARGIN, TOP_MARGIN + psd_plot_pygame_image.get_height() + 20))


        rms_text = f"uvRMS = {int(min(uvrms)):2d}‥{int(max(uvrms)):<3d}  {' '.join(f"{int(d):2d}" for d in uvrms)}"
        text = font.render(rms_text, True, black) 
        screen.blit(text, (LEFT_MARGIN / 2, TOP_MARGIN / 2))

        pygame.display.flip()
    
    time.sleep(0.3)