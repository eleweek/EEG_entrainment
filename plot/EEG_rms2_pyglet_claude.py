import time
import argparse

import numpy as np
import mne
import pyglet

from mne_lsl.lsl import resolve_streams, StreamInlet

from libs.filters import filter_and_drop_dead_channels
from libs.parse import get_channels_from_xml_desc
from libs.plot import plot_psd

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt

def plot_raw_eeg(raw, duration, start_offset=1.0, end_offset=1.0):
    # Get the data, channel names, and sampling frequency
    data = raw.get_data()
    ch_names = raw.ch_names
    sfreq = raw.info['sfreq']

    # Define the time range to plot
    data_length_seconds = len(data[0]) / sfreq  # Total seconds of data available
    effective_duration = min(data_length_seconds, duration)  # Effective duration to plot
    start_time = start_offset
    end_time = effective_duration - end_offset
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)

    # Create a figure and subplots
    n_channels = len(ch_names)
    fig, axes = plt.subplots(n_channels, 1, figsize=(8, n_channels * 0.75), sharex=True)

    y_min = 50 * 1e-6
    y_max = -50 * 1e-6

    # Plot each channel in its own subplot
    # Calculate time array to only span actual data duration but ensure it's right-aligned by ending at 0
    total_samples = end_sample - start_sample
    if duration - effective_duration > 0:
        # Shift start point further left if the effective duration is less than the full duration
        left_bound = -duration + (duration - effective_duration) + start_offset
    else:
        left_bound = -duration + start_offset

    right_bound = -end_offset  # Ensuring the latest data aligns with time 0
    time_offsets = np.linspace(left_bound, right_bound, total_samples)

    for i, ax in enumerate(axes):
        ax.plot(time_offsets, data[i, start_sample:end_sample], color='black', linewidth=0.25)
        ax.set_ylabel(ch_names[i], rotation=0, labelpad=5, ha='left')
        ax.set_xlim(-duration + start_offset, -end_offset)  # Fixed x-axis from -duration to 0
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


def plot_to_pyglet(agg, fig):
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    width, height = canvas.get_width_height()
    return pyglet.image.ImageData(width, height, 'RGB', raw_data, pitch=-width * 3)

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

window = pyglet.window.Window(1000, 1000)
pyglet.gl.glClearColor(1, 1, 1, 1)

label = pyglet.text.Label('', font_name='monospace', font_size=18, x=10, y=10, color=(0, 0, 0, 255))

TOP_MARGIN = 20
LEFT_MARGIN = 20

all_data = None

@window.event
def on_draw():
    window.clear()
    label.draw()
    if 'psd_plot_pyglet_image' in globals():
        psd_plot_pyglet_image.blit(LEFT_MARGIN, window.height - TOP_MARGIN - psd_plot_pyglet_image.height)
    if 'raw_plot_pyglet_image' in globals():
        raw_plot_pyglet_image.blit(LEFT_MARGIN, window.height - TOP_MARGIN - psd_plot_pyglet_image.height - raw_plot_pyglet_image.height - 20)

def update(dt):
    global all_data, psd_plot_pyglet_image, raw_plot_pyglet_image, label

    # Pull data from the LSL stream
    data, _ = inlet.pull_chunk()

    if all_data is None:
        all_data = data.copy()
    else:
        all_data = np.concatenate((all_data, data), axis=0)
    
    if len(all_data) > int(sampling_rate) * max_seconds:
        all_data = all_data[-int(sampling_rate) * max_seconds:]

    if len(data) > 0:
        raw = mne.io.RawArray(all_data.T * scale_factor, mne.create_info(names, sampling_rate, ch_types='eeg'))
        filter_and_drop_dead_channels(raw, None)

        second_before_the_last_data = raw.get_data(start=len(raw.times) - int(sampling_rate) * 2, stop=len(raw.times) - int(sampling_rate))
        uvrms = np.sqrt(np.mean(second_before_the_last_data ** 2, axis=1)) * 1e6

        psd = raw.compute_psd(fmin=1.0, fmax=45.0)
        fig, _ = plot_psd(psd, title="PSD", average=True, ylim=(-20, 30))

        psd_plot_pyglet_image = plot_to_pyglet(agg, fig)

        if len(raw.times) > 3 * sampling_rate:
            fig = plot_raw_eeg(raw, max_seconds, start_offset=1.5, end_offset=1.5)
            raw_plot_pyglet_image = plot_to_pyglet(agg, fig)

        rms_text = f"uvRMS = {int(min(uvrms)):2d}‥{int(max(uvrms)):<3d}  {' '.join(f"{int(d):2d}" for d in uvrms)}"
        label.text = rms_text

pyglet.clock.schedule_interval(update, 0.3)
pyglet.app.run()