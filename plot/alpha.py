import sys 
import argparse
import numpy as np
import mne
from mne.time_frequency import tfr_multitaper, tfr_stockwell, tfr_morlet
import matplotlib.pyplot as plt

from libs.file_formats import load_recording
from libs.filters import filter_and_drop_dead_channels
from libs.plot import plot_psd
from libs.parse import parse_picks
from libs.psd import get_peak_alpha_freq

parser = argparse.ArgumentParser(
                    prog='alpha_from_xdf',
                    description='Creates an alpha frequency plot and spectrograms from an XDF file')

parser.add_argument('input_xdf_filename', type=str, help='Path to the XDF file')
parser.add_argument('--picks', type=str, default=None, help='Comma or space-separated list of channels to use')
parser.add_argument('--separate-channels', action='store_true', help='Plot each channel separately')

args = parser.parse_args()
input_filename = args.input_xdf_filename
picks = parse_picks(args.picks)
separate_channels = args.separate_channels

def sliding_window_iaf(raw, window_size=5, step_size=1):
    iaf_estimates = []
    
    for start in range(0, len(raw.times) - int(window_size * raw.info['sfreq']), int(step_size * raw.info['sfreq'])):
        end = start + int(window_size * raw.info['sfreq'])
        window_raw = raw.copy().crop(tmin=raw.times[start], tmax=raw.times[end])
        window_psd = window_raw.compute_psd(fmin=1.0, fmax=45.0)
        iaf = get_peak_alpha_freq(window_psd)
        iaf_estimates.append(iaf)
    
    return iaf_estimates

def plot_iaf_histogram(iaf_estimates):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(iaf_estimates, bins=50, edgecolor='black')
    ax.set_title('Histogram of IAF Estimates')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Count')

    plt.tight_layout()
    return fig

def plot_spectrogram(raw, single_best_plot=True, multitaper=True, morlet=False, stockwell=False):
    BASELINE = (0.0, 0.1)

    # Create a single, large epoch encompassing the entire raw object
    events = np.array([[0, 0, 1]])
    event_id = dict(whole_recording=1)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=raw.times[0], tmax=raw.times[-1], baseline=None, preload=True)

    freqs = np.arange(7.0, 16.0, 0.25)

    fig_main, ax_main = plt.subplots(1, 1, figsize=(10, 5))

    if single_best_plot:
        power = tfr_multitaper(
            epochs,
            freqs=freqs,
            n_cycles=len(freqs),
            time_bandwidth=2.0,
            return_itc=False,
        )

        power.plot(
            [0],
            mode="mean",
            axes=ax_main,
            show=False,
            colorbar=True,
        )

    if multitaper:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
        for n_cycles, time_bandwidth, ax, title in zip(
            [freqs / 2, freqs, freqs / 2],
            [2.0, 4.0, 8.0],
            axs,
            [
                "Least smoothing, most variance",
                "Less frequency smoothing,\nmore time smoothing",
                "Less time smoothing,\nmore frequency smoothing",
            ],
        ):
            power = tfr_multitaper(
                epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                time_bandwidth=time_bandwidth,
                return_itc=False,
            )
            ax.set_title(title)
            power.plot(
                [0],
                baseline=BASELINE,
                mode="mean",
                axes=ax,
                show=False,
                colorbar=False,
            )

    if morlet:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")

        all_n_cycles = [1, 3, freqs / 2.0]
        for n_cycles, ax in zip(all_n_cycles, axs):
            power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
            power.plot(
                [0],
                mode="mean",
                baseline=BASELINE,
                axes=ax,
                show=False,
                colorbar=False,
            )
            n_cycles = "scaled by freqs" if not isinstance(n_cycles, int) else n_cycles
            ax.set_title(f"Using Morlet wavelet, n_cycles = {n_cycles}")

    if stockwell:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")

        fmin, fmax = freqs[[0, -1]]
        for width, ax in zip((0.2, 0.7, 3.0), axs):
            power = tfr_stockwell(epochs, fmin=fmin, fmax=fmax, width=width)
            power.plot(
                [0], baseline=BASELINE, mode="mean", axes=ax, show=False, colorbar=False
            )
            ax.set_title("Using S transform, width = {:0.1f}".format(width))

    return fig_main

# Main script
raw = load_recording(input_filename)
filter_and_drop_dead_channels(raw, picks)

psd = raw.compute_psd(fmin=1.0, fmax=45.0)

duration_seconds = len(raw.get_data()[0]) / raw.info['sfreq']
hours = int(duration_seconds // 3600)
minutes = int((duration_seconds % 3600) // 60)
seconds = int(duration_seconds % 60)

# Plot PSD in a separate window
title = f"PSD of the whole recording ({hours:02d}:{minutes:02d}:{seconds:02d}), channels = " + " ".join(raw.ch_names)
fig_psd, psd_data = plot_psd(psd, title=title, average=not separate_channels)

# Perform sliding window IAF estimation
iaf_estimates = sliding_window_iaf(raw, window_size=5, step_size=1)

# Plot IAF histogram
fig_iaf_hist = plot_iaf_histogram(iaf_estimates)

# Plot spectrogram
fig_spectrogram = plot_spectrogram(raw.copy(), single_best_plot=True, multitaper=False, morlet=False, stockwell=False)

plt.show()