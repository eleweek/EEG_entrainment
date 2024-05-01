import sys

import mne
import numpy as np
import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.plot import add_red_line_with_value, plot_psd


def concatenate_and_get_psd(raws):
    concatenated = mne.concatenate_raws(raws)
    psd = concatenated.compute_psd(fmin=1.0, fmax=60.0)

    return psd

def get_raws_from_annotations(annotations, type):
    DURATION = 30.0  # hardcode for now for simplicity, TODO: potentially unhardcode later

    chunks = []
    for annotation in annotations:
        if annotation["description"] == type:
            chunk_data = raw.copy().crop(tmin=annotation["onset"], tmax=annotation["onset"] + DURATION)
            chunks.append(chunk_data)

    return chunks


def plot_subtracted_data(subtracted_data, freqs):
    median_data = np.median(subtracted_data, axis=0)
    freq_idx = np.where(eyes_open_psd.freqs >= 4)[0][0]


    fig, ax = plt.subplots()
    ax.plot(freqs[freq_idx:], median_data[freq_idx:], linewidth=1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title("Eyes closed - Eyes open")

    peak_alpha_freq = freqs[np.argmax(median_data[freq_idx:]) + freq_idx]
    add_red_line_with_value(fig, peak_alpha_freq, None)

    plt.show()


input_xdf_filename = sys.argv[1]

raw = load_raw_xdf(input_xdf_filename)
filter_and_drop_dead_channels(raw)

eyes_open_psd = concatenate_and_get_psd(get_raws_from_annotations(raw.annotations, "/open"))
eyes_closed_psd = concatenate_and_get_psd(get_raws_from_annotations(raw.annotations, "/close"))

eo_data, eo_freqs = eyes_open_psd.get_data(return_freqs=True)
ec_data, ec_freqs = eyes_closed_psd.get_data(return_freqs=True)

# Subtract the PSD data
subtracted_data = ec_data - eo_data.data

plot_psd(eyes_open_psd, "Eyes open")
plot_psd(eyes_closed_psd, "Eyes closed")
plot_subtracted_data(subtracted_data, eo_freqs)


plt.show()