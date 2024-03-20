import sys 

import mne
import mne.time_frequency
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf

raw = load_raw_xdf(sys.argv[1])

print("Loaded annotations", raw.annotations)

zero_chan_indices = [4] # np.where(np.all(raw.get_data() < 1e-20, axis=1))[0]
zero_chan_names = [raw.ch_names[i] for i in zero_chan_indices]
raw.drop_channels(zero_chan_names)
print("Dropping channels", len(zero_chan_names), zero_chan_indices)

raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
raw.notch_filter(50, notch_widths=4, verbose=False)

psd = raw.compute_psd(fmax=45.0)
psds, freqs = psd.get_data(return_freqs=True)
avg_psd = np.mean(psds, axis=0)

alpha_range = (freqs >= 8) & (freqs <= 12)
peak_alpha_freq = freqs[alpha_range][np.argmax(avg_psd[alpha_range])]
print("PSDs alpha freq", psds[0, alpha_range])
print(f"Peak alpha frequency: {peak_alpha_freq:.2f} Hz")

fig = psd.plot(average=True)
for ax in fig.axes:
    ax.axvline(x=peak_alpha_freq, color='red', linestyle='-', linewidth=1.0)

    y_min, y_max = ax.get_ylim()
    y_shift = (y_max - y_min) * 0.05  # Adjust the multiplication factor as needed

    offset = matplotlib.transforms.ScaledTranslation(2/72, 0, fig.dpi_scale_trans)
    text_transform = ax.transData + offset

    ax.text(peak_alpha_freq, y_max - y_shift, f'{peak_alpha_freq:.2f}',
            ha='left', va='top', color='red', fontsize=8, transform=text_transform)

plt.show()
