import sys

import mne
import numpy as np
import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.plot import add_red_line_with_value
from libs.psd import get_peak_alpha_freq, fit_one_over_f_curve

# Load the MNE Raw file
raw = load_raw_xdf(sys.argv[1])
filter_and_drop_dead_channels(raw)

chunk_duration = 10.0

n_chunks = int(np.floor(raw.times[-1] / chunk_duration))

peak_alpha_freqs = []
dbs = []

# Iterate over the raw data in chunks
for i in range(n_chunks):
    print(f'Processing chunk {i + 1}/{n_chunks}')
    tmin = i * chunk_duration
    tmax = (i + 1) * chunk_duration
    
    # Get the data for the current chunk
    chunk_data = raw.copy().crop(tmin=tmin, tmax=tmax)

    psd = chunk_data.compute_psd(fmin=1.0, fmax=60.0)

    # Call the functions to get peak alpha frequency and dB
    peak_alpha_freq = get_peak_alpha_freq(psd)
    psd_freqs, fit_freq_range, fitted_curve, delta_db = fit_one_over_f_curve(psd, min_freq=3, max_freq=40, peak_alpha_freq=peak_alpha_freq)

    fig = psd.plot(average=True, show=False)
    ax = fig.get_axes()[0]
    ax.plot(psd_freqs[fit_freq_range], fitted_curve, label='1/f fit', linewidth=1, color='darkmagenta')

    add_red_line_with_value(fig, peak_alpha_freq, delta_db)
    
    peak_alpha_freqs.append(peak_alpha_freq)
    dbs.append(delta_db)

# Create the first chart for peak alpha frequency
fig, ax = plt.subplots()
ax.plot(np.arange(n_chunks) * chunk_duration, peak_alpha_freqs)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Peak Alpha Frequency (Hz)')
ax.set_title('Peak Alpha Frequency over Time')

# Create the second chart for dB
fig, ax = plt.subplots()
ax.plot(np.arange(n_chunks) * chunk_duration, dbs)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Decibel (dB)')
ax.set_title('Decibel over Time')

# Display the charts
plt.show()