import sys

import numpy as np
import mne
import matplotlib.pyplot as plt

from libs.file_formats import load_recording
from libs.filters import filter_and_drop_dead_channels

raw = load_recording(sys.argv[1])
filter_and_drop_dead_channels(raw, None)

# Get the data, channel names, and sampling frequency
data = raw.get_data()
ch_names = raw.ch_names
sfreq = raw.info['sfreq']

# Define the time range to plot (first 10 seconds)
start_time = 5
end_time = 15
start_sample = int(start_time * sfreq)
end_sample = int(end_time * sfreq)

# Create a figure and subplots
n_channels = len(ch_names)
fig, axes = plt.subplots(n_channels, 1, figsize=(8, n_channels), sharex=True)

ch_indices = [ch_names.index(ch) for ch in ['O1', 'O2', 'Oz'] if ch in ch_names]

y_min = np.min(data[ch_indices, start_sample:end_sample])
y_max = np.max(data[ch_indices, start_sample:end_sample])

# Plot each channel in its own subplot
time = np.arange(start_sample, end_sample) / sfreq
for i, ax in enumerate(axes):
    ax.plot(time, data[i, start_sample:end_sample], color='black', linewidth=0.25)
    ax.set_ylabel(ch_names[i], rotation=0, labelpad=5, ha='left')
    ax.set_xlim(start_time, end_time)
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

# Show the plot
plt.show()