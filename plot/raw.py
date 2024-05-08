import sys

import mne
from libs.file_formats import load_recording
import matplotlib.pyplot as plt

from libs.filters import filter_and_drop_dead_channels

raw = load_recording(sys.argv[1])
filter_and_drop_dead_channels(raw, None)

# raw.plot(scalings=dict(eeg=50e-6))
raw.plot(scalings="auto")
plt.show()