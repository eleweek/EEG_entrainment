import sys 

import mne
import mne.time_frequency
import numpy as np

import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.plot import add_red_line_with_value
from libs.psd import get_peak_alpha_freq

raw = load_raw_xdf(sys.argv[1])
filter_and_drop_dead_channels(raw)

psd = raw.compute_psd(fmax=60.0)
peak_alpha_freq = get_peak_alpha_freq(psd)
fig = psd.plot(average=True)

add_red_line_with_value(fig, peak_alpha_freq)

plt.show()
