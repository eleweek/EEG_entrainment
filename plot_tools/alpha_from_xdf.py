import sys 

import numpy as np

import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.plot import plot_psd


raw = load_raw_xdf(sys.argv[1])
filter_and_drop_dead_channels(raw)

psd = raw.compute_psd(fmin=1.0, fmax=60.0)
plot_psd(psd, title="PSD of the whole recording")

plt.show()
