import sys

import numpy as np
import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels

from specparam import SpectralModel


input_filename = sys.argv[1]

raw = load_raw_xdf(input_filename)
filter_and_drop_dead_channels(raw, None)

psd = raw.compute_psd(fmin=1.0, fmax=45.0)
psd_values, psd_freqs = psd.get_data(return_freqs=True)

fm = SpectralModel()
print(psd_freqs.shape, psd_values.shape)
fm.report(psd_freqs, psd_values[-1], [3, 45])
fm.plot()

plt.show()