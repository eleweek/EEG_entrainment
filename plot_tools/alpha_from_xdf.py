import sys 

import numpy as np

import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.plot import add_red_line_with_value
from libs.psd import get_peak_alpha_freq, fit_one_over_f_curve


raw = load_raw_xdf(sys.argv[1])
filter_and_drop_dead_channels(raw)

psd = raw.compute_psd(fmin=1.0, fmax=60.0)
peak_alpha_freq = get_peak_alpha_freq(psd)

psd_freqs, fit_freq_range, fitted_curve, delta_db = fit_one_over_f_curve(psd, min_freq=3, max_freq=40, peak_alpha_freq=peak_alpha_freq)

fig = psd.plot(average=True, show=False)
ax = fig.get_axes()[0]
ax.plot(psd_freqs[fit_freq_range], fitted_curve, label='1/f fit', linewidth=1, color='darkmagenta')


add_red_line_with_value(fig, peak_alpha_freq, delta_db)

plt.show()
