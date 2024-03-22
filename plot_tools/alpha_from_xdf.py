import sys 

import mne.time_frequency
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.plot import add_red_line_with_value
from libs.psd import get_peak_alpha_freq

def one_over_f(freq, alpha, beta):
    return alpha / freq ** beta


raw = load_raw_xdf(sys.argv[1])
filter_and_drop_dead_channels(raw)

psd = raw.compute_psd(fmax=60.0)
peak_alpha_freq = get_peak_alpha_freq(psd)

psd_values, psd_freqs = raw.compute_psd(fmin=1.0, fmax=45.0).get_data(return_freqs=True)

psd_values_db = 10 * np.log10(psd_values)
psd_mean = np.mean(psd_values_db, axis=0)


fit_freq_range = (psd_freqs > 3) & (psd_freqs < 20)  # Adjust the range as needed

psd_mean = psd_mean - np.mean(psd_mean)
# TODO XXX: I'm not sure why this is needed but the fit is better with it
psd_mean = psd_mean + np.mean(psd_mean[fit_freq_range])  

print("Fit data", psd_freqs[fit_freq_range], psd_mean)
popt, _ = curve_fit(one_over_f, psd_freqs[fit_freq_range], psd_mean[fit_freq_range])
alpha, beta = popt

print("Fitted alpha and beta:", alpha, beta)

fitted_curve = one_over_f(psd_freqs[fit_freq_range], alpha, beta)



fig = psd.plot(average=True, show=False)
ax = fig.get_axes()[0]
ax.plot(psd_freqs[fit_freq_range], fitted_curve, label='1/f fit', linewidth=1, color='darkmagenta')
add_red_line_with_value(fig, peak_alpha_freq)

plt.show()
