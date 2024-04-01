import numpy as np
from scipy.optimize import curve_fit


def get_peak_alpha_freq(psd):
    psds, freqs = psd.get_data(return_freqs=True)
    avg_psd = np.mean(psds, axis=0)
    alpha_range = (freqs >= 8) & (freqs <= 12)
    peak_alpha_freq = freqs[alpha_range][np.argmax(avg_psd[alpha_range])]

    return peak_alpha_freq


# Not the best API, returns 4 values, which is clunky
# TODO: refactor later into several functions?
def fit_one_over_f_curve(psd, min_freq, max_freq, peak_alpha_freq):
    def one_over_f(freq, alpha, beta):
        return alpha / freq ** beta

    psd_values, psd_freqs = psd.get_data(return_freqs=True)
    fit_freq_range = (psd_freqs >= min_freq) & (psd_freqs <= max_freq)

    psd_values_db = 10 * np.log10(psd_values * 1e6 * 1e6) 
    psd_mean = np.mean(psd_values_db, axis=0)


    popt, _ = curve_fit(one_over_f, psd_freqs[fit_freq_range], psd_mean[fit_freq_range])

    alpha, beta = popt
    fitted_curve = one_over_f(psd_freqs[fit_freq_range], alpha, beta)

    delta_db = psd_mean[np.where(psd_freqs == peak_alpha_freq)][0] - one_over_f(peak_alpha_freq, alpha, beta)

    return psd_freqs, fit_freq_range, fitted_curve, delta_db