import numpy as np

def get_peak_alpha_freq(psd):
    psds, freqs = psd.get_data(return_freqs=True)
    avg_psd = np.mean(psds, axis=0)
    alpha_range = (freqs >= 8) & (freqs <= 12)
    peak_alpha_freq = freqs[alpha_range][np.argmax(avg_psd[alpha_range])]

    return peak_alpha_freq