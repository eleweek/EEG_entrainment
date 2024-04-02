import matplotlib

from libs.psd import fit_one_over_f_curve, get_peak_alpha_freq

def add_red_line_with_value(fig, value, delta_db):
    for ax in fig.axes:
        ax.axvline(x=value, color='red', linestyle='-', linewidth=1.0)

        y_min, y_max = ax.get_ylim()
        y_shift = (y_max - y_min) * 0.05  # Adjust the multiplication factor as needed

        offset = matplotlib.transforms.ScaledTranslation(2/72, 0, fig.dpi_scale_trans)
        text_transform = ax.transData + offset

        ax.text(value, y_max - y_shift, f'{value:.2f} Hz' + f', {delta_db:.2f} dB' if delta_db is not None else '',
                ha='left', va='top', color='red', fontsize=8, transform=text_transform)


def plot_psd(psd, title=None):
    peak_alpha_freq = get_peak_alpha_freq(psd)
    psd_freqs, fit_freq_range, fitted_curve, delta_db = fit_one_over_f_curve(psd, min_freq=3, max_freq=40, peak_alpha_freq=peak_alpha_freq)

    fig = psd.plot(average=True, show=False)
    ax = fig.get_axes()[0]
    ax.plot(psd_freqs[fit_freq_range], fitted_curve, label='1/f fit', linewidth=1, color='darkmagenta')
    
    if title is not None:
        ax.set_title(title)

    add_red_line_with_value(fig, peak_alpha_freq, delta_db)