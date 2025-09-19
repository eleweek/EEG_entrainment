import pygame

from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt

from mne.viz import plot_sensors

from libs.psd import fit_one_over_f_curve, get_peak_alpha_freq

def add_red_line_with_value(fig, value, delta_db):
    ax = fig.get_axes()[0]

    ax.axvline(x=value, color='red', linestyle='-', linewidth=1.0)

    y_min, y_max = ax.get_ylim()
    y_shift = (y_max - y_min) * 0.05  # Adjust the multiplication factor as needed

    if delta_db is not None:
        text = f'{value:.2f} Hz, {delta_db:.2f} dB'
    else:
        text = f'{value:.2f} Hz'

    offset = matplotlib.transforms.ScaledTranslation(2/72, 0, fig.dpi_scale_trans)
    text_transform = ax.transData + offset

    ax.text(value, y_max - y_shift, text,
            ha='left', va='top', color='red', fontsize=8, transform=text_transform)


def plot_psd(psd, title=None, average=True, ylim=None):
    # TODO: do our own custom mapping of electrodes to colors
    COLOR_VALUES = ["brown", "red", "orange", "magenta", "green", "blue", "purple", "black"]

    peak_alpha_freq = get_peak_alpha_freq(psd)
    psd_freqs, fit_freq_range, fitted_curve, delta_db = fit_one_over_f_curve(psd, min_freq=3, max_freq=40, peak_alpha_freq=peak_alpha_freq)

    fig = psd.plot(average=average, show=False, spatial_colors=True)
    # Remove borders/spines around all axes in the figure
    for ax in fig.get_axes():
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, right=False)

    axes = fig.get_axes()
    main_ax = axes[0]
    
    if title is not None:
        main_ax.set_title(title)
    
    if ylim is not None:
        main_ax.set_ylim(*ylim)

    if not average and len(axes) != 2:
        raise Exception(f'Expected 2 axes, got {len(axes)}')
    
    if not average:
        sensor_ax = axes[1]

        lines = main_ax.get_lines()
        for i, line in enumerate(lines):
            if 1 < i <= len(psd.ch_names) + 1: 
                line.set_color(COLOR_VALUES[i - 2])
    

        scatter_collection = sensor_ax.collections[0]
        scatter_collection.set_facecolors(COLOR_VALUES)
        scatter_collection.set_edgecolors(COLOR_VALUES)

    
    main_ax.plot(psd_freqs[fit_freq_range], fitted_curve, label='1/f fit', linewidth=1, color='darkmagenta')
    add_red_line_with_value(fig, peak_alpha_freq, delta_db)

    return fig, namedtuple('PSDData', ['peak_alpha_freq', 'delta_db'])(peak_alpha_freq, delta_db)


def plot_to_pygame(agg, fig):
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "RGB")