import sys
import argparse

from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import mne

from specparam import SpectralModel, SpectralGroupModel
from specparam.bands import Bands
from specparam.data.periodic import get_band_peak_group, get_band_peak
from specparam.plts.spectra import plot_spectra


from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.parse import parse_picks



def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data


# Define frequency bands of interest
bands = Bands({'theta': [3, 7],
               'alpha': [7, 14],
               'beta': [15, 30]})

parser = argparse.ArgumentParser(
                    prog='xdf_to_specparam',
                    description='Analyze XDF with specparam and plot band topographies/models')

parser.add_argument('input_xdf_filename', type=str, help='Path to the XDF file')
parser.add_argument('--picks', type=str, default=None, help='Comma or space-separated list of channels to use')

args = parser.parse_args()
input_filename = args.input_xdf_filename
picks = parse_picks(args.picks)

raw = load_raw_xdf(input_filename)
filter_and_drop_dead_channels(raw, picks=picks)
print(raw.ch_names)

psd = raw.compute_psd(fmin=1.0, fmax=45.0)
psd_values, psd_freqs = psd.get_data(return_freqs=True)

fg = SpectralGroupModel()
print(psd_freqs.shape, psd_values.shape)
fg.report(psd_freqs, psd_values, [3, 45])

# Plot the topographies across different frequency bands
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ind, (label, band_def) in enumerate(bands):
    alpha_peaks = get_band_peak_group(fg, bands['alpha'])

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_group(fg, band_def)[:, 1])
    print("Band def", band_def)
    print("Band power", band_power.shape, band_power)

    # Create a topomap for the current oscillation band
    mne.viz.plot_topomap(band_power, raw.info, cmap=cm.viridis, contours=0, axes=axes[ind], show=False)

    # Set the plot title
    axes[ind].set_title(label + ' power', {'fontsize' : 20})

alpha_peaks = get_band_peak_group(fg, bands['alpha'])
print("Best alpha peaks", alpha_peaks)


peaks = np.empty((0, 3))
for i in range(len(fg.get_results())):
    model = fg.get_model(i)
    peaks = np.vstack((peaks, get_band_peak(model, bands['alpha'], select_highest=False)))

print("All alpha peaks", peaks)

num_channels = len(fg.get_results())
if num_channels > 0:
    ncols = int(np.ceil(np.sqrt(num_channels)))
    nrows = int(np.ceil(num_channels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.5))
    axes = np.array(axes).reshape(-1)

    used_axes = []
    for channel_index in range(num_channels):
        ax = axes[channel_index]
        fg.get_model(ind=channel_index, regenerate=True).plot(
            ax=ax,
            linewidth=0.9,
            data_kwargs={'color': 'gray', 'alpha': 0.4},
            model_kwargs={'color': 'red', 'alpha': 0.9},
            aperiodic_kwargs={'color': 'blue', 'alpha': 0.9, 'linestyle': '--'}
        )
        ax.set_title(raw.ch_names[channel_index], {'fontsize': 10})
        used_axes.append(ax)

    # Hide any unused axes
    for ax in axes[num_channels:]:
        ax.set_visible(False)

    # Harmonize y-limits across used subplots
    min_y_lim = np.min([ax.get_ylim()[0] for ax in used_axes])
    max_y_lim = np.max([ax.get_ylim()[1] for ax in used_axes])
    for ax in used_axes:
        ax.set_ylim(min_y_lim, max_y_lim)
        ax.grid(False)
        # Ensure no per-axis legend remains
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Add a single, shared legend for all subplots
    legend_elements = [
        Line2D([0], [0], color='gray', alpha=0.4, lw=0.9, label='Data'),
        Line2D([0], [0], color='red', alpha=0.9, lw=0.9, label='Model'),
        Line2D([0], [0], color='blue', alpha=0.9, lw=0.9, linestyle='--', label='Aperiodic fit'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    # Make space at the top for the shared legend
    fig.tight_layout(rect=(0, 0, 1, 0.95))



print("Before plt.show()")
plt.show()