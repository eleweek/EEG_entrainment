import sys

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import mne

from specparam import SpectralModel, SpectralGroupModel
from specparam.bands import Bands
from specparam.analysis import get_band_peak_group
from specparam.plts.spectra import plot_spectra


from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels



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

input_filename = sys.argv[1]

raw = load_raw_xdf(input_filename)
filter_and_drop_dead_channels(raw, None)
print(raw.ch_names)

psd = raw.compute_psd(fmin=1.0, fmax=45.0)
psd_values, psd_freqs = psd.get_data(return_freqs=True)

fg = SpectralGroupModel()
print(psd_freqs.shape, psd_values.shape)
fg.report(psd_freqs, psd_values, [3, 45])
fg.plot()

# Plot the topographies across different frequency bands
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_group(fg, band_def)[:, 1])
    print("Band def", band_def)
    print("Band power", band_power.shape, band_power)

    # Create a topomap for the current oscillation band
    mne.viz.plot_topomap(band_power, raw.info, cmap=cm.viridis, contours=0, axes=axes[ind], show=False)

    # Set the plot title
    axes[ind].set_title(label + ' power', {'fontsize' : 20})


fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ind, (label, band_def) in enumerate(bands):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_group(fg, band_def)[:, 1])

    channel_index = np.argmax(band_power)
    # Extracted and plot the power spectrum model with the most band power
    fg.get_model(np.argmax(channel_index)).plot(ax=axes[ind], add_legend=False, linewidth=1.0, data_kwargs={'color' : 'gray'}, model_kwargs={'color' : 'red', 'alpha' : 1.0})

    # Set some plot aesthetics & plot title
    axes[ind].yaxis.set_ticklabels([])
    axes[ind].set_title('biggest ' + label + ' peak ' + raw.ch_names[channel_index], {'fontsize' : 16})


print("Before plt.show()")
plt.show()