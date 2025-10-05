import sys
import argparse

from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import find_peaks

from specparam import SpectralModel, SpectralGroupModel
from specparam.bands import Bands
from specparam.data.periodic import get_band_peak_group, get_band_peak
from specparam.plts.spectra import plot_spectra
from specparam.sim.gen import gen_periodic


from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.parse import parse_picks
from libs.eeg_qc import check_session_qc, print_qc_report

MAX_N_PEAKS=8

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
parser.add_argument('--avgref', action='store_true', help='Re-reference EEG to average after filtering')
parser.add_argument('--to-drop', type=str, default=None, help='Comma or space-separated list of channels to drop before filtering')

args = parser.parse_args()
input_filename = args.input_xdf_filename
picks = parse_picks(args.picks)
to_drop = parse_picks(args.to_drop)

raw = load_raw_xdf(input_filename)
filter_and_drop_dead_channels(raw, picks=picks, to_drop=to_drop, avgref=args.avgref)
print(raw.ch_names)

psd = raw.compute_psd(fmin=1.0, fmax=40.0, n_fft=int(raw.info['sfreq'] * 10))
psd_values, psd_freqs = psd.get_data(return_freqs=True)

fg = SpectralGroupModel(max_n_peaks=MAX_N_PEAKS)
print(psd_freqs.shape, psd_values.shape)
fg.report(psd_freqs, psd_values, [3, 40])
print("\n" + "="*60)
print("RUNNING QUALITY CONTROL")
print("="*60)

qc_results = check_session_qc(
    fg,
    channel_names=raw.ch_names,
    posterior_channels=['O1', 'Oz', 'O2'],
    max_posterior_iaf_std=0.5,
)

print_qc_report(qc_results)

# Exit early if critical QC failures
if not qc_results.passed:
    print("\n⚠️  WARNING: QC checks failed. Review issues above.")
    print("Consider excluding this recording or channels from analysis.\n")
    # Uncomment to exit on failure:
    # sys.exit(1)

# Compute IAF by subtracting each channel's aperiodic model, averaging residuals, and finding peak
def _aperiodic_log10_from_params(freqs, params):
    # params can be [offset, exp] or [offset, knee, exp]
    if len(params) == 2:
        offset, exponent = params
        return offset - exponent * np.log10(freqs)
    elif len(params) == 3:
        offset, knee, exponent = params
        return offset - np.log10(knee + np.power(freqs, exponent))
    else:
        raise ValueError('Unexpected aperiodic params length: {}'.format(len(params)))

def detect_residual_peaks(freqs,
                          residual_log10,
                          peak_width_limits=(0.5, 12.0),
                          max_n_peaks=np.inf,
                          min_peak_height=0.0,
                          peak_threshold=1.5,
                          verbose=False):
    """Fit peaks directly on pre-computed log10 residuals (aperiodic-subtracted).

    Returns
    -------
    peak_params : ndarray, shape (n_peaks, 3)
        Peak parameters as [CF, PW, BW] for each detected peak.
    peak_fit : ndarray, shape (n_freqs,)
        The modeled periodic fit (sum of Gaussians) over `freqs` in log10 units.
    """

    # Initialize a model to leverage specparam's internal peak fitting
    sm = SpectralModel(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        verbose=verbose,
    )

    # Add dummy linear-power data to establish frequency context
    dummy_power_linear = np.ones_like(freqs)
    sm.add_data(freqs, dummy_power_linear)

    # Provide the residuals (already in log10 space) for peak fitting
    sm._spectrum_flat = residual_log10.copy()
    sm.freq_range = [freqs[0], freqs[-1]]
    if len(freqs) > 1:
        sm.freq_res = freqs[1] - freqs[0]

    # Run internal peak fitting on the residuals
    gaussian_params = sm._fit_peaks(sm._spectrum_flat.copy())
    if gaussian_params.size == 0:
        return np.empty((0, 3)), np.zeros_like(freqs)

    # Construct modeled spectrum for peak param conversion; baseline is zero in residual space
    peak_fit = gen_periodic(freqs, gaussian_params.flatten())
    sm.modeled_spectrum_ = peak_fit
    sm._ap_fit = np.zeros_like(freqs)

    peak_params = sm._create_peak_params(gaussian_params)
    return peak_params, peak_fit

try:
    # Safeguard against zeros before log10
    log_psd_values = np.log10(np.maximum(psd_values, 1e-30))

    residuals = []
    for ch_idx in range(len(fg.get_results())):
        model = fg.get_model(ind=ch_idx)
        ap_params = model.get_params('aperiodic_params')
        ap_log = _aperiodic_log10_from_params(psd_freqs, ap_params)
        residuals.append(log_psd_values[ch_idx] - ap_log)

    residuals = np.vstack(residuals) if len(residuals) > 0 else np.empty_like(psd_values)
    mean_residual = np.nanmean(residuals, axis=0) if residuals.size else None

    iaf_hz = None
    if mean_residual is not None:
        alpha_mask = (psd_freqs >= 7.0) & (psd_freqs <= 14.0)
        if np.any(alpha_mask):
            peak_idx = np.argmax(mean_residual[alpha_mask])
            iaf_hz = psd_freqs[alpha_mask][peak_idx]
            iaf_residual_log10 = float(mean_residual[alpha_mask][peak_idx])

            iaf_residual_db = 10.0 * iaf_residual_log10
            print("IAF (aperiodic-adjusted residual peak): {:.3f} Hz @ {:.2f} dB".format(float(iaf_hz), iaf_residual_db))

            # Also print all alpha peaks detected on the averaged residuals
            residual_peak_params, residual_peak_fit = detect_residual_peaks(
                psd_freqs, mean_residual, max_n_peaks=MAX_N_PEAKS, peak_threshold=1.5, min_peak_height=0.0, verbose=False
            )
            if residual_peak_params.size:
                alpha_residual_peaks = residual_peak_params[
                    (residual_peak_params[:, 0] >= 7.0) & (residual_peak_params[:, 0] <= 14.0)
                ]
                formatted = np.array2string(
                    alpha_residual_peaks,
                    formatter={'float_kind': lambda x: f"{float(x):.4f}"}
                )
                print("Alpha peaks on residuals [CF, PW, BW]:", formatted)
            else:
                print("Alpha peaks on residuals [CF, PW, BW]: []")
        else:
            print("Warning: alpha band mask returned empty range; cannot compute IAF.")
    else:
        print("Warning: no residuals computed; cannot compute IAF.")
except Exception as e:
    print("Failed to compute IAF from aperiodic-subtracted residuals:", e)

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
    # Reserve one extra axis for the aperiodic-adjusted average, if available
    num_plots = num_channels + (1 if ('mean_residual' in locals() and mean_residual is not None) else 0)
    ncols = int(np.ceil(np.sqrt(num_plots)))
    nrows = int(np.ceil(num_plots / ncols))

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

    # Plot aperiodic-adjusted average residual spectrum on the next axis, if computed
    avg_ax = None
    if 'mean_residual' in locals() and mean_residual is not None:
        if num_channels < len(axes):
            avg_ax = axes[num_channels]
            avg_ax.plot(psd_freqs, mean_residual, color='green', linewidth=1.0)
            avg_ax.axvspan(7.0, 14.0, color='lightgray', alpha=0.12, zorder=0)
            # Highlight the typical alpha band 8–12 Hz with a darker shade
            avg_ax.axvspan(8.0, 12.0, color='gray', alpha=0.25, zorder=1)
            if 'iaf_hz' in locals() and iaf_hz is not None:
                avg_ax.axvline(iaf_hz, color='green', linestyle=':', linewidth=1.0)
            # Overlay the modeled periodic fit on the residuals
            if 'residual_peak_fit' in locals() and residual_peak_fit is not None:
                # Only draw within alpha band region for clarity
                alpha_mask = (psd_freqs >= 7.0) & (psd_freqs <= 14.0)
                avg_ax.plot(psd_freqs[alpha_mask], residual_peak_fit[alpha_mask],
                            color='orange', linewidth=1.2, alpha=0.9, label='Residual fit')
            title = 'aperiodic-adjusted average'
            if 'iaf_hz' in locals() and iaf_hz is not None:
                title = f'aperiodic-adjusted average (IAF={iaf_hz:.2f} Hz)'
            avg_ax.set_title(title, {'fontsize': 10})
            avg_ax.grid(False)

    # Hide any unused axes
    end_used = num_channels + (1 if avg_ax is not None else 0)
    for ax in axes[end_used:]:
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
plt.tight_layout()
plt.show()