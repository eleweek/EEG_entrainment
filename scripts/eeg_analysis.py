from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import hilbert

from libs.file_formats import load_recording
from libs.filters import filter_and_drop_dead_channels
from libs.parse import parse_picks


def compute_alpha_envelope_epoched(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    iaf: float,
    bandwidth: float = 1.0,
    tmin: float = -3.0,
    tmax: float = 3.0,
    baseline: Tuple[float, float] = (-2.8, -2.7),
    *,
    average_channels: bool = True,
) -> Tuple[np.ndarray, np.ndarray, mne.Epochs]:
    """Compute alpha-band amplitude envelope with proper per-epoch z-scoring.
    
    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data
    events : np.ndarray
        Events array from mne.events_from_annotations
    event_id : dict
        Event ID mapping
    iaf : float
        Individual Alpha Frequency (Hz)
    bandwidth : float
        Filter bandwidth around IAF (e.g., 1.0 for IAF±0.5 Hz)
    tmin, tmax : float
        Epoch time limits (seconds)
    baseline : tuple
        Baseline window for z-scoring (start, end) in seconds
    average_channels : bool
        If True, average channels before envelope extraction
        
    Returns
    -------
    times : np.ndarray
        Time vector for epochs
    envelope_avg : np.ndarray
        Average z-scored envelope across epochs
    epochs : mne.Epochs
        Epochs object (for further analysis)
    """
    # 1. Create epochs
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,  # Don't baseline correct raw data
        preload=True,
        detrend=None,
        verbose=False,
    )
    
    # 2. Filter at IAF ± bandwidth/2
    fmin = iaf - bandwidth / 2.0
    fmax = iaf + bandwidth / 2.0
    epochs_filtered = epochs.copy().filter(l_freq=fmin, h_freq=fmax, verbose=False)
    
    # 3. Get data: (n_epochs, n_channels, n_times)
    data = epochs_filtered.get_data()
    
    # Average channels first if requested
    if average_channels:
        data = data.mean(axis=1, keepdims=True)  # (n_epochs, 1, n_times)
    
    # 4. Hilbert transform per epoch and channel
    analytic = hilbert(data, axis=-1)  # Apply along time axis
    envelope = np.abs(analytic)  # (n_epochs, n_channels, n_times)
    
    # Average across channels if not done before
    if not average_channels:
        envelope = envelope.mean(axis=1, keepdims=True)  # (n_epochs, 1, n_times)
    
    # Now envelope is (n_epochs, 1, n_times)
    envelope = envelope.squeeze(axis=1)  # (n_epochs, n_times)
    
    # 5. Per-epoch z-scoring relative to baseline window
    times = epochs.times
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    
    envelope_z = np.zeros_like(envelope)
    for i in range(envelope.shape[0]):
        baseline_data = envelope[i, baseline_mask]
        baseline_mean = baseline_data.mean()
        baseline_std = baseline_data.std(ddof=0)
        
        if baseline_std > 0:
            envelope_z[i, :] = (envelope[i, :] - baseline_mean) / baseline_std
        else:
            envelope_z[i, :] = envelope[i, :] - baseline_mean
    
    # 6. Average across epochs
    envelope_avg = envelope_z.mean(axis=0)  # (n_times,)
    
    return times, envelope_avg, epochs



def build_event_id_from_annotations(raw: mne.io.BaseRaw, keep_events: List[str]) -> Dict[str, int]:
    """Map annotation descriptions to integer event IDs, filtered by keep_events order."""
    if not raw.annotations:
        return {}
    present = {ann['description'] for ann in raw.annotations}
    selected = [ev for ev in keep_events if ev in present]
    return {ev: idx + 1 for idx, ev in enumerate(selected)}


def run_erp_analysis(
    raw: mne.io.BaseRaw,
    event_names: List[str],
    tmin: float,
    tmax: float,
    baseline: Optional[Tuple[Optional[float], Optional[float]]],
    average_channels: bool = True,
) -> Optional[Tuple[Dict[str, mne.Evoked], mne.Epochs]]:
    """Create ERPs for the requested annotation descriptions."""

    raw_erp = raw.copy().filter(l_freq=None, h_freq=30.0, verbose=False)

    if average_channels:
        data = raw_erp.get_data()
        avg = data.mean(axis=0, keepdims=True)
        sfreq = raw_erp.info['sfreq']
        info_avg = mne.create_info(['posterior_avg'], sfreq, ch_types=['eeg'], verbose=False)
        raw_erp = mne.io.RawArray(avg, info_avg, verbose=False)
        raw_erp.set_annotations(raw.annotations)

    event_id = build_event_id_from_annotations(raw_erp, event_names)
    if not event_id:
        print("No requested ERP events found in annotations; skipping ERP.")
        return None

    events, _ = mne.events_from_annotations(raw_erp, event_id=event_id, verbose=False)
    if events.size == 0:
        print("No events extracted from annotations; skipping ERP.")
        return None

    epochs = mne.Epochs(
        raw_erp,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        detrend=1,
        verbose=False,
    )

    evokeds = {name: epochs[name].average() for name in event_id.keys()}
    return evokeds, epochs


def plot_alpha_envelope(times: np.ndarray, envelope_avg: np.ndarray, title: str, 
                        mark_windows: Optional[List[Tuple[float, float, str]]] = None):
    """Plot alpha envelope with optional time window markers."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(times, envelope_avg, color='tab:blue', lw=1.5, label='Alpha envelope (z-scored)')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Mark time windows of interest
    if mark_windows:
        for tmin, tmax, label in mark_windows:
            ax.axvspan(tmin, tmax, alpha=0.2, label=label)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z-scored amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, right=False)
    fig.tight_layout()
    return fig


def plot_erps(evokeds: Dict[str, mne.Evoked]):
    if not evokeds:
        return None
    n = len(evokeds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for ax, (name, ev) in zip(axes[0], evokeds.items()):
        ev.plot(axes=ax, show=False, spatial_colors=True, time_unit='s')
        ax.set_title(f"ERP: {name}")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, right=False)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="EEG alpha envelope and ERP analysis with plots")
    ap.add_argument('input_file', type=str, help='Path to EEG recording (e.g., .xdf)')
    ap.add_argument('--picks', type=str, default=None, help='Comma/space-separated channel names to include')

    # Alpha envelope params
    ap.add_argument('--iaf', type=float, required=True, help='Individual Alpha Frequency (Hz) - REQUIRED')
    ap.add_argument('--alpha-bandwidth', type=float, default=1.0, help='Bandwidth around IAF (Hz), e.g., 1.0 for IAF±0.5')
    ap.add_argument('--epoch-event', type=str, default='stim_flip_done', help='Annotation description to epoch on')
    ap.add_argument('--epoch-tmin', type=float, default=-3.0, help='Epoch start (s) relative to event')
    ap.add_argument('--epoch-tmax', type=float, default=3.0, help='Epoch end (s) relative to event')
    ap.add_argument('--baseline-start', type=float, default=-2.8, help='Baseline window start (s)')
    ap.add_argument('--baseline-end', type=float, default=-2.7, help='Baseline window end (s)')
    
    # Mark time windows on plot
    ap.add_argument('--mark-entrainment', type=str, default='-1.0,-0.5', 
                    help='Entrainment window to mark (start,end in seconds)')
    ap.add_argument('--mark-poststim', type=str, default='0.4,0.6',
                    help='Post-stimulus window to mark (start,end in seconds)')

    # ERP params
    ap.add_argument('--erp-events', type=str, default='stim_flip_done,response', 
                    help='Comma-separated annotation descriptions to epoch for ERP')
    ap.add_argument('--erp-tmin', type=float, default=-0.2, help='ERP epoch start (s)')
    ap.add_argument('--erp-tmax', type=float, default=0.8, help='ERP epoch end (s)')
    ap.add_argument('--erp-baseline', type=str, default='-0.2,0.0', 
                    help='ERP baseline tuple start,end in seconds or "none"')

    ap.add_argument('--save-prefix', type=str, default=None, 
                    help='If set, save figures with this prefix instead of showing')
    
    # Legacy mode for backward compatibility
    ap.add_argument('--use-continuous', action='store_true',
                    help='Use old continuous envelope method (not recommended)')

    args = ap.parse_args()

    picks = parse_picks(args.picks)
    erp_event_names = [s.strip() for s in args.erp_events.split(',') if s.strip()]

    # Load and basic filter
    raw = load_recording(args.input_file)
    filter_and_drop_dead_channels(raw, picks)

    # Report annotations summary
    anns = raw.annotations if raw.annotations is not None else None
    n_anns = (len(anns) if anns is not None else 0)
    print(f"Loaded annotations: {n_anns}")
    if n_anns > 0:
        head = min(5, n_anns)
        for i in range(head):
            print(
                f"  {i+1}) t={anns.onset[i]:.3f}s dur={anns.duration[i]:.3f}s desc={anns.description[i]}"
            )

    # Alpha envelope (proper epoched method)

    # Get events for epoching
    event_id = build_event_id_from_annotations(raw, [args.epoch_event])
    if not event_id:
        print(f"ERROR: Event '{args.epoch_event}' not found in annotations!")
        return 1
    
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    if events.size == 0:
        print(f"ERROR: No events extracted for '{args.epoch_event}'!")
        return 1
    
    print(f"Found {len(events)} epochs for event '{args.epoch_event}'")
    print(f"Computing envelope at IAF={args.iaf:.1f} Hz (±{args.alpha_bandwidth/2:.1f} Hz)")
    
    times, env_avg, epochs = compute_alpha_envelope_epoched(
        raw,
        events,
        event_id,
        iaf=args.iaf,
        bandwidth=args.alpha_bandwidth,
        tmin=args.epoch_tmin,
        tmax=args.epoch_tmax,
        baseline=(args.baseline_start, args.baseline_end),
        average_channels=True,
    )
    
    # Parse time windows to mark
    mark_windows = []
    if args.mark_entrainment:
        t1, t2 = map(float, args.mark_entrainment.split(','))
        mark_windows.append((t1, t2, 'Entrainment'))
    if args.mark_poststim:
        t1, t2 = map(float, args.mark_poststim.split(','))
        mark_windows.append((t1, t2, 'Post-stimulus'))
    
    # Report mean amplitude in windows
    if mark_windows:
        print("\nMean z-scored amplitude in time windows:")
        for tmin, tmax, label in mark_windows:
            mask = (times >= tmin) & (times <= tmax)
            mean_amp = env_avg[mask].mean()
            print(f"  {label} [{tmin:.2f}, {tmax:.2f}]s: {mean_amp:.3f}")
    
    fig_env = plot_alpha_envelope(
        times, env_avg, 
        title=f"Alpha envelope at IAF={args.iaf:.1f} Hz (z-scored per epoch)",
        mark_windows=mark_windows
    )


    # ERP analysis
    baseline = None if args.erp_baseline.lower() == 'none' else tuple(float(x) for x in args.erp_baseline.split(','))  # type: ignore
    erp_out = run_erp_analysis(raw, erp_event_names, tmin=args.erp_tmin, tmax=args.erp_tmax, baseline=baseline)  # type: ignore
    fig_erp = None
    if erp_out is not None:
        evokeds, _epochs = erp_out
        fig_erp = plot_erps(evokeds)

    # Save or show
    if args.save_prefix:
        if fig_env is not None:
            fig_env.savefig(f"{args.save_prefix}_alpha_envelope.png", dpi=150, bbox_inches='tight')
            print(f"Saved: {args.save_prefix}_alpha_envelope.png")
        if fig_erp is not None:
            fig_erp.savefig(f"{args.save_prefix}_erps.png", dpi=150, bbox_inches='tight')
            print(f"Saved: {args.save_prefix}_erps.png")
        plt.close('all')
    else:
        plt.show()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())