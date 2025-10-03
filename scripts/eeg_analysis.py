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


def compute_alpha_envelope(raw: mne.io.BaseRaw, fmin: float, fmax: float, smooth_sec: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute alpha-band amplitude envelope.

    Returns (times, envelope_avg) where envelope_avg is averaged across channels.
    """
    sfreq = float(raw.info['sfreq'])
    raw_alpha = raw.copy().filter(l_freq=fmin, h_freq=fmax, verbose=False)

    data_alpha = raw_alpha.get_data()
    # Analytic amplitude via Hilbert transform
    analytic = hilbert(data_alpha, axis=1)
    envelope = np.abs(analytic)

    if smooth_sec and smooth_sec > 0:
        win = max(1, int(round(sfreq * smooth_sec)))
        if win > 1:
            kernel = np.ones(win, dtype=float) / float(win)
            envelope = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=envelope)

    envelope_avg = envelope.mean(axis=0)
    return raw_alpha.times.copy(), envelope_avg


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
) -> Optional[Tuple[Dict[str, mne.Evoked], mne.Epochs]]:
    """Create ERPs for the requested annotation descriptions.

    Returns dict of evokeds and the epochs container.
    """
    # Slightly tailor filtering for ERP (optional low-pass)
    raw_erp = raw.copy().filter(l_freq=None, h_freq=30.0, verbose=False)

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


def plot_alpha_envelope(times: np.ndarray, envelope_avg: np.ndarray, title: str):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(times, envelope_avg, color='tab:blue', lw=1.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Alpha envelope (a.u.)')
    ax.set_title(title)
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
    ap.add_argument('--alpha-low', type=float, default=8.0, help='Alpha band lower freq (Hz)')
    ap.add_argument('--alpha-high', type=float, default=12.0, help='Alpha band upper freq (Hz)')
    ap.add_argument('--smooth-sec', type=float, default=0.2, help='Smoothing window for envelope (seconds)')

    # ERP params
    ap.add_argument('--erp-events', type=str, default='stim_flip_done,response', help='Comma-separated annotation descriptions to epoch')
    ap.add_argument('--tmin', type=float, default=-0.2, help='ERP epoch start (s)')
    ap.add_argument('--tmax', type=float, default=0.8, help='ERP epoch end (s)')
    ap.add_argument('--baseline', type=str, default='-0.2,0.0', help='ERP baseline tuple start,end in seconds or "none"')

    ap.add_argument('--save-prefix', type=str, default=None, help='If set, save figures with this prefix instead of showing')

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

    # Alpha envelope
    times, env_avg = compute_alpha_envelope(raw, args.alpha_low, args.alpha_high, args.smooth_sec)
    fig_env = plot_alpha_envelope(times, env_avg, title=f"Alpha envelope {args.alpha_low:.1f}-{args.alpha_high:.1f} Hz")

    # ERP
    baseline = None if args.baseline.lower() == 'none' else tuple(float(x) for x in args.baseline.split(','))  # type: ignore
    erp_out = run_erp_analysis(raw, erp_event_names, tmin=args.tmin, tmax=args.tmax, baseline=baseline)  # type: ignore
    fig_erp = None
    if erp_out is not None:
        evokeds, _epochs = erp_out
        fig_erp = plot_erps(evokeds)

    if args.save_prefix:
        if fig_env is not None:
            fig_env.savefig(f"{args.save_prefix}_alpha_envelope.png", dpi=150, bbox_inches='tight')
        if fig_erp is not None:
            fig_erp.savefig(f"{args.save_prefix}_erps.png", dpi=150, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main())


