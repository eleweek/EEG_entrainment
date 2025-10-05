"""EEG Quality Control for Specparam Analysis"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from specparam import SpectralGroupModel
from specparam.bands import Bands
from specparam.data.periodic import get_band_peak


@dataclass
class ChannelQC:
    """QC results for a single channel."""
    channel: str
    passed: bool
    issues: List[str]
    
    # Fit quality
    r_squared: float
    error: float
    
    # Aperiodic
    exponent: float
    offset: float
    
    # Alpha peak
    has_alpha: bool
    alpha_cf: Optional[float]
    alpha_pw: Optional[float]
    alpha_bw: Optional[float]


@dataclass
class SessionQC:
    """Overall session QC results."""
    passed: bool
    channels: Dict[str, ChannelQC]
    
    # Posterior consistency
    posterior_iaf_std: Optional[float] 
    posterior_alpha_present: int  # How many of O1/Oz/O2 have alpha
    
    summary: List[str]
    warnings: List[str]


def check_channel_qc(
    fg: SpectralGroupModel,
    channel_idx: int,
    channel_name: str,
    alpha_band: Tuple[float, float] = (7.0, 14.0),
    # The exponent range is a bit wider range of plausible ranges from:
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC8800045/
    exponent_range: Tuple[float, float] = (0.7, 2.75),
    min_r_squared: float = 0.90,
    min_alpha_snr: float = 1.5,
) -> ChannelQC:
    """Run QC checks on a single channel.
    
    Parameters
    ----------
    fg : SpectralGroupModel
        Fitted group model
    channel_idx : int
        Index of channel in group model
    channel_name : str
        Name of channel
    alpha_band : tuple
        (min, max) Hz for alpha band
    exponent_range : tuple
        Acceptable (min, max) for aperiodic exponent
    min_r_squared : float
        Minimum acceptable R² for model fit
    min_alpha_snr : float
        Minimum SNR for alpha peak (peak_power / median_band_power)
    
    Returns
    -------
    ChannelQC
        QC results for this channel
    """
    issues = []
    
    model = fg.get_model(ind=channel_idx)
    
    # Extract fit quality
    r_squared = model.r_squared_
    error = model.error_
    
    # Extract aperiodic params
    ap_params = model.aperiodic_params_
    if len(ap_params) == 2:
        offset, exponent = ap_params
    elif len(ap_params) == 3:
        offset, _, exponent = ap_params
    else:
        offset, exponent = np.nan, np.nan
        issues.append("Invalid aperiodic params")
    
    # Check aperiodic physiological range
    if not (exponent_range[0] <= exponent <= exponent_range[1]):
        issues.append(f"Exponent out of range: {exponent:.2f} (expect {exponent_range})")
    
    # offset < -12: Signal too weak (bad contact, disconnected electrode)
    # offset > -6: Signal too strong (artifact, saturation, wrong units)
    if offset < -12 or offset > -6:
        issues.append(f"Offset out of range: {offset:.2f}")
    
    # Check fit quality
    if r_squared < min_r_squared:
        issues.append(f"Poor fit quality: R²={r_squared:.3f} (expect >{min_r_squared})")
    
    has_alpha = False
    alpha_cf, alpha_pw, alpha_bw = None, None, None

    # Use specparam's utility to extract alpha peak
    alpha_peak = get_band_peak(model, alpha_band, select_highest=True)

    if alpha_peak.size > 0:
        alpha_cf, alpha_pw, alpha_bw = alpha_peak
        has_alpha = True
        
        # Check peak quality using PW directly (already in log10 units above baseline)
        if alpha_pw < 0.3:
            issues.append(f"Weak alpha peak: PW={alpha_pw:.2f} (expect >0.3)")
    else:
        # Distinguish between "no peaks at all" vs "peaks but not in alpha band"
        if model.peak_params_.size == 0:
            issues.append("No peaks detected at all")
        else:
            issues.append("No alpha peak detected")
    
    passed = len(issues) == 0
    
    return ChannelQC(
        channel=channel_name,
        passed=passed,
        issues=issues,
        r_squared=r_squared,
        error=error,
        exponent=exponent,
        offset=offset,
        has_alpha=has_alpha,
        alpha_cf=alpha_cf,
        alpha_pw=alpha_pw,
        alpha_bw=alpha_bw,
    )


def check_session_qc(
    fg: SpectralGroupModel,
    channel_names: List[str],
    posterior_channels: List[str] = ['O1', 'Oz', 'O2'],
    max_posterior_iaf_std: float = 0.5,
) -> SessionQC:
    """Run QC checks on entire session.
    
    Parameters
    ----------
    fg : SpectralGroupModel
        Fitted group model
    channel_names : list
        Names of all channels in order
    posterior_channels : list
        Names of posterior channels (should have strong alpha)
    reference_channel : str or None
        Name of reference channel (should have weak alpha)
    max_posterior_iaf_std : float
        Maximum acceptable std dev of IAF across posterior channels (Hz)
    max_reference_contamination : float
        Maximum ratio of reference_alpha / posterior_alpha
    
    Returns
    -------
    SessionQC
        Overall QC results
    """
    # Run per-channel QC
    channel_qc = {}
    for idx, name in enumerate(channel_names):
        channel_qc[name] = check_channel_qc(fg, idx, name)
    
    summary = []
    warnings = []
    
    # Check posterior consistency
    posterior_iaf_std = None
    posterior_alpha_present = 0
    posterior_iafs = []
    
    for ch in posterior_channels:
        if ch in channel_qc:
            qc = channel_qc[ch]
            if qc.has_alpha:
                posterior_alpha_present += 1
                posterior_iafs.append(qc.alpha_cf)
    
    if len(posterior_iafs) >= 2:
        posterior_iaf_std = np.std(posterior_iafs)
        if posterior_iaf_std > max_posterior_iaf_std:
            warnings.append(
                f"Posterior IAF inconsistent: std={posterior_iaf_std:.2f} Hz "
                f"(IAFs: {', '.join(f'{x:.2f}' for x in posterior_iafs)})"
            )
    
    if posterior_alpha_present < len([ch for ch in posterior_channels if ch in channel_names]):
        summary.append(
            f"Only {posterior_alpha_present}/{len(posterior_channels)} "
            f"posterior channels have alpha peak"
        )
    
    # Overall pass/fail
    failed_channels = [name for name, qc in channel_qc.items() if not qc.passed]
    passed = len(failed_channels) == 0 and len(summary) == 0
    
    if failed_channels:
        summary.insert(0, f"Failed channels: {', '.join(failed_channels)}")
    
    return SessionQC(
        passed=passed,
        channels=channel_qc,
        posterior_iaf_std=posterior_iaf_std,
        posterior_alpha_present=posterior_alpha_present,
        summary=summary,
        warnings=warnings
    )


def print_qc_report(qc: SessionQC):
    """Pretty-print QC report to console."""
    
    print("\n" + "="*60)
    print("EEG QUALITY CONTROL REPORT")
    print("="*60)
    
    # Overall status
    status = "✓ PASS" if qc.passed else "✗ FAIL"
    print(f"\nOverall Status: {status}\n")
    
    # Summary issues
    if qc.summary:
        print("Summary Issues:")
        for issue in qc.summary:
            print(f"  • {issue}")
        print()
    
    # Posterior consistency
    if qc.posterior_iaf_std is not None:
        print(f"Posterior IAF consistency: {qc.posterior_iaf_std:.2f} Hz std dev")
    print(f"Posterior channels with alpha: {qc.posterior_alpha_present}")
    
    # Per-channel details
    print("\n" + "-"*60)
    print("Per-Channel Details:")
    print("-"*60)
    
    for name, ch_qc in qc.channels.items():
        status_symbol = "✓" if ch_qc.passed else "✗"
        print(f"\n{status_symbol} {name}:")
        print(f"  R² = {ch_qc.r_squared:.3f}, Error = {ch_qc.error:.3f}")
        print(f"  Aperiodic: offset={ch_qc.offset:.2f}, exponent={ch_qc.exponent:.2f}")
        
        if ch_qc.has_alpha:
            print(f"  Alpha: CF={ch_qc.alpha_cf:.2f} Hz, "
                  f"PW={ch_qc.alpha_pw:.2f}, "
                  f"BW={ch_qc.alpha_bw:.2f}, ")
        else:
            print(f"  Alpha: None detected")
        
        if ch_qc.issues:
            print(f"  Issues:")
            for issue in ch_qc.issues:
                print(f"    - {issue}")
    
    print("\n" + "="*60 + "\n")