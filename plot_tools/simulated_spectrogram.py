import numpy as np
from matplotlib import pyplot as plt

from mne import Epochs, create_info
from mne.baseline import rescale
from mne.io import RawArray
from mne.time_frequency import (
    AverageTFR,
    tfr_array_morlet,
    tfr_morlet,
    tfr_multitaper,
    tfr_stockwell,
)
from mne.viz import centers_to_edges

sfreq = 1000.0
ch_names = ["SIM0001", "SIM0002"]
ch_types = ["grad", "grad"]
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

n_times = 1024  # Just over 1 second epochs
n_epochs = 40
seed = 42
rng = np.random.RandomState(seed)
data = rng.randn(len(ch_names), n_times * n_epochs + 200)  # buffer

# Add a 50 Hz sinusoidal burst to the noise and ramp it.
t = np.arange(n_times, dtype=np.float64) / sfreq
signal = np.sin(np.pi * 2.0 * 50.0 * t)  # 50 Hz sinusoid signal
signal[np.logical_or(t < 0.45, t > 0.55)] = 0.0  # hard windowing
on_time = np.logical_and(t >= 0.45, t <= 0.55)
signal[on_time] *= np.hanning(on_time.sum())  # ramping
data[:, 100:-100] += np.tile(signal, n_epochs)  # add signal

raw = RawArray(data, info)
events = np.zeros((n_epochs, 3), dtype=int)
events[:, 0] = np.arange(n_epochs) * n_times
epochs = Epochs(
    raw,
    events,
    dict(sin50hz=0),
    tmin=0,
    tmax=n_times / sfreq,
    reject=dict(grad=4000),
    baseline=None,
)

epochs.average().plot()




freqs = np.arange(5., 100., 3.)
vmin, vmax = -3., 3.  # Define our color limits.


n_cycles = freqs / 2.
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax,
           title='Sim: Least smoothing, most variance')
