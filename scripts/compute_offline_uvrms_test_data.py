import os
import numpy as np
import mne
from libs.file_formats import load_openbci_txt

def real_uvrms(data):
    """Compute the real root mean square."""
    return np.sqrt(np.mean(data ** 2, axis=1)) * 1e6

def fake_uvrms(data):
    """Compute the 'fake RMS', which is the standard deviation."""
    return np.std(data, axis=1) * 1e6

script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_dir = os.path.join(os.path.dirname(script_dir), 'sample_data')
sample_file_path = os.path.join(sample_data_dir, 'sample1-openbci-gui.txt')

# Your loaded Raw object
# raw = mne.io.read_raw_bdf('mydataexample1.bdf', preload=True, exclude=['Accel X', 'Accel Y', 'Accel Z'])
raw = load_openbci_txt(sample_file_path)
raw.filter(l_freq=1.0, h_freq=45.0)

# Split the continuous raw data into 1-second epochs (chunks)
sfreq = raw.info['sfreq']  # Sampling frequency
print("Sampling frequency", sfreq)
print("Channels", raw.ch_names)
epochs_length = 1  # seconds
events = mne.make_fixed_length_events(raw, duration=epochs_length)
epochs = mne.Epochs(raw, events, tmin=0, tmax=epochs_length - 1 / sfreq, baseline=None, preload=True)

# Preallocate space for the RMS values
real_rms_values = np.zeros((len(epochs), len(raw.ch_names)))
fake_rms_values = np.zeros((len(epochs), len(raw.ch_names)))

# Compute real and fake RMS for each epoch and channel
for i, epoch in enumerate(epochs.get_data()):
    real_rms_values[i, :] = real_uvrms(epoch)
    fake_rms_values[i, :] = fake_uvrms(epoch)

print("Real RMS values (shape = epochs x channels):", real_rms_values.shape)
print(real_rms_values[len(epochs) // 2:len(epochs) // 2 + 5, :])
print("\nFake RMS values (standard deviation, shape = epochs x channels):", fake_rms_values.shape)
print(fake_rms_values[len(epochs) // 2:len(epochs) // 2 + 5, :])