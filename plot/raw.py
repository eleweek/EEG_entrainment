import argparse

import mne
from libs.file_formats import load_recording
import matplotlib.pyplot as plt

from libs.filters import filter_and_drop_dead_channels
from libs.parse import parse_picks

parser = argparse.ArgumentParser(description="Plot raw EEG data")
parser.add_argument('file', type=str, help='Path to the recording file')
parser.add_argument('--picks', type=str, default=None, help='Comma or space-separated list of channels to use')
args = parser.parse_args()

raw = load_recording(args.file)
picks = parse_picks(args.picks)
filter_and_drop_dead_channels(raw, picks)

# raw.plot(scalings=dict(eeg=50e-6))
raw.plot(scalings="auto")
plt.show()
