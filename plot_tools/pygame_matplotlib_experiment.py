import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from libs.file_formats import load_openbci_txt
from libs.filters import create_new_raw_with_brainflow_filters_applied

script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_dir = os.path.join(os.path.dirname(script_dir), 'sample_data')
sample_file_path = os.path.join(sample_data_dir, 'sample1-openbci-gui.txt')

raw = load_openbci_txt(sample_file_path)

brainflow_raw = create_new_raw_with_brainflow_filters_applied(raw)

raw.filter(l_freq=1.0, h_freq=45.0)
raw.notch_filter(50, notch_widths=4)
raw.notch_filter(60, notch_widths=4)
raw.plot()

psd = raw.compute_psd()
psd.plot(average=True)

brainflow_raw.plot()
brainflow_psd = brainflow_raw.compute_psd()
brainflow_psd.plot(average=True)


plt.show()