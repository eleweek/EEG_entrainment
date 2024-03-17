import os
import numpy as np
import matplotlib.pyplot as plt
from libs.file_formats import load_openbci_txt

script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_dir = os.path.join(os.path.dirname(script_dir), 'sample_data')
sample_file_path = os.path.join(sample_data_dir, 'sample1-openbci-gui.txt')

raw = load_openbci_txt(sample_file_path)
raw.filter(l_freq=1.0, h_freq=45.0)
raw.notch_filter(np.array([50, 60]), notch_widths=5, filter_length='auto', phase='zero', fir_design='firwin')
raw.plot()

psd = raw.compute_psd()
psd.plot(average=True)


plt.show()