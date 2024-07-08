# Andrew Alpha IAF
# We want robust Individualized Alpha Freq identification
# axs 2024

import numpy as np
import mne
import pyxdf

from pandas import read_csv
from os import getcwd, sep

data_folder = "C:\eegdata\Andrew\MindTunes\setup\sub-P001\ses-S001\eeg"
xdf_file = data_folder + "/sub-P001_ses-S001_task-Default_run-001_eeg.xdf"

debug = 1

#raw = mne.io.read_raw(xdf_file,stream_ids=[1],preload=True)
streams, header = pyxdf.load_xdf(xdf_file)

streamcount = len(streams)
EEG_mask = np.zeros(streamcount)
dataset_info_str = f"There are {streamcount+1} streams here:"
print(dataset_info_str)
for i in range(streamcount):
    if streams[i]["info"]["type"] == ["EEG"]:
        EEG_mask[i] = 1
    print(f'{i} - {streams[i]["info"]["name"][0]} - isEEG-{EEG_mask[i]}')

EEG_streamcount = int(sum(EEG_mask))

EEG_to_analyze = EEG_mask

def get_locs(loc_type, n_elec):
    electrode_names = False
    folder_here = getcwd()

    if folder_here.endswith("scripts"):
        loc_folder = folder_here[0:-8] + sep +'hardware' + sep + 'EEG_electrode_locs' + sep
    else:
        loc_folder = folder_here + sep +'hardware' + sep + 'EEG_electrode_locs' + sep


    if loc_type.lower() == 'biosemi':
        
        loc_df = read_csv(loc_folder+'Biosemi_32.csv')
        electrode_list = loc_df.values.tolist()
        electrode_names = []

        for i in electrode_list:
            electrode_names.append(i[0])  # make simple single list of str

    
    if electrode_names == False:
        electrode_names = []
        for i in range(n_elec):
            i_str = f"i"
            electrode_names.append(i_str)
        
    return electrode_names






# Loop thru EEG sources
for EEG_stream in range(EEG_streamcount):
    stream_index = np.where(EEG_to_analyze == 1)[0][0]
    EEG_to_analyze[stream_index] = 0

    data = streams[stream_index]["time_series"].T
    n_elec = data.shape[0]

    type_tag = streams[stream_index]['info']['name'][0]
    type_tag = type_tag.lower()

    if n_elec > 31:
        n_elec = 31
        data = data[0:n_elec]

    srate = streams[stream_index]["info"]["nominal_srate"][0]

    loc_type = streams[stream_index]["info"]["name"][0]
    electrode_names = get_locs(loc_type,n_elec)


    # Sanity check
    rough_inferred_time_min = round(data.shape[1] / float(srate) / 60)
    debug_str = f"Here, we have {streams[stream_index]['info']['name'][0]} data with {data.shape[0]+1} elecs and {data.shape[1]+1} timepoints. At this sample rate of {srate}, that is around {rough_inferred_time_min} minutes"
    if debug:    
        print(debug_str)

    # Make the MNE data
    info = mne.create_info(electrode_names, srate, "eeg")

    raw = mne.io.RawArray(data,info)

    # Process Step 1 - re-reference
    if type_tag in "biosemi,brainvision":
        raw.set_eeg_reference(ref_channels=["Cz"])
    
    # Process Step 2 - filter
    low_freq_filter = 0.5
    high_freq_filter = 45
    raw_bandpassed = raw.copy().filter(l_freq=low_freq_filter, h_freq=high_freq_filter)

    # Plot
    raw.plot()











    

