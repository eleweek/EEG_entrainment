import base64
import sys
from io import BytesIO

import mne
import mpld3
import numpy as np
import matplotlib.pyplot as plt

from libs.file_formats import load_raw_xdf
from libs.filters import filter_and_drop_dead_channels
from libs.plot import add_red_line_with_value
from libs.psd import get_peak_alpha_freq, fit_one_over_f_curve


def matplotlib_to_img(fig):
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    plt.close(fig)
    
    # Base64 encode the image
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    return img_base64

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f'{minutes:02}:{seconds:02}'

input_xdf_filename = sys.argv[1]
output_report_filename = sys.argv[2]

# Load the MNE Raw file
raw = load_raw_xdf(input_xdf_filename)
filter_and_drop_dead_channels(raw)

chunk_duration = 30.0
chunk_shift = 5

n_chunks = int(np.floor((raw.times[-1] - chunk_duration) / chunk_shift)) + 1


peak_alpha_freqs = []
dbs = []

psd_figs = []

# Iterate over the raw data in chunks
for i in range(n_chunks):
    print(f'Processing chunk {i + 1}/{n_chunks}')
    tmin = i * chunk_shift
    tmax = i * chunk_shift + chunk_duration
    
    # Get the data for the current chunk
    chunk_data = raw.copy().crop(tmin=tmin, tmax=tmax)

    psd = chunk_data.compute_psd(fmin=1.0, fmax=60.0)

    # Call the functions to get peak alpha frequency and dB
    peak_alpha_freq = get_peak_alpha_freq(psd)
    psd_freqs, fit_freq_range, fitted_curve, delta_db = fit_one_over_f_curve(psd, min_freq=3, max_freq=40, peak_alpha_freq=peak_alpha_freq)

    fig = psd.plot(average=True, show=False)
    psd_figs.append(fig)
    ax = fig.get_axes()[0]
    ax.plot(psd_freqs[fit_freq_range], fitted_curve, label='1/f fit', linewidth=1, color='darkmagenta')
    ax.set_title(f'PSD for time {format_time(tmin)}..{format_time(tmax)}')

    add_red_line_with_value(fig, peak_alpha_freq, delta_db)
    
    peak_alpha_freqs.append(peak_alpha_freq)
    dbs.append(delta_db)

# Create the first chart for peak alpha frequency
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(n_chunks) * chunk_duration, peak_alpha_freqs)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Peak Alpha Frequency (Hz)')
ax1.set_title('Peak Alpha Frequency over Time')

# Create the second chart for dB
fig2, ax2 = plt.subplots()
ax2.plot(np.arange(n_chunks) * chunk_duration, dbs)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Delta peak (dB)')
ax2.set_title('Delta peak over time')

# Display the charts
# plt.show()

fig1_html = mpld3.fig_to_html(fig1)
fig2_html = mpld3.fig_to_html(fig2)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>EEG Analysis Report</title>
</head>
<body>
    <h1>EEG Analysis Report</h1>
    <h2>Peak Alpha Frequency over Time (Last 3 Channels)</h2>
    {fig1}
    <h2>Decibel over Time (Last 3 Channels)</h2>
    {fig2}

    <h2>PSD for each chunk</h2>
    {psd_images_html}
</body>
</html>
"""

report_html = html_template.format(fig1=fig1_html, fig2=fig2_html, psd_images_html=''.join([f'<img src="data:image/png;base64,{matplotlib_to_img(fig)}">' for fig in psd_figs]))
with open(output_report_filename, 'w') as f:
    f.write(report_html)