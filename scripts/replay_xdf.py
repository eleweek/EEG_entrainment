import sys
import tempfile
import time

import numpy as np
from mne_lsl.player import PlayerLSL

from libs.file_formats import load_raw_xdf


xdf_file_name = sys.argv[1]

raw = load_raw_xdf(xdf_file_name)
with tempfile.TemporaryDirectory() as tempdir:
    temp_file_name = f'{tempdir}/temp.fif'
    raw.save(temp_file_name, overwrite=True)

    player = PlayerLSL(temp_file_name)
    player.start()

    sfreq = player.info["sfreq"]
    chunk_size = player.chunk_size
    interval = chunk_size / sfreq  # in seconds
    print(f"Interval between 2 push operations: {interval} seconds.")

    seconds_played = 0
    while True:
        print("Still playing (probably). For " + str(seconds_played) + " seconds.")
        time.sleep(1.0)
        seconds_played += 1