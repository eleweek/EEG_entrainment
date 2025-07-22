import sys
from pprint import pprint
from collections import Counter

import pyxdf

filepath = sys.argv[1]

streams, _ = pyxdf.load_xdf(filepath)



filepath = sys.argv[1]
streams, header = pyxdf.load_xdf(filepath)

# Analyze all streams
print("=== XDF File Summary ===")
print(f"Number of streams: {len(streams)}")
for i, stream in enumerate(streams):
    info = stream['info']
    print(f"\nStream {i+1}: {info['name'][0]}")
    print(f"  Type: {info['type'][0]}")
    print(f"  Channels: {info['channel_count'][0]}")
    print(f"  Samples: {len(stream['time_stamps'])}")
    print(f"  Duration: {stream['time_stamps'][-1] - stream['time_stamps'][0]:.2f} seconds")

# Process marker stream specifically
print("\n=== Markers ===")
for stream in streams:
    if stream['info']['type'][0] == 'Markers':
        markers = stream['time_series']
        timestamps = stream['time_stamps']

        for i in range(len(timestamps)):
            print(f"  [{timestamps[i]:10.4f}] {markers[i][0]}")
