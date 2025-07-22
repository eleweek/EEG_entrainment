import sys
import pyxdf
from pprint import pprint

filepath = sys.argv[1]
streams, header = pyxdf.load_xdf(filepath)

def print_dict_tree(d, indent=0):
    """Recursively print nested dictionaries with proper indentation"""
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict_tree(value, indent + 1)
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict):
                print("  " * indent + f"{key}:")
                for i, item in enumerate(value):
                    print("  " * (indent + 1) + f"[{i}]:")
                    print_dict_tree(item, indent + 2)

            else:
                print("  " * indent + f"{key}: {value[0] if len(value) == 1 else value}")
        else:
            print("  " * indent + f"{key}: {value}")

# Print all metadata for each stream
for i, stream in enumerate(streams):
    print(f"\n{'='*60}")
    print(f"STREAM {i+1}")
    print(f"{'='*60}")
    
    # Convert defaultdict to regular dict for prettier printing
    info_dict = dict(stream['info'])
    
    # Print all info fields
    print("\n--- Stream Metadata ---")
    print_dict_tree(info_dict)
    
    # Print data statistics
    print(f"\n--- Data Statistics ---")
    print(f"  Total samples: {len(stream['time_stamps'])}")
    if len(stream['time_stamps']) > 0:
        print(f"  First timestamp: {stream['time_stamps'][0]:.4f}")
        print(f"  Last timestamp: {stream['time_stamps'][-1]:.4f}")
        print(f"  Duration: {stream['time_stamps'][-1] - stream['time_stamps'][0]:.2f} seconds")
        
        # Calculate actual sample rate
        if len(stream['time_stamps']) > 1:
            actual_rate = len(stream['time_stamps']) / (stream['time_stamps'][-1] - stream['time_stamps'][0])
            print(f"  Actual sample rate: {actual_rate:.2f} Hz")


# Process marker stream specifically
print("\n=== Markers ===")
for stream in streams:
    if stream['info']['type'][0] == 'Markers':
        markers = stream['time_series']
        timestamps = stream['time_stamps']

        for i in range(len(timestamps)):
            print(f"  [{timestamps[i]:10.4f}] {markers[i][0]}")
