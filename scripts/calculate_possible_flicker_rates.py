import sys

def calculate_flicker_rates(refresh_rates):
    flicker_rates = []
    for refresh_rate in refresh_rates:
        for off_frames in range(1, refresh_rate):
            total_frames = off_frames + 1  # 1 frame on
            flicker_rate = refresh_rate / total_frames
            if 6 <= flicker_rate <= 14:
                flicker_rates.append((round(flicker_rate, 2), refresh_rate, off_frames))
    return flicker_rates

refresh_rates = [165, 144, 120, 100] if len(sys.argv) == 1 else [int(arg) for arg in sys.argv[1:]]

results = calculate_flicker_rates(refresh_rates)
sorted_results = sorted(results, reverse=True)

print("Flicker rates between 6 and 14 Hz, sorted in decreasing order:")
prev_rate = None
for i, (flicker_rate, refresh_rate, off_frames) in enumerate(sorted_results):
    if i == 0:
        print(f"{flicker_rate:.2f} @ {refresh_rate} Hz with 1 on and {off_frames} off")
    else:
        delta = prev_rate - flicker_rate
        print(f"{flicker_rate:.2f} @ {refresh_rate} Hz Δ{delta:.2f} with 1 on and {off_frames} off")
    prev_rate = flicker_rate