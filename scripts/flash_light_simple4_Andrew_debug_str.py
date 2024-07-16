import time
import pygame
import sys
import argparse
import platform

import numpy as np
from math import floor

# Uses Community Edition of PyGame, from:
# pip install pygame-ce --upgrade


def flicker_freq_calc(desired_flick,mon_freq):
    # Imagine one bright frame, followed by some integer number of dark frames
    dark_frame_count = np.zeros(mon_freq[0])
    both_frame_count = np.zeros(mon_freq[0])
    flicker_rates = np.zeros(mon_freq[0])

    for i in range(mon_freq[0]):
        dark_frame_count[i] = i
        both_frame_count[i] = i+1
        flicker_rates[i] = 1 / (both_frame_count[i] / mon_freq[0])

    # Find closest flicker rate to desired
    closest_idx = np.searchsorted(flicker_rates[::-1],desired_flick)
    closest_idx = mon_freq[0] - closest_idx  # inverse as decending above

    print(f"At {mon_freq[0]} Hz hardware refresh, the closest integer flicker to {desired_flick} Hz is {flicker_rates[closest_idx]} Hz, with 1 bright frame and {dark_frame_count[closest_idx]} dark")

    return dark_frame_count[closest_idx], flicker_rates[closest_idx], flicker_rates




def main():
    parser = argparse.ArgumentParser(description="Flash Lights")
    parser.add_argument('--frequency',type=float,default=144.0)
    parser.add_argument('--max_frequency',type=float,default=144.0)
    parser.add_argument('--desired_flicker',type=float,default=10.0)
    
    args = parser.parse_args()
    frequency = args.frequency
    max_frequency = args.max_frequency
    desired_flicker = args.desired_flicker
    
    WIDTH = 1920
    HEIGHT = 1080

    SQUARE_SIDE = 600

    # Initialize pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.SCALED | pygame.FULLSCREEN, vsync=1)
    mon_freq = pygame.display.get_desktop_refresh_rates()
    # Uses Community Edition of PyGame, from:
    # pip install pygame-ce --upgrade

    if all(x >= frequency for x in mon_freq):
        print('All monitors can be driven at or above target frequency')
    else:
        print('*****')
        print('WARNING - at least one monitor has refresh BELOW requested frequency')
        print('*****')
        time.sleep(0.3)
    
    # Check flicker
    off_frames_per_each_on, flicker_int_target, possible_flicker_rates = flicker_freq_calc(desired_flicker,mon_freq)

    # Report vars
    log_len = frequency
    log_count = 0
    recent_ft = np.zeros(int(log_len))
    recent_rect_on = np.zeros(int(log_len))

    def find_target_fps(frequency, min_frequency=48, max_frequency=165):
        result = floor(max_frequency / frequency) * frequency

        if result < min_frequency:
            # It shouldn't happen, but we are throwing an error just in case
            # For example, if the code is modified in the future
            raise ValueError("Frequency too low")
        
        if result > max_frequency:
            raise ValueError("Frequency too high")
        
        return result


    target_fps = find_target_fps(frequency, max_frequency=max_frequency)
    interval = 1.0 / target_fps
    #off_frames_per_each_on = int((target_fps - frequency) / frequency)
    #off_frames_per_each_on = 5


    # Define the white rectangle
    rect_width, rect_height = SQUARE_SIDE, SQUARE_SIDE
    rect_x = (WIDTH - rect_width) // 2
    rect_y = (HEIGHT - rect_height) // 2

    frame_count = 0
    rectangle_on = False

    screen.fill((0, 0, 0))

    last_draw_time = time.time()
    previous_draw_time = last_draw_time

    last_log_time = time_here = time.perf_counter_ns()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        rectangle_on = frame_count % (off_frames_per_each_on + 1) == 0

        # Draw the white rectangle if it should be visible
        if rectangle_on:
            pygame.draw.rect(screen, (255, 255, 255), (rect_x, rect_y, rect_width, rect_height))
            recent_rect_on[log_count] = 1
        else:
            pygame.draw.rect(screen, (0, 0, 0), (rect_x, rect_y, rect_width, rect_height))

        delay = interval - (time.time() - last_draw_time)
        frame_prep_time = 1/frequency * 0.33  # Leave some time to prep the next frame promptly
        if delay > 0:
            # Sleep for the most of the delay and then loop until we hit the right moment
            # Should wake in time to draw the next frame promptly
            time.sleep(delay * 0.5)
            while time.time() < last_draw_time + interval - frame_prep_time:
                continue

        previous_draw_time = last_draw_time
        last_draw_time = time.time()


        # Update the display
        pygame.display.flip()
        frame_count += 1
        

        # Report str
        # Print once per sec
        recent_ft[log_count] = last_draw_time - previous_draw_time
        log_count += 1
        if log_count >= log_len:
            
            log_time = time.perf_counter_ns() - last_log_time
            last_log_time = time.perf_counter_ns()

            report_time = f"Rendered {log_count} frames in {round(log_time / 1e9,3)} s, implying {round(log_count / (log_time / 1e9),2)} Hz"
            print(report_time)
            log_count = 0

            frametime_mean = np.mean(recent_ft)
            ft_as_Hz = round(1/frametime_mean,1)
            report_str = f"Apparent frametime of {round(frametime_mean,4)} s, implying {ft_as_Hz} Hz real draw rate"
            print(report_str)

            report_rect = f"Rect on for {sum(recent_rect_on)}, so {sum(recent_rect_on)} Hz target flicker, {round(sum(recent_rect_on)/(log_time / 1e9),2)} Hz real, and {flicker_int_target} integer frame target"
            print(report_rect)
            recent_rect_on = np.zeros(int(log_len))

            print()


if __name__ == "__main__":
    main()
