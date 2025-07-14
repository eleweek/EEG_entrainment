import time
import pygame
import sys
import argparse
from math import floor
import gc

# TODO: make sure to reenable at some point?
# We currently don't want the garbage collector to run
# But in production if we keep it off for a while we need to make sure we are doing
# manual gc.collect() calls or not using much RAM so that we aren't overflowing
gc.disable()

def find_target_fps(frequency, target_min_refresh_rate, target_max_refresh_rate):
    result = floor(target_max_refresh_rate / frequency) * frequency

    if result < target_min_refresh_rate:
        raise ValueError("Frequency too low")

    if result > target_max_refresh_rate:
        raise ValueError("Frequency too high")

    return result


def main():
    parser = argparse.ArgumentParser(description="Flicker. Expects a monitor with a variable refresh rate.")

    parser.add_argument(
        '--flicker-frequency',
        type=float,
        default=10.0
    )

    parser.add_argument('--target-min-refresh-rate',
                        type=float,
                        default=120.0,
                        help=r"This should be at least 20-25% higher than your min monitor frequency to allow for some jitter in drawing times"
    )

    parser.add_argument('--target-max-refresh-rate',
                        type=float,
                        default=120.0,
                        help=r"This should be 20-25% lower than your actual max monitor frequency to allow for some jitter in drawing times"
    )

    args = parser.parse_args()
    frequency = args.flicker_frequency
    target_max_refresh_rate = args.target_max_refresh_rate
    target_min_refresh_rate = args.target_min_refresh_rate
    
    WIDTH = 1920
    HEIGHT = 1080
    SQUARE_SIDE = 600

    # Initialize pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.SCALED | pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE , vsync=1)

    target_fps = find_target_fps(frequency, target_min_refresh_rate, target_max_refresh_rate)

    interval = 1.0 / target_fps
    off_frames_per_each_on = int((target_fps - frequency) / frequency)

    # Define the white rectangle
    rect_width, rect_height = SQUARE_SIDE, SQUARE_SIDE
    rect_x = (WIDTH - rect_width) // 2
    rect_y = (HEIGHT - rect_height) // 2

    frame_count = 0
    rectangle_on = False

    screen.fill((0, 0, 0))

    # Use perf_counter for high-precision timing
    last_draw_time = time.perf_counter()
    previous_draw_time = last_draw_time
    post_flip_time = time.perf_counter()
    previous_post_flip_time = time.perf_counter()
    start_time = last_draw_time  # For drift correction

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
        else:
            pygame.draw.rect(screen, (0, 0, 0), (rect_x, rect_y, rect_width, rect_height))

        # Calculate next frame time to prevent drift
        next_frame_time = start_time + (frame_count + 1) * interval
        current_time = time.perf_counter()
        delay = next_frame_time - current_time

        if delay > 0:
            # Sleep for most of the delay
            if delay > 0.001:  # Only sleep if delay is significant
                time.sleep(delay * 0.75)
            
            # Busy-wait for the remainder for precision
            while time.perf_counter() < next_frame_time:
                pass

        previous_draw_time = last_draw_time
        last_draw_time = time.perf_counter()

        # if rectangle_on or not (1 < frame_count % (off_frames_per_each_on + 1) <= off_frames_per_each_on):
        #     pygame.display.flip()
        pygame.display.flip()
        previous_post_flip_time = post_flip_time
        post_flip_time = time.perf_counter()
        flip_duration = (post_flip_time - last_draw_time) * 1000
        
        # Calculate actual timing metrics
        actual_interval = last_draw_time - previous_draw_time
        actual_interval_2 = post_flip_time - previous_post_flip_time
        actual_fps = 1.0 / actual_interval if actual_interval > 0 else 0
        actual_fps_2 = 1.0 / actual_interval_2 if actual_interval > 0 else 0 
        timing_error = (actual_interval - interval) * 1000 
        timing_error_2 = (actual_interval_2 - interval) * 1000 
        
        print(f'{"ON " if rectangle_on else "OFF"}: '
              f'FPS pre: {actual_fps:.2f}, '
              f'FPS post: {actual_fps_2:.2f}, '
              f'Target FPS: {target_fps:.2f}, '
              f'Error pre: {timing_error:.3f} ms, '
              f'Error post: {timing_error_2:.3f} ms, '
              f'Frame: {frame_count}, '
              f'Flip time: {flip_duration:.3f} ms')
        if flip_duration > 3.5:
            print("Warning: flip duration is high!")

        frame_count += 1

if __name__ == "__main__":
    main()