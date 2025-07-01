import time
import pygame
import sys
import argparse
from math import floor
import gc

gc.disable()

def main():
    parser = argparse.ArgumentParser(description="Flash Lights")
    parser.add_argument('--frequency', type=float, default=10.0)
    parser.add_argument('--max-monitor-frequency', type=float, default=60.0)

    args = parser.parse_args()
    frequency = args.frequency
    max_frequency = args.max_monitor_frequency
    
    WIDTH = 1920
    HEIGHT = 1080
    SQUARE_SIDE = 600

    # Initialize pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.SCALED | pygame.FULLSCREEN, vsync=1)

    def find_target_fps(frequency, min_frequency=48, max_frequency=165):
        result = floor(max_frequency / frequency) * frequency

        if result < min_frequency:
            raise ValueError("Frequency too low")
        
        if result > max_frequency:
            raise ValueError("Frequency too high")
        
        return result

    target_fps = find_target_fps(frequency, max_frequency=max_frequency)
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

        # Update the display
        pygame.display.flip()
        post_flip_time = time.perf_counter()
        flip_duration = (post_flip_time - last_draw_time) * 1000
        
        # Calculate actual timing metrics
        actual_interval = last_draw_time - previous_draw_time
        actual_fps = 1.0 / actual_interval if actual_interval > 0 else 0
        timing_error = (actual_interval - interval) * 1000  # in milliseconds
        
        print(f'{"ON " if rectangle_on else "OFF"}: '
              f'Actual FPS: {actual_fps:.2f}, '
              f'Target FPS: {target_fps:.2f}, '
              f'Timing error: {timing_error:.3f} ms, '
              f'Frame: {frame_count}, '
              f'Flip time: {flip_duration:.3f} ms')

        frame_count += 1

if __name__ == "__main__":
    main()