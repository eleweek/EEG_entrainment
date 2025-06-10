import time
import pygame
import sys
import argparse

from math import floor

def main():
    parser = argparse.ArgumentParser(description="Flash Lights")
    parser.add_argument('--frequency',type=float, default=10.0)
    parser.add_argument('--max-monitor-frequency',type=float, default=60.0)

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
            # It shouldn't happen, but we are throwing an error just in case
            # For example, if the code is modified in the future
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

    last_draw_time = time.time()
    previous_draw_time = last_draw_time

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

        delay = interval - (time.time() - last_draw_time)
        if delay > 0:
            # Sleep for the most of the delay and then loop until we hit the right moment
            time.sleep(delay * 0.75)
            while time.time() < last_draw_time + interval:
                continue

        previous_draw_time = last_draw_time
        last_draw_time = time.time()


        # Update the display
        pygame.display.flip()
        frame_count += 1
        print(f"{"on " if rectangle_on else "off"}: Flip returned after", time.time() - last_draw_time, "Interval", interval, "Delay", delay, "Expected FPS", 1 / (last_draw_time - previous_draw_time), "Target FPS", target_fps)
        print()

if __name__ == "__main__":
    main()
