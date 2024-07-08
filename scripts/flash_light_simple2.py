import time
import pygame
import sys
from math import floor

frequency = float(sys.argv[1])

WIDTH = 1920
HEIGHT = 1080

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


target_fps = find_target_fps(frequency, max_frequency=60)
interval = 1.0 / target_fps
off_frames_per_each_on = int((target_fps - frequency) / frequency)


# Define the white rectangle
rect_width, rect_height = 300, 300
rect_x = (WIDTH - rect_width) // 2
rect_y = (HEIGHT - rect_height) // 2

clock = pygame.time.Clock()
frame_count = 0
rectangle_on = False

screen.fill((0, 0, 0))

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

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    frame_count += 1
    print(f"Frame {frame_count}. FPS: {clock.get_fps()} Target FPS: {target_fps}")
    clock.tick(target_fps)