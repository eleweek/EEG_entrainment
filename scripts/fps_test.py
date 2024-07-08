import pygame
import time

# Initialize Pygame
pygame.init()

# Set the dimensions of the window
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)


EXPECTED_FPS = 117

# Set the title of the window
pygame.display.set_caption(f'{EXPECTED_FPS} Hz Frame Test')

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Colors
black = (0, 0, 0)
white = (255, 255, 255)

# Main loop flag
running = True

# Start time
start_time = time.time()

# Frame count
frames = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with black
    screen.fill(black)

    # Draw a simple white rectangle
    pygame.draw.rect(screen, white, (150, 100, 500, 400))

    # Update the display
    pygame.display.flip()

    # Tick the clock to control the frame rate
    clock.tick(EXPECTED_FPS)

    # Frame count
    frames += 1
    print(f"Frame {frames}. FPS", clock.get_fps())

# Calculate and print the FPS
end_time = time.time()
elapsed_time = end_time - start_time
fps = frames / elapsed_time
print(f"Expected FPS: {EXPECTED_FPS}")
print(f"Actual FPS: {fps}")

# Quit Pygame
pygame.quit()