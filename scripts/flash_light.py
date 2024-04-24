import pygame
import time

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 400, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Flashing Square")

frequency = 8

white = (255, 255, 255)
black = (0, 0, 0)

screen.fill(black)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    start_flash_time = time.time()
    flash_duration = 1 / 30.0

    pygame.draw.rect(screen, white, (150, 150, 100, 100))
    pygame.display.flip()
    time.sleep(flash_duration)

    pygame.draw.rect(screen, black, (150, 150, 100, 100))
    pygame.display.flip()
    time.sleep(1.0 / frequency - time.time() + start_flash_time)

# Quit Pygame
pygame.quit()