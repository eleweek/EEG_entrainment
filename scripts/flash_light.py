import pygame
import time

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 400, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Flashing Square")

frequency = 10

white = (255, 255, 255)
black = (0, 0, 0)

screen.fill(black)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    flash_duration = 1 / (2 * frequency)

    pygame.draw.rect(screen, black, (150, 150, 100, 100))
    pygame.display.flip()
    time.sleep(flash_duration)

    pygame.draw.rect(screen, white, (150, 150, 100, 100))
    pygame.display.flip()
    time.sleep(flash_duration)

# Quit Pygame
pygame.quit()