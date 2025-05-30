import sys
import pygame
import time
import numpy as np
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 400, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Flashing Square")

monitor_Hz = int(sys.argv[1])

frequency = 8.5

white = (255, 255, 255)
black = (0, 0, 0)

screen.fill(black)

t0 = time.perf_counter_ns()
last_check = t0
post_flip_wait = 0.51 * (1/monitor_Hz)

frametime_log = np.zeros(1000)
f_i = 0

print("Frequency", frequency)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            print("Pygame keydown", event.key, pygame.K_LEFT, pygame.K_RIGHT)
            if event.key == pygame.K_LEFT:
                frequency -= 0.25
                print("Decreasing frequency to", frequency)
            if event.key == pygame.K_RIGHT:
                frequency += 0.25
                print("Increasing frequency to", frequency)

    
    

    pygame.draw.rect(screen, white, (150, 150, 100, 100))
    pygame.display.flip()
    time_here = time.perf_counter_ns()
    frametime_log[f_i] = time_here - last_check
    last_check = time_here
    f_i += 1
    
    time.sleep(post_flip_wait)

    pygame.draw.rect(screen, black, (150, 150, 100, 100))
    pygame.display.flip()
    time_here = time.perf_counter_ns()
    frametime_log[f_i] = time_here - last_check
    last_check = time_here
    f_i += 1
    
    time.sleep(post_flip_wait)

    if f_i > 999:
        running = False

# Simple plot for Frametime
plt.plot(frametime_log)
plt.show()

# Quit Pygame
pygame.quit()
