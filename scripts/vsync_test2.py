# Adapted from: https://github.com/pygame/pygame/issues/3085

import pygame

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

pygame.init()
pygame.display.set_caption('Vsync test')
vsync = 0
display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), flags=pygame.SCALED, vsync=vsync)

background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
background.fill(pygame.Color(0, 0, 0))

font = pygame.font.Font(None, 36)


clock = pygame.time.Clock()

is_running = True
while is_running:
    time_delta = clock.tick() / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            if vsync:
                vsync = 0
            else:
                vsync = 1
            window_surface = pygame.display.set_mode((800, 600), flags=pygame.SCALED, vsync=vsync)


    display.blit(background, (0, 0))
    text = font.render(f"VSync = {1 if vsync else 0}, {clock.get_fps():.2f} FPS", True, pygame.Color(255, 255, 255)) 
    display.blit(text, (350, 280))
    

    pygame.display.flip()
