# Copied from: https://github.com/pygame/pygame/issues/3085

import pygame
import pygame_gui


pygame.init()


pygame.display.set_caption('Vsync test')
vsync=0
window_surface = pygame.display.set_mode((800, 600), flags=pygame.SCALED, vsync=vsync)
manager = pygame_gui.UIManager((800, 600))

background = pygame.Surface((800, 600))
background.fill(pygame.Color(0, 0, 0))


fps_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((350, 280), (150, 30)),
                                        text='0 FPS',
                                        manager=manager)

clock = pygame.time.Clock()
is_running = True

while is_running:
    time_delta = clock.tick()/1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
            if vsync:
                vsync = 0
            else:
                vsync = 1
            print("Vsync", vsync)
            window_surface = pygame.display.set_mode((800, 600), flags=pygame.SCALED, vsync=vsync)

        manager.process_events(event)

    fps_label.set_text(f"{clock.get_fps():.2f}" + " FPS")

    manager.update(time_delta)

    window_surface.blit(background, (0, 0))
    manager.draw_ui(window_surface)

    pygame.display.update()
