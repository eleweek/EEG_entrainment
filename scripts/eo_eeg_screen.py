import os, pygame

BG = (0,0,0)            # Black
DOT = (64, 64, 64)      # gray
RADIUS_PX = 5           # ≈0.2° diameter if ~52 px/deg

def main():
    # Hi-DPI friendly fullscreen at native pixels
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "0")
    pygame.init()
    pygame.key.set_repeat(0)

    # native_w, native_h = pygame.display.list_modes()[0]
    screen = pygame.display.set_mode(
        (1920, 1080),
        flags=pygame.FULLSCREEN | pygame.SCALED,  # SCALED→renderer+vsync, no scaling at native res
        vsync=1
    )

    center = screen.get_rect().center
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

        screen.fill(BG)
        pygame.draw.circle(screen, DOT, center, RADIUS_PX)
        pygame.display.flip()
        pygame.time.delay(5)  # be nice to CPU

    pygame.quit()

if __name__ == "__main__":
    main()

