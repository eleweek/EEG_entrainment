# experiment_glass_with_flicker.py
import math
import random
import time
import pygame

from flicker import run_flicker
from glass import draw_glass

def wait_exact(seconds: float):
    """Busy-wait for precise pre-stim delays """
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < seconds:
        pass


def run_trial(
    screen: pygame.Surface,
    aperture_rect: pygame.Rect,
    flicker_rect: pygame.Rect,
    *,
    freq_hz: float,
    cycles: int,         # e.g., 15
    delay_cycles: float, # e.g., 2.0 for P, 2.5 for T
    stim_ms: int,        # e.g., 200
    # Glass params (match your previous script)
    angle_deg: float,
    snr: float,
    density: float,
    shift_px: float,
    dot_radius_px: int,
    handed: str = "cw",
    seed: int | None = None,
    # flicker scheduling envelope (same as your CLI)
    target_min_refresh_rate: float = 120.0,
    target_max_refresh_rate: float = 120.0,
):
    W, H = screen.get_size()

    # 1) Flicker (pulses)
    run_flicker(
        screen, flicker_rect,
        frequency=freq_hz,
        target_min_refresh_rate=target_min_refresh_rate,
        target_max_refresh_rate=target_max_refresh_rate,
        cycles=cycles,                    # blocks until N ON-pulses shown
        report_every=10_000               # quiet
    )

    # 2) Pre-stimulus delay at the intended phase offset
    wait_exact(delay_cycles / freq_hz)

    # 3) Draw Glass pattern off-screen, then blit to aperture
    aw, ah = aperture_rect.size
    stim = pygame.Surface((aw, ah))
    # Center in its own surface
    draw_glass(
        stim,
        center=(aw // 2, ah // 2),
        size=aw,                          # square aperture
        angle_deg=angle_deg,
        snr=snr,
        density=density,
        shift=shift_px,
        dot_r=dot_radius_px,
        handed=handed,
        seed=seed if seed is not None else random.randrange(1 << 30),
    )

    # Clear frame, blit stimulus, flip
    screen.fill((0, 0, 0))
    screen.blit(stim, aperture_rect.topleft)
    pygame.display.flip()

    # Keep it for stim_ms
    pygame.time.delay(stim_ms)

    # Clear after stimulus window (optional)
    screen.fill((0, 0, 0))
    pygame.display.flip()

def demo():
    pygame.init()
    WIDTH, HEIGHT = 1920, 1080
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT),
        flags=pygame.SCALED | pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE,
        vsync=1
    )

    # Geometry: big central flicker; smaller central aperture for Glass
    flicker_side = 600
    aperture_side = 412                  # ~7.9° at your earlier 83 cm example
    flicker_rect  = pygame.Rect((WIDTH - flicker_side)//2,  (HEIGHT - flicker_side)//2,  flicker_side,  flicker_side)
    aperture_rect = pygame.Rect((WIDTH - aperture_side)//2, (HEIGHT - aperture_side)//2, aperture_side, aperture_side)

    running = True
    trial = 0
    conds = ["P", "T"]                   # alternate P-match / T-match
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        cond = conds[trial % 2]
        delay = random.choice([1,2,3]) + (0.5 if cond == "T" else 0.0)

        run_trial(
            screen, aperture_rect, flicker_rect,
            freq_hz=10.0, cycles=15, delay_cycles=delay, stim_ms=200,
            angle_deg=45.0, snr=0.24, density=0.03, shift_px=14, dot_radius_px=1,
            handed="cw",
            target_min_refresh_rate=120.0, target_max_refresh_rate=120.0,
        )

        trial += 1

        pygame.time.delay(1200)

    pygame.quit()

if __name__ == "__main__":
    demo()
