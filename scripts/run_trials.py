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

    # after clearing the screen post-stimulus
    W, H = screen.get_size()
    center = (W // 2, H // 2)

    screen.fill((0, 0, 0))
    draw_fixation(screen, center)
    pygame.display.flip()

    ground_truth_side = 'left' if angle_deg == 0.0 else 'right'

    resp_key, correct, rt_ms, timed_out = collect_response_with_feedback(
        screen, center,
        timeout_ms=1300,
        show_feedback=True,
        correct_side=ground_truth_side    # 'left' or 'right' for that trial
    )


# --- fixation + feedback glyphs ---
def draw_fixation(screen, center, color=(160,160,160)):
    x,y = center; s=12; w=2
    pygame.draw.line(screen, color, (x-s, y), (x+s, y), w)
    pygame.draw.line(screen, color, (x, y-s), (x, y+s), w)

def draw_tick(screen, center):
    x,y = center; w=3
    pygame.draw.lines(screen, (80,200,80), False,
                      [(x-10, y+2), (x-2, y+10), (x+12, y-8)], w)

def draw_cross(screen, center):
    x,y = center; w=3
    pygame.draw.line(screen, (200,80,80), (x-10,y-10), (x+10,y+10), w)
    pygame.draw.line(screen, (200,80,80), (x-10,y+10), (x+10,y-10), w)

def draw_timeout(screen, center):
    x,y = center; w=3
    pygame.draw.circle(screen, (200,180,60), (x,y), 10, w)

# --- response collection with timeout + feedback ---
def collect_response_with_feedback(screen, center, timeout_ms=1300,
                                   show_feedback=True, correct_side='left'):
    """
    Wait for LEFT/RIGHT within timeout_ms.
    Returns (resp_key, correct_bool, rt_ms, timed_out)
    """
    pygame.event.clear()
    t0 = time.perf_counter()
    resp_key = None; rt_ms = None

    while (time.perf_counter() - t0) * 1000 < timeout_ms:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: raise SystemExit
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    resp_key = e.key
                    rt_ms = int((time.perf_counter() - t0) * 1000)
                    break
                if e.key == pygame.K_ESCAPE: raise SystemExit
        if resp_key is not None:
            break
        pygame.time.delay(1)

    timed_out = resp_key is None
    # decide correctness (you set the ground truth elsewhere)
    is_left_correct = (correct_side == 'left')
    if not timed_out:
        said_left = (resp_key == pygame.K_LEFT)
        correct = (said_left == is_left_correct)
    else:
        correct = False

    if show_feedback:
        # neutral background, then small glyph at fixation for 100 ms
        # (paper used a tick/cross at fixation, 100 ms)
        screen.fill((0,0,0))
        draw_fixation(screen, center)
        if timed_out:
            draw_timeout(screen, center)
        else:
            (draw_tick if correct else draw_cross)(screen, center)
        pygame.display.flip()
        pygame.time.delay(100)

    return resp_key, correct, rt_ms if rt_ms is not None else -1, timed_out


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

        angle_deg = 0.0 if random.random() < 0.5 else 90.0
        run_trial(
            screen, aperture_rect, flicker_rect,
            freq_hz=10.0, cycles=15, delay_cycles=delay, stim_ms=200,
            angle_deg=angle_deg, snr=0.24, density=0.03, shift_px=14, dot_radius_px=1,
            handed="cw",
            target_min_refresh_rate=120.0, target_max_refresh_rate=120.0,
        )

        trial += 1

        pygame.time.delay(1200)

    pygame.quit()

if __name__ == "__main__":
    demo()
