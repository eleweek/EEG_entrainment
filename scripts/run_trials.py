# run_trials.py
from __future__ import annotations
import argparse, enum, random, time, os
from dataclasses import dataclass

import pygame

from flicker import run_flicker          # your pulse-train function
from glass   import draw_glass           # your Glass generator (draws onto a Surface)

# ============================ Config / Dataclasses ============================

@dataclass
class TaskConfig:
    freq_hz: float = 10.0
    cycles: int = 15
    delay_choices_peak = (1.0, 2.0, 3.0)
    delay_choices_trough = (1.5, 2.5, 3.5)
    stim_ms: int = 200                    # fixed 200 ms (paper)
    resp_extra_ms: int = 1300             # +1.3 s fixation = 1.5 s total from onset
    iti_ms: int = 1500
    iti_jitter_ms: int = 250
    feedback_ms: int = 100
    show_feedback: bool = True            # set False for Session 2

@dataclass
class StimulusConfig:
    aperture_side_px: int = 412           # ~7.9° in your earlier example
    flicker_side_px: int = 600            # square pulse patch
    dot_r_px: int = 1                     # 2 px diameter
    shift_px: int = 14                    # ~16.2' at your ppd
    density: float = 0.03
    snr_level: float = 0.24               # base; add ±1–3% jitter per trial
    handed: str = "cw"                    # your generator option

class Phase(enum.Enum):
    FIX   = 0
    FLICK = 1
    DELAY = 2
    STIM  = 3
    RESP  = 4
    FB    = 5
    ITI   = 6

# ============================ Small drawing helpers ===========================

def draw_fixation(screen: pygame.Surface, center: tuple[int,int], color=(120, 120, 120)):
    radius_px = 5
    pygame.draw.circle(screen, color, center, radius_px)

def glyph_tick(screen, c):
    pygame.draw.lines(screen,(80,200,80),False,[(c[0]-10,c[1]+2),(c[0]-2,c[1]+10),(c[0]+12,c[1]-8)],3)

def glyph_cross(screen, c):
    pygame.draw.line(screen,(200,80,80),(c[0]-10,c[1]-10),(c[0]+10,c[1]+10),3)
    pygame.draw.line(screen,(200,80,80),(c[0]-10,c[1]+10),(c[0]+10,c[1]-10),3)

# ============================ Geometry =======================================

def build_centered_rect(center: tuple[int,int], side: int) -> pygame.Rect:
    cx, cy = center
    return pygame.Rect(cx - side//2, cy - side//2, side, side)

# ============================ Trial runner (FSM) ==============================

def run_one_trial(
    screen: pygame.Surface,
    task: TaskConfig,
    stimcfg: StimulusConfig,
    *,
    cond: str,                     # "P" or "T"
    angle_deg: float,              # 0.0 (concentric) or 90.0 (radial) or anything in between
    snr_jitter: float,             # e.g., uniform(-0.03, +0.03)
    seed: int,
    use_debug_overlay: bool=False
):
    W, H = screen.get_size()
    center_screen = (W//2, H//2)

    # Build rects from a single center source of truth
    flicker_rect  = build_centered_rect(center_screen, stimcfg.flicker_side_px)
    aperture_rect = build_centered_rect(center_screen, stimcfg.aperture_side_px)

    # Pre-render the Glass stimulus during ITI / setup
    stim = pygame.Surface(aperture_rect.size)
    stim.fill((0,0,0))
    draw_glass(
        stim, center=(aperture_rect.w//2, aperture_rect.h//2), size=aperture_rect.w,
        angle_deg=angle_deg,
        snr=stimcfg.snr_level + snr_jitter,
        density=stimcfg.density,
        shift=stimcfg.shift_px,
        dot_r=stimcfg.dot_r_px,
        handed=stimcfg.handed,
        seed=seed,
    )

    # Ground-truth mapping: LEFT for concentric (0°), RIGHT for radial (90°)
    is_left_correct = (angle_deg == 0.0)

    # Choose delay cycles by condition (peak vs trough)
    delay_choices = task.delay_choices_peak if cond == "P" else task.delay_choices_trough
    delay_cycles = random.choice(delay_choices)

    # FSM vars
    phase = Phase.FIX
    stim_on_t = None
    resp_deadline = None
    fb_deadline = None
    iti_deadline = None

    resp_key = None
    correct = False
    timed_out = False
    response_enabled = False
    rt_ms = -1

    # Prepare static background frame (black)
    screen.fill((0,0,0))
    draw_fixation(screen, center_screen)
    pygame.display.flip()

    clock = pygame.time.Clock()
    running = True
    while running:
        # ---- 1) Events: one poll per frame ----
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); raise SystemExit
                if e.key == pygame.K_F1:
                    use_debug_overlay = not use_debug_overlay

                # Accept arrows during STIM and RESP phases
                if e.key in (pygame.K_LEFT, pygame.K_RIGHT) and response_enabled and phase in (Phase.STIM, Phase.RESP):
                    resp_key = e.key
                    rt_ms = int((time.perf_counter() - stim_on_t) * 1000)
                    said_left = (resp_key == pygame.K_LEFT)
                    correct = (said_left == is_left_correct)
                    timed_out = False
                    # Move straight to feedback; stimulus will be cleared at 200 ms below
                    phase = Phase.FB
                    fb_deadline = time.perf_counter() + task.feedback_ms/1000.0

        now = time.perf_counter()

        # ---- 2) Phase logic ----
        if phase == Phase.FIX:
            # Draw fixation (already on-screen); advance to FLICK
            phase = Phase.FLICK

        elif phase == Phase.FLICK:
            run_flicker(
                screen, flicker_rect,
                frequency=task.freq_hz,
                target_min_refresh_rate=120.0,
                target_max_refresh_rate=120.0,
                cycles=task.cycles,
                report_every=10_000
            )
            phase = Phase.DELAY

        elif phase == Phase.DELAY:
            # Busy-wait but pump events so window stays responsive
            target = now + (delay_cycles / task.freq_hz)
            while time.perf_counter() < target and phase == Phase.DELAY:
                pygame.event.pump()
            phase = Phase.STIM

        elif phase == Phase.STIM:
            # Show stimulus, start the 1.5 s total response window
            screen.fill((0,0,0))
            screen.blit(stim, aperture_rect.topleft)
            if use_debug_overlay:
                pygame.draw.rect(screen, (0,255,0), flicker_rect, 1)
                pygame.draw.rect(screen, (255,0,0), aperture_rect, 1)
                draw_fixation(screen, center_screen, (0,0,255))

            pygame.event.clear() # drop any pre-onset key presses
            pygame.display.flip() # stimulus appears

            stim_on_t = time.perf_counter()
            response_enabled = True 
            # total window = 200 ms stimulus + 1.3 s fixation
            resp_deadline = stim_on_t + (task.stim_ms + task.resp_extra_ms) / 1000.0
            phase = Phase.RESP

        elif phase == Phase.RESP:
            # Keep stimulus visible for its 200 ms; then blank to fixation until deadline
            if now - stim_on_t >= task.stim_ms/1000.0:
                # Clear to fixation only once
                screen.fill((0,0,0))
                draw_fixation(screen, center_screen)
                if use_debug_overlay:
                    pygame.draw.rect(screen, (0,255,0), flicker_rect, 1)
                    pygame.draw.rect(screen, (255,0,0), aperture_rect, 1)
                    draw_fixation(screen, center_screen, (0,0,255))
                pygame.display.flip()
                # Switch to post-stim response waiting but we remain in Phase.RESP
                phase = Phase.RESP  # explicit, for clarity

            if now >= resp_deadline and resp_key is None:
                timed_out = True
                correct = False
                rt_ms = -1
                response_enabled = False
                phase = Phase.FB
                fb_deadline = now + task.feedback_ms/1000.0

        elif phase == Phase.FB:
            response_enabled = False
            # Show 100 ms feedback (paper showed only after response; you can skip for timeouts)
            screen.fill((0,0,0))
            draw_fixation(screen, center_screen)
            if not timed_out and task.show_feedback:
                (glyph_tick if correct else glyph_cross)(screen, center_screen)
            pygame.display.flip()

            if time.perf_counter() >= fb_deadline:
                phase = Phase.ITI
                jitter = random.randint(-task.iti_jitter_ms, task.iti_jitter_ms)
                iti_deadline = time.perf_counter() + (task.iti_ms + jitter)/1000.0

        elif phase == Phase.ITI:
            # ITI idle; perfect place to save to DB/PNG if you add that later
            if time.perf_counter() >= iti_deadline:
                running = False

        # ---- 3) Optional debug overlay during FIX (kept simple) ----
        if use_debug_overlay and phase == Phase.FIX:
            screen.fill((0,0,0))
            pygame.draw.rect(screen, (0,255,0), flicker_rect, 1)
            pygame.draw.rect(screen, (255,0,0), aperture_rect, 1)
            draw_fixation(screen, center_screen, (0,0,255))
            pygame.display.flip()

        # ---- 4) Frame cap (non-critical) ----
        pygame.time.delay(1)      # yield a bit
        # Or: clock.tick(120)

    return resp_key, correct, rt_ms, timed_out

# ============================ CLI / Main =====================================

def main():
    ap = argparse.ArgumentParser(description="Glass-pattern trials with IAF flicker (FSM).")
    ap.add_argument("--freq", type=float, default=10.0, help="Flicker frequency (Hz)")
    ap.add_argument("--cycles", type=int, default=15, help="Number of pulses before target")
    ap.add_argument("--trials", type=int, default=20, help="How many trials to run")
    ap.add_argument("--scaled", action="store_true", help="Use pygame.SCALED (not recommended for pixel-exact)")
    ap.add_argument("--debug", action="store_true", help="Start with the debug overlay on (F1 toggles)")
    args = ap.parse_args()

    # Enable Hi-DPI backing (macOS) and open native-pixel fullscreen by default
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "0")
    pygame.init()

    if args.scaled:
        # Convenience mode: looks centered even if desktop res ≠ requested res, but rescales pixels
        screen = pygame.display.set_mode(
            (1920,1080),
            flags=pygame.SCALED | pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE,
            vsync=1
        )
    else:
        native_w, native_h = pygame.display.list_modes()[0]
        screen = pygame.display.set_mode(
            (native_w, native_h),
            flags=pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE,
            vsync=1
        )

    W,H = screen.get_size()
    print("Window size:", (W,H), "| Desktop mode:", (pygame.display.Info().current_w, pygame.display.Info().current_h))

    task   = TaskConfig(freq_hz=args.freq, cycles=args.cycles)
    stimcf = StimulusConfig()

    # Alternate conditions ABAB… and randomize angle per trial (0° or 90°)
    conds = ["P", "T"]
    for t in range(args.trials):
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                pygame.quit(); return

        cond = conds[t % 2]
        angle = 0.0 if random.random() < 0.5 else 90.0

        # small jitter (±1–3%) around base SNR level
        snr_jitter = random.uniform(-0.03, 0.03)

        seed = random.randrange(1<<30)

        resp_key, correct, rt_ms, timed_out = run_one_trial(
            screen, task, stimcf,
            cond=cond, angle_deg=angle, snr_jitter=snr_jitter, seed=seed,
            use_debug_overlay=args.debug
        )

        # Minimal console log; plug your SQLite/LSL here if you want
        print(f"trial {t+1:03d} cond={cond} angle={angle:.1f} "
              f"resp={'L' if resp_key==pygame.K_LEFT else 'R' if resp_key==pygame.K_RIGHT else '—'} "
              f"correct={int(correct)} rt={rt_ms} timeout={int(timed_out)}")

    pygame.quit()

if __name__ == "__main__":
    main()
