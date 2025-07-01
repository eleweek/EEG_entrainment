#!/usr/bin/env python3

import time
import pygame
import sys
import argparse
from math import floor
import gc

gc.disable()   # optional – leave it disabled if you measured GC jitters

def main() -> None:
    parser = argparse.ArgumentParser(description="Flash Lights")
    parser.add_argument("--frequency",              type=float, default=10.0,
                        help="Flash frequency in Hz (white-on frames / second)")
    parser.add_argument("--max-monitor-frequency",  type=float, default=60.0,
                        help="Max refresh to target when deriving an integer multiple")
    args = parser.parse_args()

    frequency     = args.frequency
    max_frequency = args.max_monitor_frequency     # user override (≤ panel max)

    WIDTH, HEIGHT   = 1920, 1080
    SQUARE_SIDE     = 600
    BLACK, WHITE    = (0, 0, 0), (255, 255, 255)

    # ------------------------------------------------------------------
    #  Initialise Pygame
    # ------------------------------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT),
        flags = pygame.SCALED | pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE,
        vsync = 1 # *** VSYNC ON – tear-free VRR ***
    )
    clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    #  Derive a frame-rate that’s an integer multiple of the flash rate
    # ------------------------------------------------------------------
    def find_target_fps(freq: float, *, lo: int = 48, hi: int = 165) -> int:
        result = floor(hi / freq) * freq
        if result < lo:
            raise ValueError("Frequency too low for VRR range")
        if result > hi:
            raise ValueError("Frequency too high for monitor")
        return int(result)

    target_fps = find_target_fps(frequency, hi=max_frequency)
    interval   = 1.0 / target_fps                     # ideal frame interval (s)
    off_frames_per_each_on = int((target_fps - frequency) / frequency)

    # ------------------------------------------------------------------
    #  Prepare geometry
    # ------------------------------------------------------------------
    rect        = pygame.Rect(0, 0, SQUARE_SIDE, SQUARE_SIDE)
    rect.center = (WIDTH // 2, HEIGHT // 2)

    # ------------------------------------------------------------------
    #  Main loop
    # ------------------------------------------------------------------
    frame_count         = 0
    previous_flip_time  = time.perf_counter()

    screen.fill(BLACK)                  # start black

    while True:
        # -- event pump --------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                sys.exit()

        # -- draw --------------------------------------------------------
        rectangle_on = frame_count % (off_frames_per_each_on + 1) == 0

        screen.fill(BLACK)
        if rectangle_on:
            pygame.draw.rect(screen, WHITE, rect)

        # -- present & measure ------------------------------------------
        flip_start      = time.perf_counter()
        pygame.display.flip()                         # blocks until v-blank
        post_flip_time  = time.perf_counter()
        flip_duration   = (post_flip_time - flip_start) * 1000.0  # ms

        actual_interval = post_flip_time - previous_flip_time
        previous_flip_time = post_flip_time

        actual_fps   = 1.0 / actual_interval if actual_interval > 0 else 0.0
        timing_error = (actual_interval - interval) * 1000.0                # ms

        print(
            f'{"ON " if rectangle_on else "OFF"}: '
            f'Actual FPS: {actual_fps:6.2f} | '
            f'Target FPS: {target_fps:6.2f} | '
            f'Error: {timing_error:+7.3f} ms | '
            f'Frame: {frame_count:6d} | '
            f'Flip: {flip_duration:6.3f} ms'
        )

        frame_count += 1
        clock.tick_busy_loop(target_fps)   # keeps us early for the next v-blank


if __name__ == "__main__":
    main()
