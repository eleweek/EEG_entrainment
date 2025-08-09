import sys
import argparse
import gc
import time
import statistics
import math
from dataclasses import dataclass, field
from collections import deque

import pygame
from typing import Callable, Optional


REPORT_EVERY = 300

# ---------- Rolling stats ----------
@dataclass
class RollingStat:
    name: str
    window: int | None = 1000          # None ⇒ keep everything
    _buf: deque = field(default_factory=deque, repr=False)

    def add(self, value: float) -> None:
        self._buf.append(value)
        if self.window and len(self._buf) > self.window:
            self._buf.popleft()

    @property
    def n(self):      return len(self._buf)
    @property
    def mean(self):   return statistics.fmean(self._buf) if self._buf else 0.0
    @property
    def stdev(self):  return statistics.pstdev(self._buf) if self.n > 1 else 0.0
    @property
    def min(self):    return min(self._buf) if self._buf else 0.0
    @property
    def max(self):    return max(self._buf) if self._buf else 0.0

    def summary_dict(self):
        return {k: round(getattr(self, k), 3)
                for k in ('mean', 'stdev', 'min', 'max', 'n')}

# ---------- helpers ----------
def find_target_fps(frequency, target_min_refresh_rate, target_max_refresh_rate):
    result = math.floor(target_max_refresh_rate / frequency) * frequency
    if result < target_min_refresh_rate:
        raise ValueError("Frequency too low")
    if result > target_max_refresh_rate:
        raise ValueError("Frequency too high")
    return result

# ---------- NEW: refactored flicker loop ----------
def run_flicker(
    screen: pygame.Surface,
    rect: pygame.Rect,
    *,
    frequency: float,
    target_min_refresh_rate: float,
    target_max_refresh_rate: float,
    cycles: int | None = None,
    report_every: int = REPORT_EVERY,
    overlay_off_frame: Optional[Callable[[pygame.Surface], None]] = None,
):
    """
    Flicker a centered rectangle as 1-frame ON followed by N OFF frames so that
    ON-to-ON interval ≈ 1/frequency. If `cycles` is None, run indefinitely.
    Returns a dict of timing summaries.
    """
    target_fps = find_target_fps(frequency, target_min_refresh_rate, target_max_refresh_rate)
    interval = 1.0 / target_fps
    off_frames_per_each_on = int((target_fps - frequency) / frequency)

    frame_count = 0
    pulses_emitted = 0
    rectangle_on = False

    # Clear once
    screen.fill((0, 0, 0))
    pygame.display.flip()

    # High-precision time anchors
    last_draw_time = time.perf_counter()
    previous_draw_time = last_draw_time
    post_flip_time = last_draw_time
    previous_post_flip_time = last_draw_time
    start_time = last_draw_time

    # Rolling stats
    flip_ms   = RollingStat('flip_ms',  50 * max(1, off_frames_per_each_on))
    err_pre   = RollingStat('err_pre',  50 * max(1, off_frames_per_each_on))
    err_post  = RollingStat('err_post', 50 * max(1, off_frames_per_each_on))

    flip_on_ms   = RollingStat('flip_ms_on',  50)
    err_on_pre   = RollingStat('err_pre_on',  50)
    err_on_post  = RollingStat('err_post_on', 50)

    # Main loop
    while True:
        # Quit handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return {
                    "flip_ms": flip_ms.summary_dict(),
                    "err_pre": err_pre.summary_dict(),
                    "err_post": err_post.summary_dict(),
                    "flip_ms_on": flip_on_ms.summary_dict(),
                    "err_on_pre": err_on_pre.summary_dict(),
                    "err_on_post": err_on_post.summary_dict(),
                }
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return {
                    "flip_ms": flip_ms.summary_dict(),
                    "err_pre": err_pre.summary_dict(),
                    "err_post": err_post.summary_dict(),
                    "flip_ms_on": flip_on_ms.summary_dict(),
                    "err_on_pre": err_on_pre.summary_dict(),
                    "err_on_post": err_on_post.summary_dict(),
                }

        previous_on = rectangle_on
        # 1 frame ON, then N OFF
        # starting from black frames as the first ones sometimes get dropped
        rectangle_on = (frame_count % (off_frames_per_each_on + 1) == off_frames_per_each_on)
        if rectangle_on:
            pulses_emitted += 1

        # Draw
        if rectangle_on:
            pygame.draw.rect(screen, (255, 255, 255), rect)
        else:
            pygame.draw.rect(screen, (0, 0, 0), rect)
            if overlay_off_frame is not None:
                overlay_off_frame(screen)

        # Drift-free target time for next frame
        next_frame_time = start_time + (frame_count + 1) * interval
        current_time = time.perf_counter()
        delay = next_frame_time - current_time

        if delay > 0:
            if delay > 0.001:
                time.sleep(delay * 0.75)           # coarse sleep
            while time.perf_counter() < next_frame_time:
                pass                                # fine busy-wait

        previous_draw_time = last_draw_time
        last_draw_time = time.perf_counter()

        pygame.display.flip()

        previous_post_flip_time = post_flip_time
        post_flip_time = time.perf_counter()

        # Metrics
        flip_duration_ms = (post_flip_time - last_draw_time) * 1000.0
        actual_interval_pre  = last_draw_time - previous_draw_time
        actual_interval_post = post_flip_time - previous_post_flip_time
        timing_error_pre_ms  = (actual_interval_pre  - interval) * 1000.0
        timing_error_post_ms = (actual_interval_post - interval) * 1000.0

        flip_ms.add(flip_duration_ms)
        err_pre.add(timing_error_pre_ms)
        err_post.add(timing_error_post_ms)

        if rectangle_on:
            flip_on_ms.add(flip_duration_ms)
            err_on_pre.add(timing_error_pre_ms)
            err_on_post.add(timing_error_post_ms)

        if flip_duration_ms > 3.5:
            if rectangle_on or previous_on:
                print(f'⚠️  Flip too long: {flip_duration_ms:.3f} ms | '
                    f'Frame {frame_count} | {"ON" if rectangle_on else "OFF"} | '
                    f'Err pre {timing_error_pre_ms:.3f} ms | Err post {timing_error_post_ms:.3f} ms')

        if frame_count and frame_count % report_every == 0:
            print(f'\n— summary over last {flip_ms.n} frames —')
            for stat in (flip_ms, err_pre, err_post):
                s = stat.summary_dict()
                print(f'Total {stat.name:10s}: mean={s["mean"]:.3f} ms  '
                      f'std={s["stdev"]:.3f}  min={s["min"]:.3f}  max={s["max"]:.3f}')
            print()
            for stat in (flip_on_ms, err_on_pre, err_on_post):
                s = stat.summary_dict()
                print(f'On    {stat.name:10s}: mean={s["mean"]:.3f} ms  '
                      f'std={s["stdev"]:.3f}  min={s["min"]:.3f}  max={s["max"]:.3f}')
            print("\n")

        frame_count += 1

        # Stop when enough pulses (cycles) have been emitted
        # And we are not on the flash (so it's only for a brief period)
        if cycles is not None and not rectangle_on and pulses_emitted >= cycles:
            return {
                "flip_ms": flip_ms.summary_dict(),
                "err_pre": err_pre.summary_dict(),
                "err_post": err_post.summary_dict(),
                "flip_ms_on": flip_on_ms.summary_dict(),
                "err_on_pre": err_on_pre.summary_dict(),
                "err_on_post": err_on_post.summary_dict(),
            }

# ---------- CLI main (keeps existing behavior) ----------
def main():
    parser = argparse.ArgumentParser(description="Flicker. Expects a monitor with a variable refresh rate.")
    parser.add_argument('--flicker-frequency', type=float, default=10.0)
    parser.add_argument('--target-min-refresh-rate', type=float, default=120.0,
                        help="≥20–25% higher than your min monitor refresh to allow draw jitter")
    parser.add_argument('--target-max-refresh-rate', type=float, default=120.0,
                        help="20–25% lower than your max monitor refresh to allow draw jitter")
    parser.add_argument('--cycles', type=int, default=None,
                        help="Number of ON pulses to present (None = run forever)")
    args = parser.parse_args()

    # We currently don't want the garbage collector to run
    gc.disable()

    WIDTH, HEIGHT = 1920, 1080
    SQUARE_SIDE = 600

    pygame.init()
    screen = pygame.display.set_mode(
        (WIDTH, HEIGHT),
        flags=pygame.SCALED | pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE,
        vsync=1
    )

    rect = pygame.Rect(
        (WIDTH - SQUARE_SIDE) // 2,
        (HEIGHT - SQUARE_SIDE) // 2,
        SQUARE_SIDE, SQUARE_SIDE
    )

    stats = run_flicker(
        screen, rect,
        frequency=args.flicker_frequency,
        target_min_refresh_rate=args.target_min_refresh_rate,
        target_max_refresh_rate=args.target_max_refresh_rate,
        cycles=args.cycles,            # None = infinite (old behavior)
        report_every=REPORT_EVERY
    )

    # If we exit the loop cleanly or after N cycles, print final stats
    if stats:
        print("\nFinal timing summaries:")
        for k, v in stats.items():
            print(f"{k:12s}: {v}")

    pygame.quit()

if __name__ == "__main__":
    main()
