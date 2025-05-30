# EEG entrainment study replication

TODO: fill in the project description section

## Replication Hardware

### Alexander

1. EEG Headset: OpenBCI Ultracortex "Mark IV" EEG Headset with 8-channel Cyton Board and Active Electrodes
2. Monitor: LG UltraGear 24GQ50F 1920×1080 with AMD FreeSync (AMD's VSync) 48..165 Hz. Available refresh rate: 165 Hz, 144 Hz, 120 Hz, 100 Hz, 60 Hz, 50 Hz.

### Andrew

TODO

## Flicker code

Currently the repository contains multiple versions of flicker code, each with its own pros and cons. It's unfortunately tricky to generate flicker with a stable rate, so we experimented with different ways of doing it.

All of the scripts currently have the same drawback: the physical size of the flash doesn't necessarily match the target physical size (7.9o × 7.9o (2.3 × 2.3 arc min2 per dot)). We'll implement calculating the physical flash size once the actual flash is stable.

Note that the scripts aren't standardised. In particular, some scripts take monitor hz as their arguments, but most take the target refresh rate. Each of the scripts is an experiment, and we'll likely end up with only one working version.

1. `python scripts/flash_light.py <monitor hz>` . Pygame-based code that flickers in a window. Use left and right to adjust the frequency. The script stops after 1000 screen updates (500 flash cycles) and shows a chart with the frame update time distribution
2. `python scripts/flash_light_simple2.py <flicker hz>`. Pygame-based code that attempts to utilize VSync for the exact perfect flicker rate. Currently it's unclear how well it works (just a guess: it doesn't). It attempts to find a target monitor refresh rate between 48 Hz and 165 Hz, then it draws X "off" frames and 1 "on" frame where X is computed based on the target refresh rate and the desired flicker frequency. The script also prints the actual FPS and the target FPS to the terminal allowing to estimate how precise flicker is. Uses `pygame.time.Clock()` for timing
3. `python3 flash_light_simple2_better_loop.py <flicker hz>`. A version of the previous code that uses `time.sleep()` to sleep for 75% of the inter-frame interval. Then busy-waits (spinlock-style) in a `while` loop. This produces a more stable FPS. Additionally prints more debug information on how long `flip()` takes plus target interval and the actual delay.
4. `python3 flash_light_simple2_pyglet.py <flicker hz>`. A version of the previous two scripts (i.e. it also attempts to vsync to a dynamically computed refresh rate) but it uses pyglet instead of pygame. Fullscreen, shows FPS, different framework. After throwing the code into Claude Opus 4, it says that the there is a far synchronization flaw between `update()` and `on_draw()`.
5. `python3 flash_light_metal3.py`. Uses Apple Metal API for GPU-accelerated low-level rendering. It currently hardcodes the flicker rate to be 10Hz and the monitor refresh rate to be 60 Hz. Single-frame flashing (as the study requires). For performance-monitoring, aggregates FPS over 60 and 300 frames to allow estimating how stable the flicker is
6. `python3 flash_light_metal4.py`. Another script that uses Metal. Also AI-generated, simplified architecture. It also uses simplified flashing: 50% on, 50% off (wouldn't work for the study but potentially useful for debugging)

## Display testing scripts

1. `python scripts/fps_test.py`. Tests if a monitor can actually achieve a target refresh rate (hardcoded as 117 Hz) on a monitor. It does call `flip()` but it doesn't do any flicker. Prints expected FPS and actual FPS (computed via `clock.get_fps()`). It's unclear if this script is particularly useful.

2. `python scripts/vsync_test.py`. Full-screen Vsync-toggle test both `pygame_gui` and `pygame`. Press Enter to toggle on/off Vsync to see how it'd affect the FPS that's displayed in the center of the screen.

3. `python scripts/vsync_test.py`. Full-screen Vsync-toggle test that only uses `pygame`, an updated version of the previous script.
