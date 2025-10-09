# glass_pattern.py
import argparse, math, random, sys
import pygame

def parse_args():
    ap = argparse.ArgumentParser(description="Glass pattern generator (pygame)")
    ap.add_argument("--angle", type=float, default=0.0,
                    help="Spiral angle in degrees (0..90). 0=radial, 90=concentric.")
    ap.add_argument("--snr", type=float, default=0.24,
                    help="Signal fraction (0..1). Default 0.24 (~24%%).")
    ap.add_argument("--density", type=float, default=0.03,
                    help="Approx. dot area density (0..1). Default 0.03 (~3%%).")
    ap.add_argument("--shift", type=float, default=14.0,
                    help="Glass shift (pixels) between dots in a dipole. Default 14 (matches run_trials).")
    ap.add_argument("--dotr", type=int, default=1,
                    help="Dot radius in pixels. Default 1 (2 px diameter; matches run_trials).")
    ap.add_argument("--size", type=int, default=412,
                    help="Aperture size in pixels (square). Default 412 (matches run_trials).")
    ap.add_argument("--handed", choices=["cw","ccw"], default="cw",
                    help="Spiral handedness for 0<angle<90. Default cw.")
    ap.add_argument("--shape", choices=["circle","square"], default="circle",
                    help="Dot shape. Default circle.")
    ap.add_argument("--fullscreen", action="store_true",
                    help="Open in fullscreen (scaled) instead of a window.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    ap.add_argument("output", nargs="?", help="If provided, save PNG to this path and exit.")
    return ap.parse_args()

def compute_num_dipoles(ap_size, dot_r, density, shape: str = "circle"):
    # density ≈ (total dot area) / aperture area  =>  N ≈ density*Area / (2*area_per_dot)
    area = ap_size * ap_size
    if shape == "square":
        # Interpret dot_r as half-side; side = 2*dot_r
        single_dot_area = (2.0 * dot_r) * (2.0 * dot_r)
    else:
        single_dot_area = math.pi * (dot_r**2)
    n = int(max(1, round((density * area) / (2.0 * single_dot_area))))
    return n

def draw_dot_square(surf, color, center_xy, half_side):
    cx_i = int(round(center_xy[0]))
    cy_i = int(round(center_xy[1]))
    hs = int(round(half_side))
    # Ensure at least 1x1
    side = max(1, 2*hs)
    rect = pygame.Rect(cx_i - hs, cy_i - hs, side, side)
    pygame.draw.rect(surf, color, rect)

def draw_glass(surface, center, size, angle_deg, snr, density, shift, dot_r, handed, seed=None, shape: str = "circle"):
    if seed is not None:
        random.seed(seed)
    w = h = size
    cx, cy = center

    # Precompute counts and margins
    N = compute_num_dipoles(size, dot_r, density, shape)
    N_signal = int(round(snr * N))
    N_noise = N - N_signal
    half_shift = shift / 2.0
    margin = int(math.ceil(half_shift + dot_r + 1))

    # Choose handedness sign (screen y grows downward, so flip for 'cw')
    # We'll define: cw = negative rotation, ccw = positive rotation (conventional screen coords).
    sign = -1.0 if handed == "cw" else 1.0
    theta = math.radians(max(0.0, min(90.0, angle_deg))) * sign

    # Helpers
    def rand_pos():
        return (random.randint(margin, w - margin - 1),
                random.randint(margin, h - margin - 1))

    def place_dipole(pos, ori_angle):
        ux = math.cos(ori_angle); uy = math.sin(ori_angle)
        x, y = pos
        p1 = (x + ux * half_shift, y + uy * half_shift)
        p2 = (x - ux * half_shift, y - uy * half_shift)
        return p1, p2

    def signal_orientation(pos):
        x, y = pos
        rx = x - cx
        ry = y - cy
        rlen = math.hypot(rx, ry)
        if rlen < 1e-6:
            # Near center: arbitrary orientation
            return random.random() * math.pi
        base = math.atan2(ry, rx)  # radial direction
        return base + theta

    # Clear
    surface.fill((0,0,0))

    # Prepare lists of dipoles (signal followed by noise)
    dipoles = []
    for _ in range(N_signal):
        p = rand_pos()
        ori = signal_orientation(p)
        dipoles.append((p, ori))
    for _ in range(N_noise):
        p = rand_pos()
        ori = random.random() * math.pi  # [0, π) is enough (undirected)
        dipoles.append((p, ori))

    random.shuffle(dipoles)

    # Draw all dots
    draw_dot_circle = pygame.draw.circle

    for (pos, ori) in dipoles:
        (x1, y1), (x2, y2) = place_dipole(pos, ori)

        # Clip to aperture: skip if either dot would lie outside
        if not (dot_r <= x1 < w - dot_r and dot_r <= y1 < h - dot_r): 
            continue
        if not (dot_r <= x2 < w - dot_r and dot_r <= y2 < h - dot_r): 
            continue

        if shape == "square":
            draw_dot_square(surface, (255,255,255), (x1, y1), dot_r)
            draw_dot_square(surface, (255,255,255), (x2, y2), dot_r)
        else:
            draw_dot_circle(surface, (255,255,255), (int(round(x1)), int(round(y1))), dot_r)
            draw_dot_circle(surface, (255,255,255), (int(round(x2)), int(round(y2))), dot_r)

def main():
    args = parse_args()
    pygame.init()
    # Headless save path if an output filename is provided
    if args.output:
        outfile = args.output
        if not outfile.lower().endswith(".png"):
            outfile += ".png"
        surface = pygame.Surface((args.size, args.size))
        draw_glass(surface, (args.size//2, args.size//2), args.size,
                   float(args.angle), args.snr, args.density, args.shift,
                   args.dotr, args.handed, args.seed, args.shape)
        pygame.image.save(surface, outfile)
        print(f"Saved: {outfile}")
        pygame.quit()
        return 0

    flags = 0
    ap_surface = None
    if args.fullscreen:
        flags |= pygame.SCALED | pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE
        # Match run_trials style: fullscreen 1920x1080, vsync on
        screen = pygame.display.set_mode((1920, 1080), flags=flags, vsync=1)
        screen_w, screen_h = screen.get_size()
        ap_size = args.size
        ap_surface = pygame.Surface((ap_size, ap_size))
    else:
        # Windowed mode: restore original behavior (square window of --size)
        screen = pygame.display.set_mode((args.size, args.size))
        screen_w, screen_h = screen.get_size()
    pygame.display.set_caption("Glass pattern")
    clock = pygame.time.Clock()

    angle = float(args.angle)
    handed = args.handed
    seed = args.seed

    def refresh(seed_override=None):
        title = f"Glass pattern | angle={angle:.1f}° ({handed})  SNR={args.snr:.2f}  density={args.density:.2f}  shift={args.shift}px  dotr={args.dotr}px  shape={args.shape}"
        pygame.display.set_caption(title)
        if ap_surface is None:
            # Direct draw into window (square)
            draw_glass(screen, (screen_w//2, screen_h//2), screen_w, angle,
                       args.snr, args.density, args.shift, args.dotr, handed,
                       seed_override if seed_override is not None else seed, args.shape)
        else:
            # Draw into square surface and center on fullscreen 1920x1080
            draw_glass(ap_surface, (ap_size//2, ap_size//2), ap_size, angle,
                       args.snr, args.density, args.shift, args.dotr, handed,
                       seed_override if seed_override is not None else seed, args.shape)
            screen.fill((0,0,0))
            topleft = ((screen_w - ap_size)//2, (screen_h - ap_size)//2)
            screen.blit(ap_surface, topleft)
        pygame.display.flip()

    refresh()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running=False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running=False
                elif event.key == pygame.K_SPACE:
                    # regenerate with fresh randomness
                    refresh(seed_override=random.randrange(1<<30))
                elif event.key == pygame.K_UP:
                    angle = min(90.0, angle + 1.0)
                    refresh()
                elif event.key == pygame.K_DOWN:
                    angle = max(0.0, angle - 1.0)
                    refresh()
                elif event.key == pygame.K_h:
                    handed = "ccw" if handed == "cw" else "cw"
                    refresh()
                elif event.key == pygame.K_s:
                    path = f"glass_{int(round(angle))}_{handed}.png"
                    pygame.image.save(screen, path)
                    print(f"Saved: {path}")

        clock.tick(60)

    pygame.quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())

