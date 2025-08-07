# glass_task_fsm.py
import enum, math, random, time, pygame

from flicker import run_flicker                 # your refactored pulse train
from glass   import draw_glass                  # your Glass generator

# ---------- helpers ----------------------------------------------------------
def wait_exact(seconds: float):
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < seconds:
        pygame.event.pump()                     # keep window responsive

def draw_fix(screen, c, col=(160,160,160)):
    x,y=c 
    s=12
    w=2
    pygame.draw.line(screen,col,(x-s,y),(x+s,y),w); pygame.draw.line(screen,col,(x,y-s),(x,y+s),w)

def glyph_tick(screen,c):
    pygame.draw.lines(screen,(80,200,80),False,[(c[0]-10,c[1]+2),(c[0]-2,c[1]+10),(c[0]+12,c[1]-8)],3)

def glyph_cross(screen,c):
    pygame.draw.line(screen,(200,80,80),(c[0]-10,c[1]-10),(c[0]+10,c[1]+10),3)
    pygame.draw.line(screen,(200,80,80),(c[0]-10,c[1]+10),(c[0]+10,c[1]-10),3)

# ---------- finite-state machine --------------------------------------------
class Phase(enum.Enum):
    FIX   = 0
    FLICK = 1
    DELAY = 2
    STIM  = 3
    RESP  = 4
    FB    = 5

def run_trial(screen, ap_rect, flick_rect,
              *, freq_hz, cycles, delay_cycles, stim_ms,
              angle_deg, snr, density, shift_px, dot_r_px, seed):
    W,H   = screen.get_size()
    center= (W//2,H//2)

    # pre-render Glass during previous ITI (we do it here for brevity)
    stim_surf = pygame.Surface(ap_rect.size)
    draw_glass(stim_surf, center=(ap_rect.w//2,ap_rect.h//2), size=ap_rect.w,
               angle_deg=angle_deg, snr=snr, density=density,
               shift=shift_px, dot_r=dot_r_px, handed="cw", seed=seed)

    # determine ground-truth key
    gt_left = (angle_deg == 0.0)          # concentric→LEFT, radial→RIGHT

    phase      = Phase.FIX
    stim_on_t  = None
    resp_dead  = None
    feedback_t = None
    resp_key   = None; correct=False; timed_out=False; rt_ms=-1

    clock = pygame.time.Clock()
    run   = True
    while run:
        # ------------ 1. pump events once/frame -------------
        events = pygame.event.get()
        for e in events:
            if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                pygame.quit(); raise SystemExit
            if phase==Phase.RESP and e.type==pygame.KEYDOWN and e.key in (pygame.K_LEFT,pygame.K_RIGHT):
                resp_key = e.key
                rt_ms = int((time.perf_counter()-stim_on_t)*1000)
                correct = ( (resp_key==pygame.K_LEFT)==gt_left )
                timed_out=False
                phase   = Phase.FB
                feedback_t = time.perf_counter()

        # ------------ 2. state logic ------------------------
        now = time.perf_counter()

        if phase==Phase.FIX:
            screen.fill((0,0,0)); draw_fix(screen, center)
            phase = Phase.FLICK

        elif phase==Phase.FLICK:
            run_flicker(screen, flick_rect, frequency=freq_hz,
                        target_min_refresh_rate=120, target_max_refresh_rate=120,
                        cycles=cycles, report_every=10_000)
            phase = Phase.DELAY

        elif phase==Phase.DELAY:
            wait_exact(delay_cycles/freq_hz)
            phase     = Phase.STIM

        elif phase==Phase.STIM:
            screen.fill((0,0,0))
            screen.blit(stim_surf, ap_rect.topleft)
            pygame.display.flip()
            stim_on_t = time.perf_counter()
            resp_dead = stim_on_t + 1.300      # 1.3 s window
            phase     = Phase.RESP

        elif phase==Phase.RESP:
            # keep stimulus up for full 200 ms
            if now - stim_on_t >= stim_ms/1000:
                screen.fill((0,0,0))
                draw_fix(screen, center)
                pygame.display.flip()

            if now >= resp_dead and resp_key is None:
                timed_out=True; correct=False; rt_ms=-1
                phase = Phase.FB
                feedback_t = now

        elif phase==Phase.FB:
            screen.fill((0,0,0)); draw_fix(screen, center)
            if not timed_out:
                (glyph_tick if correct else glyph_cross)(screen, center)
            pygame.display.flip()
            if now - feedback_t >= 0.100:      # 100 ms feedback
                run = False                       # trial finished

        # ------------ 3. flip already done where needed -----
        clock.tick(120)                        # cap to 120 Hz

    return resp_key, correct, rt_ms, timed_out

# ---------- demo main --------------------------------------------------------
def main():
    pygame.init()
    W,H = 1920,1080
    screen = pygame.display.set_mode((W,H),
        flags=pygame.SCALED|pygame.FULLSCREEN|pygame.DOUBLEBUF|pygame.HWSURFACE, vsync=1)

    flick_side = 600
    ap_side = 412
    flick_r = pygame.Rect((W-flick_side)//2,(H-flick_side)//2,flick_side,flick_side)
    ap_r    = pygame.Rect((W-ap_side)//2,(H-ap_side)//2,ap_side,ap_side)


    conds = ["P","T"]
    trial=0
    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                pygame.quit(); return
        cond  = conds[trial%2]
        delay = random.choice([1,2,3]) + (0.5 if cond=="T" else 0.0)
        angle = 0.0 if random.random()<0.5 else 90.0
        run_trial(screen, ap_r, flick_r,
                  freq_hz=10.0, cycles=15, delay_cycles=delay, stim_ms=200,
                  angle_deg=angle, snr=0.24, density=0.03,
                  shift_px=14, dot_r_px=1, seed=random.randrange(1<<30))
        trial += 1
        pygame.time.delay(1200)        # ITI, TODO: jitter

if __name__=="__main__":
    main()
