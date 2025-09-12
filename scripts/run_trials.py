# run_trials.py
from __future__ import annotations
import argparse, enum, random, time, os, sqlite3, hashlib, json, uuid, statistics
from dataclasses import dataclass
from typing import Optional

import pygame

from pylsl import StreamInfo, StreamOutlet, local_clock


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
    flicker_side_px: int = 412            # square pulse patch
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

FIX_COLOR = (120, 120, 120)   # neutral mid-gray

def draw_fixation_dot(screen: pygame.Surface, center: tuple[int,int], radius_px=5, color=FIX_COLOR):
    pygame.draw.circle(screen, color, center, radius_px)

def glyph_tick(screen, c):
    pygame.draw.lines(screen,(80,200,80),False,[(c[0]-10,c[1]+2),(c[0]-2,c[1]+10),(c[0]+12,c[1]-8)],3)

def glyph_cross(screen, c):
    pygame.draw.line(screen,(200,80,80),(c[0]-10,c[1]-10),(c[0]+10,c[1]+10),3)
    pygame.draw.line(screen,(200,80,80),(c[0]-10,c[1]+10),(c[0]+10,c[1]-10),3)

# ============================ Argparse validators =============================

def parse_cond_seq(value: str) -> str:
    v = (value or "").strip().upper()
    if not v:
        raise argparse.ArgumentTypeError("--cond-seq must be a non-empty string of P/T, e.g. 'PTTP'")
    invalid = [ch for ch in v if ch not in ("P", "T")]
    if invalid:
        raise argparse.ArgumentTypeError("--cond-seq may only contain 'P' and 'T'")
    return v

# ============================ Break screen ====================================

def _render_text_lines(screen: pygame.Surface, lines: list[str], *, color=(230,230,230)):
    width, height = screen.get_size()
    pygame.font.init()
    # Scale font size with height
    title_font = pygame.font.SysFont(None, max(36, height // 18))
    body_font  = pygame.font.SysFont(None, max(28, height // 28))

    rendered = []
    for idx, text in enumerate(lines):
        font = title_font if idx == 0 else body_font
        surf = font.render(text, True, color)
        rendered.append(surf)

    # Compute block size (max width among lines; total height with spacing)
    line_spacing = 12
    total_h = sum(s.get_height() for s in rendered) + (len(rendered)-1) * line_spacing
    block_w = max((s.get_width() for s in rendered), default=0)

    # Center the block while keeping lines left-aligned within the block
    x = (width - block_w) // 2
    y = (height - total_h) // 2

    for surf in rendered:
        screen.blit(surf, (x, y))
        y += surf.get_height() + line_spacing

def show_block_break_screen(
    screen: pygame.Surface,
    *,
    block_number: int,
    total_blocks: int,
    condition: str,
    trials_in_block: int,
    num_correct: int,
    num_timeouts: int,
    mean_rt_ms: float | None,
):
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); raise SystemExit
                if e.key in (pygame.K_SPACE, pygame.K_RETURN):
                    running = False

        screen.fill((0,0,0))

        accuracy_den = max(1, trials_in_block - num_timeouts)
        accuracy_pct = 100.0 * (num_correct / accuracy_den)
        mean_rt_text = (f"{mean_rt_ms:.0f} ms" if mean_rt_ms is not None else "—")

        lines = [
            f"{block_number}/{total_blocks} blocks completed",
            f"Condition: {condition}    Trials: {trials_in_block}",
            f"Accuracy: {accuracy_pct:.1f}%    Timeouts: {num_timeouts}",
            f"Mean RT: {mean_rt_text}",
            "",
            "Take a short break. Blink, relax your eyes.",
            "Press SPACE when you're ready to continue."
        ]
        _render_text_lines(screen, lines)
        pygame.display.flip()
        pygame.time.delay(50)

# ============================ Geometry =======================================

def build_centered_rect(center: tuple[int,int], side: int) -> pygame.Rect:
    cx, cy = center
    return pygame.Rect(cx - side//2, cy - side//2, side, side)

# ============================ SQLite helpers =================================

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS session(
  id TEXT PRIMARY KEY,
  participant_id TEXT,
  start_ts REAL,
  iaf_hz REAL,
  flicker_freq_hz REAL,
  notes TEXT
);
CREATE TABLE IF NOT EXISTS stimulus(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  hash TEXT UNIQUE,
  file_path TEXT,
  angle_deg REAL,
  snr_level REAL,
  snr_jitter REAL,
  density REAL,
  shift_px REAL,
  dot_r_px INTEGER,
  handed TEXT,
  seed INTEGER
);
CREATE TABLE IF NOT EXISTS trial(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  trial_index INTEGER,
  cond TEXT,
  block INTEGER,
  delay_cycles REAL,
  angle_deg REAL,
  snr_level REAL,
  snr_jitter REAL,
  seed INTEGER,
  resp_key TEXT,
  correct INTEGER,
  rt_ms INTEGER,
  timed_out INTEGER,
  stim_id INTEGER,
  ts_onset REAL,
  ts_resp REAL
);
"""

def open_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, isolation_level=None)
    db.executescript(SCHEMA)
    return db

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def ensure_png(surface: pygame.Surface, out_dir: str, h: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{h}.png")
    if not os.path.exists(fn):
        pygame.image.save(surface, fn)
    return fn

def upsert_stimulus(db: sqlite3.Connection, meta: dict) -> int:
    db.execute("""INSERT OR IGNORE INTO stimulus(hash,file_path,angle_deg,snr_level,snr_jitter,
                  density,shift_px,dot_r_px,handed,seed)
                  VALUES(:hash,:file_path,:angle_deg,:snr_level,:snr_jitter,
                         :density,:shift_px,:dot_r_px,:handed,:seed)""", meta)
    row = db.execute("SELECT id FROM stimulus WHERE hash=?", (meta["hash"],)).fetchone()
    return row[0]

def insert_trial(db: sqlite3.Connection, row: dict) -> int:
    db.execute("""INSERT INTO trial(session_id,trial_index,cond,block,delay_cycles,angle_deg,
                 snr_level,snr_jitter,seed,resp_key,correct,rt_ms,timed_out,stim_id,ts_onset,ts_resp)
                 VALUES(:session_id,:trial_index,:cond,:block,:delay_cycles,:angle_deg,
                        :snr_level,:snr_jitter,:seed,:resp_key,:correct,:rt_ms,:timed_out,:stim_id,:ts_onset,:ts_resp)""", row)
    return db.execute("SELECT last_insert_rowid()").fetchone()[0]

# ============================ LSL helpers ====================================

def make_marker_outlet(stream_name="GlassMarkers") -> Optional[StreamOutlet]:
    if StreamInfo is None or StreamOutlet is None:
        print("[LSL] pylsl not available; continuing without LSL.")
        return None
    info = StreamInfo(name=stream_name, type='Markers',
                      channel_count=1, nominal_srate=0,
                      channel_format='string',
                      source_id=f'glass-{uuid.uuid4()}')
    return StreamOutlet(info)

def push_marker(outlet: Optional[StreamOutlet], ev: str, **fields) -> float:
    ts = local_clock()
    if outlet is not None:
        payload = {"ev": ev, "ts": ts}
        payload.update(fields)
        outlet.push_sample([json.dumps(payload)], timestamp=ts)
    return ts

# ============================ Trial runner (FSM) ==============================

def run_one_trial(
    screen: pygame.Surface,
    task: TaskConfig,
    stimcfg: StimulusConfig,
    *,
    trial_index: int,
    block: int,
    session_id: str,
    db: Optional[sqlite3.Connection],
    stim_out_dir: Optional[str],
    outlet: Optional[StreamOutlet],
    cond: str,                     # "P" or "T"
    angle_deg: float,              # 0.0 / 90.0 or anything 0..90
    snr_jitter: float,             # e.g., uniform(-0.03, +0.03)
    seed: int,
    use_debug_overlay: bool=False
):
    W, H = screen.get_size()
    center_screen = (W//2, H//2)

    # Build rects from a single center source of truth
    flicker_rect  = build_centered_rect(center_screen, stimcfg.flicker_side_px)
    aperture_rect = build_centered_rect(center_screen, stimcfg.aperture_side_px)

    # Pre-render the Glass stimulus during setup
    stim = pygame.Surface(aperture_rect.size)
    stim.fill((0,0,0))
    snr_trial = stimcfg.snr_level + snr_jitter
    draw_glass(
        stim, center=(aperture_rect.w//2, aperture_rect.h//2), size=aperture_rect.w,
        angle_deg=angle_deg, snr=snr_trial, density=stimcfg.density,
        shift=stimcfg.shift_px, dot_r=stimcfg.dot_r_px, handed=stimcfg.handed, seed=seed
    )

    # Hash & (later) save PNG
    rgb_bytes = pygame.image.tostring(stim, "RGB")
    stim_hash = sha256_bytes(rgb_bytes)
    stim_path = os.path.join(stim_out_dir, f"{stim_hash}.png") if stim_out_dir else ""

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

    # Draw baseline fixation
    screen.fill((0,0,0))
    draw_fixation_dot(screen, center_screen)
    pygame.display.flip()

    # LSL markers: trial header
    push_marker(outlet, "trial_start", trial=trial_index, cond=cond,
                angle=angle_deg, snr_level=stimcfg.snr_level, snr_jitter=snr_jitter,
                seed=seed, delay_cycles=delay_cycles)

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
                if e.key in (pygame.K_LEFT, pygame.K_RIGHT) and response_enabled and phase in (Phase.STIM, Phase.RESP):
                    resp_key = e.key
                    rt_ms = int((time.perf_counter() - stim_on_t) * 1000)
                    said_left = (resp_key == pygame.K_LEFT)
                    correct = (said_left == is_left_correct)
                    timed_out = False
                    response_enabled = False
                    push_marker(outlet, "response", trial=trial_index,
                                resp=('L' if said_left else 'R'), correct=bool(correct), rt_ms=rt_ms)
                    phase = Phase.FB
                    fb_deadline = time.perf_counter() + task.feedback_ms/1000.0

        now = time.perf_counter()

        # ---- 2) Phase logic ----
        if phase == Phase.FIX:
            phase = Phase.FLICK

        elif phase == Phase.FLICK:
            push_marker(outlet, "flicker_start", trial=trial_index, freq=task.freq_hz, cycles=task.cycles)
            # keep fixation dot overlayed on OFF frames (optional)
            def _overlay_off(surf: pygame.Surface):
                draw_fixation_dot(surf, center_screen)
            run_flicker(screen, flicker_rect,
                        frequency=task.freq_hz, target_min_refresh_rate=80.0, target_max_refresh_rate=125.0,
                        cycles=task.cycles, report_every=10_000, overlay_off_frame=_overlay_off)
            push_marker(outlet, "flicker_end", trial=trial_index)
            phase = Phase.DELAY

        elif phase == Phase.DELAY:
            push_marker(outlet, "delay_start", trial=trial_index, delay_cycles=delay_cycles)
            target = now + (delay_cycles / task.freq_hz)
            while time.perf_counter() < target and phase == Phase.DELAY:
                pygame.event.pump()
            phase = Phase.STIM

        elif phase == Phase.STIM:
            screen.fill((0,0,0))
            screen.blit(stim, aperture_rect.topleft)
            if use_debug_overlay:
                pygame.draw.rect(screen, (0,255,0), flicker_rect, 1)
                pygame.draw.rect(screen, (255,0,0), aperture_rect, 1)
                draw_fixation_dot(screen, center_screen, color=(0,0,255))

            pygame.event.clear()                   # drop any pre-onset key presses
            ts_on_req = push_marker(outlet, "stim_onset_req", trial=trial_index,
                                    angle=angle_deg, snr=snr_trial, stim_hash=stim_hash)
            pygame.display.flip()                  # stimulus appears
            push_marker(outlet, "stim_flip_done", trial=trial_index)

            stim_on_t = time.perf_counter()
            response_enabled = True
            # total window = 200 ms + 1.3 s
            resp_deadline = stim_on_t + (task.stim_ms + task.resp_extra_ms) / 1000.0
            phase = Phase.RESP

        elif phase == Phase.RESP:
            # Keep stimulus visible for 200 ms, then blank to fixation
            if now - stim_on_t >= task.stim_ms/1000.0:
                screen.fill((0,0,0)); draw_fixation_dot(screen, center_screen)
                if use_debug_overlay:
                    pygame.draw.rect(screen, (0,255,0), flicker_rect, 1)
                    pygame.draw.rect(screen, (255,0,0), aperture_rect, 1)
                    draw_fixation_dot(screen, center_screen, color=(0,0,255))
                pygame.display.flip()
                # remain in RESP until key or deadline

            if now >= resp_deadline and resp_key is None:
                timed_out = True; correct = False; rt_ms = -1; response_enabled = False
                push_marker(outlet, "response", trial=trial_index, resp='none', correct=False, rt_ms=-1, timeout=True)
                phase = Phase.FB
                fb_deadline = now + task.feedback_ms/1000.0

        elif phase == Phase.FB:
            screen.fill((0,0,0)); draw_fixation_dot(screen, center_screen)
            if not timed_out and task.show_feedback:
                (glyph_tick if correct else glyph_cross)(screen, center_screen)
            pygame.display.flip()
            if time.perf_counter() >= fb_deadline:
                phase = Phase.ITI
                jitter = random.randint(-task.iti_jitter_ms, task.iti_jitter_ms)
                iti_deadline = time.perf_counter() + (task.iti_ms + jitter)/1000.0
                push_marker(outlet, "feedback_end", trial=trial_index)

        elif phase == Phase.ITI:
            if time.perf_counter() >= iti_deadline:
                running = False

        # Optional debug overlay during FIX
        if use_debug_overlay and phase == Phase.FIX:
            screen.fill((0,0,0))
            pygame.draw.rect(screen, (0,255,0), flicker_rect, 1)
            pygame.draw.rect(screen, (255,0,0), aperture_rect, 1)
            draw_fixation_dot(screen, center_screen, color=(0,0,255))
            pygame.display.flip()

        pygame.time.delay(1)  # yield

    # ---- After loop: save stimulus PNG & DB rows during ITI slack ----
    stim_id = None
    if db is not None:
        # save PNG if requested
        if stim_out_dir:
            stim_path = ensure_png(stim, stim_out_dir, stim_hash)
        meta = dict(hash=stim_hash, file_path=stim_path, angle_deg=angle_deg, snr_level=stimcfg.snr_level,
                    snr_jitter=snr_jitter, density=stimcfg.density, shift_px=stimcfg.shift_px,
                    dot_r_px=stimcfg.dot_r_px, handed=stimcfg.handed, seed=seed)
        stim_id = upsert_stimulus(db, meta)

        # pull last response marker time if you want (here we use local perf counter deltas)
        ts_onset = None  # we could store the ts from stim_onset_req if needed—left None here
        ts_resp  = None

        row = dict(session_id=session_id, trial_index=trial_index, cond=cond, block=block,
                   delay_cycles=delay_cycles, angle_deg=angle_deg, snr_level=stimcfg.snr_level,
                   snr_jitter=snr_jitter, seed=seed,
                   resp_key=('L' if resp_key==pygame.K_LEFT else 'R' if resp_key==pygame.K_RIGHT else None),
                   correct=int(bool(correct)), rt_ms=(rt_ms if rt_ms>=0 else None),
                   timed_out=int(bool(timed_out)), stim_id=stim_id,
                   ts_onset=ts_onset, ts_resp=ts_resp)
        insert_trial(db, row)

    push_marker(outlet, "trial_end", trial=trial_index, correct=bool(correct), timeout=bool(timed_out))
    return resp_key, correct, rt_ms, timed_out

def sample_abs_jitter(min_abs: float, max_abs: float) -> float:
    """Return a signed absolute jitter in *absolute SNR units* (e.g., ±0.01..±0.03)."""
    amp = random.uniform(min_abs, max_abs)
    return amp if random.random() < 0.5 else -amp

def main():
    ap = argparse.ArgumentParser(description="Glass-pattern trials with IAF flicker (FSM) + SQLite + LSL.")
    ap.add_argument("--participant", type=str, required=True)
    ap.add_argument("--session", type=str, default=None, help="Session ID (default: auto)")
    ap.add_argument("--db", type=str, default="study.db", help="SQLite DB path")
    ap.add_argument("--stimdir", type=str, default="stimuli", help="Directory to save stimulus PNGs")

    ap.add_argument("--iaf", type=float, required=True, help="IAF (in Hz) to store in session metadata")
    ap.add_argument("--freq", type=float, required=True, help="Flicker frequency (Hz)")
    ap.add_argument("--cycles", type=int, default=15, help="Number of pulses before target")

    ap.add_argument("--blocks", type=int, default=8, help="Number of blocks")
    ap.add_argument("--tperblock", type=int, default=100, help="Trials per block")
    ap.add_argument("--snr", type=float, default=0.24, help="Base SNR (signal proportion)")
    ap.add_argument("--jitter-min", type=float, default=0.01, help="Min absolute jitter (e.g., 0.01 = 1%%)")
    ap.add_argument("--jitter-max", type=float, default=0.03, help="Max absolute jitter (e.g., 0.03 = 3%%)")

    # Make blinding mutually exclusive with explicit condition scheduling
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--condition", choices=["alt", "P", "T", "seq"], default="alt",
                    help="Block schedule: alt (alternate P/T), P (all peak), T (all trough), seq (use --cond-seq). Mutually exclusive with --blind-key.")
    mx.add_argument("--blind-key", type=str, default=None,
                    help="Enable blinding: hash this secret string to pick session condition (applies to all blocks). Mutually exclusive with --condition/--cond-seq.")
    ap.add_argument("--cond-seq", type=parse_cond_seq, default=None,
                    help="Sequence of conditions for blocks when --condition=seq, e.g. 'PTTP' (repeats as needed)")
    ap.add_argument("--blind-session", type=int, choices=[1,2], default=1,
                    help="If using --blind-key, set 1 for first run and 2 for second run (flips condition)")

    ap.add_argument("--nofeedback", action="store_true", help="Disable feedback (Session 2 style)")
    ap.add_argument("--lsl", action="store_true", help="Enable LSL marker stream")
    ap.add_argument("--debug", action="store_true", help="Start with the debug overlay on (F1 toggles)")
    args = ap.parse_args()

    # Explicit incompatibility: --cond-seq cannot be combined with blinding
    # In principle, this shouldn't be difficult to make this work, but it'd require extra testing
    # Prohibiting explicitly for simplicity
    if args.blind_key and args.cond_seq:
        ap.error("--cond-seq cannot be used together with --blind-key. Choose either blinding or explicit scheduling.")

    # SQLite setup
    db = open_db(args.db)
    session_id = args.session or f"ses-{int(time.time())}"
    db.execute("INSERT OR IGNORE INTO session(id,participant_id,start_ts,iaf_hz,flicker_freq_hz,notes) VALUES(?,?,?,?,?,?)",
               (session_id, args.participant, time.time(), args.iaf, args.freq, ""))

    # LSL
    outlet = make_marker_outlet() if args.lsl else None
    if outlet is None and args.lsl:
        raise Exception("LSL ERROR: --lsl requested but pylsl not available; continuing without markers.")

    # Pygame display
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "0")
    pygame.init()
    pygame.key.set_repeat(0)

    screen = pygame.display.set_mode(
        (1920,1080),
        flags=pygame.SCALED | pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE,
        vsync=1
    )

    W,H = screen.get_size()
    print("Window size:", (W,H), "| Desktop mode:", (pygame.display.Info().current_w, pygame.display.Info().current_h))

    # config objects
    task   = TaskConfig(freq_hz=args.freq, cycles=args.cycles, show_feedback=not args.nofeedback)
    stimcf = StimulusConfig()
    stimcf.snr_level = args.snr  # <-- fixed SNR from CLI

    # --- Determine block condition schedule ---
    def _resolve_blind_cond(key: str, session_num: int) -> str:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        last_bit = int(h[-1], 16) & 1
        base = "T" if last_bit == 1 else "P"
        if session_num == 2:
            base = ("P" if base == "T" else "T")
        return base

    if args.blind_key:
        session_cond = _resolve_blind_cond(args.blind_key, args.blind_session)
        block_conds = [session_cond] * args.blocks
        print(f"[BLIND] key='{args.blind_key}' session={args.blind_session} -> cond={session_cond}; applied to all {args.blocks} blocks")
    else:
        if args.condition == "seq":
            seq_raw = (args.cond_seq or "")
            if not seq_raw:
                raise SystemExit("--condition=seq requires --cond-seq like 'PTTP'")
            seq = list(seq_raw)
            block_conds = [seq[i % len(seq)] for i in range(args.blocks)]
        elif args.condition in ("P","T"):
            block_conds = [args.condition] * args.blocks
        else:  # alt
            block_conds = [("P" if (b % 2) == 0 else "T") for b in range(args.blocks)]

    print("Block condition schedule:", " ".join(block_conds))

    # Run blocks
    for b in range(args.blocks):
        cond = block_conds[b]
        display_cond = (cond if not args.blind_key else f"BLINDED {args.blind_session}")

        # Equal angles per block (half 0°, half 90°), shuffled
        n = args.tperblock
        angles = [0.0]*(n//2) + [90.0]*(n - n//2)
        random.shuffle(angles)

        print(f"\n=== Block {b+1}/{args.blocks}  cond={cond}  trials={n}  base SNR={stimcf.snr_level:.3f} "
              f"jitter=±{int(args.jitter_min*100)}–{int(args.jitter_max*100)}% ===")

        # LSL marker: block start
        push_marker(outlet, "block_start", block=b+1, total_blocks=args.blocks, cond=cond, trials=n)

        # Per-block accumulators
        num_correct_block = 0
        num_timeouts_block = 0
        rts_correct_block: list[int] = []

        for i in range(n):
            # quick escape at block level
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                    pygame.quit(); return

            angle = angles[i]

            # jitter like in the paper: per-trial absolute ±1–3% (default)
            snr_jitter = sample_abs_jitter(args.jitter_min, args.jitter_max)

            seed = random.randrange(1<<30)

            trial_index = b * args.tperblock + i + 1
            resp_key, correct, rt_ms, timed_out = run_one_trial(
                screen, task, stimcf,
                trial_index=trial_index,
                block=b+1,
                session_id=session_id,
                db=db,
                stim_out_dir=os.path.join(args.stimdir, session_id) if args.stimdir else None,
                outlet=outlet,
                cond=cond, angle_deg=angle, snr_jitter=snr_jitter, seed=seed,
                use_debug_overlay=args.debug
            )

            print(f"trial {trial_index:03d} block={b+1} cond={cond} angle={angle:.0f} "
                  f"resp={'L' if resp_key==pygame.K_LEFT else 'R' if resp_key==pygame.K_RIGHT else '—'} "
                  f"correct={int(correct)} rt={rt_ms} timeout={int(timed_out)}")

            # Update per-block stats
            if timed_out:
                num_timeouts_block += 1
            else:
                if correct:
                    num_correct_block += 1
                    if rt_ms is not None and rt_ms >= 0:
                        rts_correct_block.append(rt_ms)

        # Compute per-block summary
        accuracy_pct = 100.0 * num_correct_block / n
        mean_rt_ms = (statistics.fmean(rts_correct_block) if rts_correct_block else None)

        # LSL marker: block end summary
        push_marker(
            outlet, "block_end",
            block=b+1, total_blocks=args.blocks, cond=cond, trials=n,
            correct=num_correct_block, timeouts=num_timeouts_block,
            accuracy_pct=round(accuracy_pct, 2),
            mean_rt_ms=(round(mean_rt_ms, 1) if mean_rt_ms is not None else None)
        )

        # On-screen break screen
        show_block_break_screen(
            screen,
            block_number=b+1,
            total_blocks=args.blocks,
            condition=display_cond,
            trials_in_block=n,
            num_correct=num_correct_block,
            num_timeouts=num_timeouts_block,
            mean_rt_ms=mean_rt_ms,
        )

    pygame.quit()

if __name__ == "__main__":
    main()
