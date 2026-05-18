# EEG entrainment replication

This repository contains code and public data for an ACX Grants 2024 replication project on EEG entrainment and perceptual learning.

The project attempted to replicate parts of Michael et al. (2023), ["Learning at your brain's rhythm: individualized entrainment boosts learning for perceptual decisions"](https://pubmed.ncbi.nlm.nih.gov/36352510/). The original study reported that visual flicker timed to a person's individual alpha rhythm improved learning on a difficult radial-vs-concentric Glass-pattern discrimination task.

The replication code covers three main jobs:

- estimating individual alpha frequency from EEG recordings
- running the Glass-pattern behavioral task with individualized flicker
- exporting, analyzing, and plotting replication data alongside the original study data

The analysis code is the most polished part of the repository. The experiment-running code is usable, but still reflects the hardware and study setup used for this project.

## Public Data

The public replication data is COMING SOON to:

```text
published_replication_data/
```

Containing:

```text
block_accuracy.csv
sessions.csv
```

`block_accuracy.csv` has one row per participant/day/block:

```text
participant_id, cond, day_index, block, n_trials, n_correct, n_timeouts, accuracy
```

`sessions.csv` has one row per participant/day:

```text
participant_id, day_index, cond, iaf_hz, flicker_freq_hz
```

Participant IDs in the public files are scrambled public IDs such as `S01`. They do not correspond to internal study IDs or the order in the acknowledgements.

The public data intentionally does not include trial-level reaction times, exact timestamps, notes, stimulus file paths, or internal participant IDs.

## Reproducing the Analysis

Create an environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the analysis from the public export after downloading the original study's data from [Cambridge Apollo repository](<https://www.repository.cam.ac.uk/items/9d396860-7623-4804-8a51-ec6db65d3429>):

```bash
python3 scripts/analyze_data.py \
  --from-export published_replication_data \
  --original-study-data-dir original_study_data
```

Run the analysis directly from a private SQLite DB:

```bash
python3 scripts/analyze_data.py \
  --db study.db \
  --original-study-data-dir original_study_data
```

By default, DB-backed charts use public participant IDs from the mapping table. For private/debug charts with internal IDs:

```bash
python3 scripts/analyze_data.py \
  --db study.db \
  --use-internal-ids \
  --original-study-data-dir original_study_data
```

`--use-internal-ids` is invalid with `--from-export`, because exported data already contains only public IDs.

Useful analysis flags:

```text
--fit-method ols|l1
--include-only-participants p001,p002
--exclude-participants p001,p002
--n-permutations 10000
--permutation-seed 42
--charts-save-dir _generated_charts
```

## Running the Experiment

If you want to run your own replication or experiment on yourself, the task runner is:

```bash
python3 scripts/run_trials.py
```

Common arguments:

```text
--participant PARTICIPANT_ID
--session SESSION_ID
--db study.db
--stimdir stimuli
--iaf 10.2
--freq 10.2
--blocks 8
--tperblock 100
--condition P|T|alt|seq
--cond-seq PTTP
--blind-key SECRET
--blind-session 1|2
--nofeedback
--lsl
```

For the blinded replication workflow, the relevant options are `--blind-key` and `--blind-session`. The first session receives one condition and the second session receives the opposite condition; the experimenter does not need to know which condition is being run.

The task code was written for a VRR monitor setup. In this project the display was an LG UltraGear 24GQ50F 1920x1080 monitor with AMD FreeSync. The flicker timing assumptions may need adjustment on other hardware.

You can use `SDL_VIDEO_WINDOW_POS` to place the task window on a particular monitor:

```bash
SDL_VIDEO_WINDOW_POS='1920,1' python3 scripts/run_trials.py ...
```

## EEG / IAF Workflow

Display a fixation screen while recording EEG:

```bash
python3 scripts/eo_eeg_screen.py
```

Estimate alpha frequency from a recording:

```bash
python3 -m plot.xdf_to_specparam --picks O1,Oz,O2 recording.xdf
```

If necessary, generate a sliding-window alpha report for debugging or investigation:

```bash
python3 -m plot.alpha_report \
  --separate-channels \
  --picks O1,O2,Oz \
  --chunk-shift 5 \
  --chunk-duration 15 \
  recording.xdf \
  report.html
```

Live-debug an EEG stream:

```bash
python3 -m plot.EEG_rms2
```

Replay an XDF recording as an LSL stream:

```bash
python3 -m scripts.replay_xdf recording.xdf
```

## Stimuli and Flicker Utilities

Generate a single Glass-pattern stimuli:

```bash
python3 scripts/glass.py --angle 0 --snr 0.24   # Radial
python3 scripts/glass.py --angle 90 --snr 0.24  # Concentric
```

Estimate possible flicker rates for fixed refresh rates:

```bash
python3 scripts/calculate_possible_flicker_rates.py 165 144 120 100
```

Run the flicker code directly:

```bash
python3 scripts/flicker.py \
  --flicker-frequency 11 \
  --target-min-refresh-rate 60 \
  --target-max-refresh-rate 120
```

## Exporting Public Replication Data

Public export is a two-step process when starting from the private SQLite DB.

First, create a one-time public ID mapping:

```bash
python3 scripts/create_public_participant_ids.py study.db
```

To exclude participants from the public mapping:

```bash
python3 scripts/create_public_participant_ids.py study.db --exclude p001,p002
```

The script refuses to run if the mapping table already exists. This is deliberate: the mapping should be stable once public data has been generated.

Then export the public CSVs:

```bash
python3 scripts/export_replication_data.py study.db published_replication_data
```

If the mapping was created with exclusions, use the same exclusions during export:

```bash
python3 scripts/export_replication_data.py \
  study.db \
  published_replication_data \
  --exclude-participants p001,p002
```

The export script writes:

```text
published_replication_data/block_accuracy.csv
published_replication_data/sessions.csv
```

## Notes on the Original Study Data

The original study data can be downloaded from the University of Cambridge Apollo repository:

<https://www.repository.cam.ac.uk/items/9d396860-7623-4804-8a51-ec6db65d3429>

Put these files in `original_study_data/`

## Hardware Used

Replication hardware:

- EEG headset: OpenBCI Ultracortex Mark IV with an 8-channel Cyton board and ThinkPulse active electrodes
- electrodes used for analysis/setup included occipital sites such as O1, Oz, and O2
- display: LG UltraGear 24GQ50F 1920x1080 with variable refresh rate support

The code may run on other setups, but timing-sensitive parts of the task should be validated before collecting data.
