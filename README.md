# EEG entrainment study replication

This is the code for the [ACX Grants (2024)](https://www.astralcodexten.com/p/acx-grants-results-2024) EEG entrainment project replicating the study [“Learning at your brain’s rhythm: individualized entrainment boosts learning for perceptual decisions”](https://pubmed.ncbi.nlm.nih.gov/36352510/). In this study, a team at Cambridge claims that entrainment (flashing a bright white light) at a person's individual peak alpha frequency helps them learn to perform a certain difficult perceptual task faster. The task is discriminating between two types of Signal-in-noise patterns between each other: radial ones and concentric ones. 

<img width="264" height="277" alt="image" src="https://github.com/user-attachments/assets/5f3b8fd1-ed8f-4bec-ad45-b834c81eb6d0" />


The “stimulus prototypes” are easy to tell apart. The ones used in the study are Signal-in-noise ones. They are much harder to distinguish — when you have 200 ms to do so. You have the ability to run the code yourself. 

If you are in London and want to volunteer for the actual replication. If you are volunteering, please don't run code that display stimuli: I'll be excluding people who have done too many patterns. 

## Replication in London

I will be running a study replication in London on 10 participants from mid-September until mid-October. 

The location will be either Newspeak House (aka the London College of Political Technology) or another venue in Bethnal Green — based on your preference and each of the venue’s availability. The total time commitment is 4 hours (2 hours on two consecutive days).

[Sign up to be a volunteer](https://forms.gle/X37zyTV3KhbSb3Ze9) — your help will be greatly appreciated.

### London Replication Hardware

1. EEG Headset: [OpenBCI Ultracortex "Mark IV" EEG Headset](https://shop.openbci.com/products/ultracortex-mark-iv) with [8-channel Cyton Board](https://shop.openbci.com/products/cyton-biosensing-board-8-channel) and [ThinkPulse Active Electrodes](https://shop.openbci.com/products/thinkpulse-active-electrode-8-channel-starter-kit). Current electrodes: Fp1, Fp2, Fz, C4, Pz, O1, Oz, O2
2. Monitor: LG UltraGear 24GQ50F 1920×1080 with AMD FreeSync (AMD's VSync) 48..165 Hz.

## Running your own replication

Right now the code is more suited for a demo of a replication and testing the pipeline, rather than easily running a full replication with a few bash commands. There is still a bit of work remaining for this: unhardcoding some constants and surfacing them as command-line arguments as well as making it easier to do double blinding.

Currently you can easily compute IAF and then run blocks of T-match and P-match trials interleaving them — this is different from what the paper did, but much more useful for doing sanity checks on the code and verifying that your setup works.

0. Run `python3 scripts/eo_eeg_screen.py` to display a fixation screen for the EEG recording (it should be recorded with open eyes).  
1. Record EEG from your headset using e.g. [Labrecorder](https://github.com/labstreaminglayer/App-LabRecorder)
2. `python3 -m plot.alpha --picks O1,Oz,O2,Oz <your recording file>`. Use whatever occipital electrodes you have available instead of O1,Oz,O2.  
3. `python3 run_trials.py --stimdir <directory where to save the stimuli patterns> --db study.db --tperblock <trials per block> --blocks <blocks count> --freq <entrainment frequency> --participant participant name`. This will run trials for the P condition and the T condition of the study interleaving them. The results of trials will be recorded in the `study.db`. Currently the code hardcodes parameters necessary for a VRR monitor with a variable refresh rate spanning at least 60..144. In principle you can make the code work with a fixed refresh rate with relatively small amount of modifications — but I haven't tried this, because using VRR allows for a much more precise flicker timing.
4. Plot linear regression of the accuracy and estimate a learning rate: `python3 plot/accuracy_linear_regression.py` to plot estimates of the learning rate (with interleaved blocks). Use `--exclude` if you want to exclude some blocks (for reasons such as burn-in, too much distractions in the environment, bugs in the modified code, etc).

You can use `SDL_VIDEO_WINDOW_POS` environment variable to target a specific monitor in your multi-monitor setup. For instance: `SDL_VIDEO_WINDOW_POS='1920,1'` (the first number is the width, the second is the 0-based monitor number).

### Early signs of the replicated effect

<img width="487" height="293" alt="image" src="https://github.com/user-attachments/assets/2ffa4b40-c302-4365-bbb7-a69a30ad6677" />

After doing the above procedure, I got my learning rate chart (stitched from multiple days). My average accuracy in the T-match condition was 64% vs only 58% in the P-match condition. I also felt like I was subjectively 'learning more' in the T-match condition. Somehow it seems that the learning rate was higher in the P-match condition than the T-match condition — the opposite of what one might predict from looking at the paper. However, we are interleaving blocks here and most of the learning could've still happened in the T-match condition (maybe I first learned to do easier patterns faster and more reliably which carried over to the next P blocks). Or maybe the data is just noisy. 

I don't think it makes sense to speculate much here — it's time to collect the actual data under conditions of the original study. If you are in London, consider [signing up to be a volunteer](https://forms.gle/X37zyTV3KhbSb3Ze9)

## Software components

There is a number of other scripts potentially useful in exploring the data. Some of them are documented below.

### Working with EEG recordings

1. `python -m scripts.replay_xdf <xdf_file>` creates an LSL stream that replays a recording by pushing data onto it every now and then (computed based on LSL chunk size and the frequency of the recording, 0.256s on OpenBCI recorded data and M1 macbook laptop). The recording is loaded via `load_xdf()` in `file_formats.py`, it does it via `pyxdf` and then converts the data to MNE format. TODO: check and document why it converts from uV to V by multiplying it 1e-6 (where does this difference in the formats is coming from?).

### Debugging a hardware connection

1. `python3 -m plot.EEG_rms2` finds and opens an EEG stream, displays uvRms (so you can check if electrodes are touching the skin properly), a spectrogram (aka PSD) and a chart with readings from each electrode. It's currently misnamed (it started with just with printing uvRms and grew into the current setup)

2. `python3 -m plot.EEG_rms2_pyglet_claude` the version of the previous script that was autogenerated with Claude. It plots everything in higher definition on macs because pygame doesn't support retina (TODO: actually check the reason for why it's higher definition).

### Computing IAF

1.  `python3 -m plot.alpha <xdf file>` computes IAF and plots various IAF-related plots from a recording via multiple methods. The first one is similar to what the authors of the original papers were doing: it gets a PSD, finds a peak and draws a red line through it. The second and third one slide a window and compute IAF in each of the ones and then plot a distribution of . The first method is fast, the second and the third one are slow (they can be commented out). The script supports multiple formats via `load_recording()` from `file_formats.py`.

2.  `python3 -m plot.alpha_report --separate-channels  --picks O1,O2,Oz --chunk-shift 5 --chunk-duration 15 recording.xdf report.html`. Computes a report of how IAF changes over time. Slides a window of size `chunk-duration` seconds (15 in the example) shifting it by `chunk-shift` seconds each. Generates an HTML report that's often easier to interpret than e.g. a spectrogram.

3.  `python3 -m plot.EOEC <xdf file>` plots IAF from eye-open-eye-closed data. Subtracts the two (EO, EC) PSDs from each other and plots the resulting delta on the screen along with the found peak. Additionally plots the EO PSD (concated from all segments) as well as EC PSD (concated from all segments) allowing estimating the difference between these methods. TODO: check if concatenation are performed correctly, as they are currently done by concatenating raw data (which might produce some artifacts).

### Generating glass images

1. `python3 scripts/glass.py --angle <angle> --snr 0.24` Use an angle of 0 for radial images and 90 for concentric ones. You can also generate images of in-between angles, however they are not required for the study.

### Running flicker code individually

`python3 scripts/flicker.py --flicker-frequency 11 --target-min-refresh-rate 60 --target-max-refresh-rate 120  `

The script currently have the same drawback: the physical size of the flash doesn't necessarily match the target physical size (7.9o × 7.9o (2.3 × 2.3 arc min2 per dot)). We'll implement calculating the physical flash size once the actual flash is stable.

You can use `SDL_VIDEO_WINDOW_POS` to move the window to the other monitor. For instance: `SDL_VIDEO_WINDOW_POS='1920,1'` (the first number is the width, the second is the 0-based monitor number). TODO: implement automatic picking

How it works. Utilizes VSync for the exact perfect flicker rate. Empirically it seems to work works fairly well. It attempts to find a target monitor refresh rate between 48 Hz and <target> Hz, then it draws X "off" frames and 1 "on" frame where X is computed based on the target refresh rate and the desired flicker frequency. A version of the previous code that uses `time.sleep()` to sleep for 75% of the inter-frame interval. Then busy-waits (spinlock-style) in a `while` loop. This produces a more stable FPS. Additionally prints more debug information on how long `flip()` takes plus target interval and the actual delay.

## Miscellaneous scripts

1. `python scripts/calculate_possible_flicker_rates.py 165 144 120 100`. Calculates possible flicker rates from a list of static fixed refresh rates as well as deltas between them so you can estimate max possible error between a person's IAF and their flicker rate. This is not needed for VRR monitors. 
