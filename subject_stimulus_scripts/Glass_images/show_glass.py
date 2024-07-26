# Show Glass images
# axs, July 2024
#
# Use conda env with Python 3.8, and Psychopy installed

import os, sys
import pygame
import wave, pyaudio

import numpy as np

from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream



# Confirm status
live_run = False
if live_run:
    user_confirms = input("LSL Stream started. Confirm recording start if needed. [Y]/N \n")

output_device_num = 0


# Start PyLSL marker stream
lsl_info = StreamInfo('MarkerSTR','Markers',1,0,'string','Glass_markers')
lsl_outlet = StreamOutlet(lsl_info)


# colors
blk = (0,0,0)
gry = (128,128,128)
wht = (255,255,255)

# Init, get files
path_here = os.getcwd()
audio_file_path = os.path.join(path_here,'subject_stimulus_scripts','Glass_images')

# start PyAudio
p = pyaudio.PyAudio()

# Start PyGame display window, pre-load images
pygame.init()
pygame.display.init() 
screen_width, screen_height = 1000, 1000

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Glass swirl test')
font = pygame.font.Font(None, 36)

text = font.render('Trial text', True, gry) 
text_rect = text.get_rect(center=(screen_width/4, screen_height/2))

def send_trig(trig_num,trig_long_str):
    lsl_outlet.push_sample([trig_long_str])

# Setup trial struct
n_trials_per_block = 4
n_blocks = 3
audio_filenames = ['/correct.wav','/wrong.wav']

send_trig(0, 'xx_Glass_begin')

# trial_loop
t_n = n_trials_per_block * n_blocks
t=0

# Trial setup
time_fix = 0.1
time_show = 1
time_respond = 2  # Not exclusive with show time

def wait_key():
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                waiting = False

# Intro
text_start = f"Welcome! You will see {n_trials_per_block} trials in a block, with {n_blocks} blocks. Press SPACE to proceed."
text = font.render(text_start, True, wht)
screen.blit(text, text_rect)
display_report = pygame.display.flip()
wait_key()


# Outro
print("Thanks! Done.")


