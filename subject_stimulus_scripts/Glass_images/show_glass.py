# Show Glass images
# axs, July 2024
#
# Use conda env with Python 3.8, and Psychopy installed

import os, sys, time
import pygame
import wave, pyaudio

import create_glass_images

import numpy as np


from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
from timeit import default_timer as timer



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
screen_center = [screen_width//2,screen_height//2]

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Glass swirl test')
font = pygame.font.Font(None, 36)

text = font.render('Trial text', True, gry) 
text_rect = text.get_rect(center=(screen_width/4, screen_height/2))



# Setup trial struct
n_trials_per_block = 4
n_blocks = 3
audio_filenames = ['/correct.wav','/wrong.wav']



# trial_loop
t_n = n_trials_per_block * n_blocks
t=0

# Trial setup
time_fix = 0.1
time_show = 1
time_respond = 2  # Not exclusive with show time

# Fixation Cross setup
fix_target_str = '+'   # Needs to be in exact center, so let's check sizes
fix_target_surface = font.render(fix_target_str, True, gry)
fix_size_x, fix_size_y = fix_target_surface.get_size()
fix_center = [(screen_width-fix_size_x) // 2, (screen_height-fix_size_y) // 2]


# Helper subfunctions
def send_trig(trig_num,trig_long_str):
    lsl_outlet.push_sample([trig_long_str])

def wait_key():
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                waiting = False

def text_arr_to_multiline(text_a):
    text_lines = '\n'
    text_lines = text_lines.join(text_a)
    return text_lines









# Intro
text_start_a = ["Welcome!",f"You will see {n_trials_per_block} trials in a block, with {n_blocks} blocks.","Press SPACE to proceed."]
text_start = text_arr_to_multiline(text_start_a)
text = font.render(text_start, True, wht)
screen.blit(text, text_rect)
display_report = pygame.display.flip()
print(text_start)
time.sleep(0.1)
wait_key()
time.sleep(0.1)

# Training loop


send_trig(0, 'xx_Glass_begin')
session_time0 = timer()

block = 0
# BLOCK LOOP
for block in range(n_blocks):
    

    screen.fill(blk)
    block_text = f"Now starting block {block+1}/{n_blocks}."
    rendered_text = font.render(block_text, True, wht)
    screen.blit(rendered_text, text_rect)
    display_report = pygame.display.flip()
    print(block_text)
    time.sleep(1)
    
    t = 0
    # TRIAL LOOP
    for trial in range(n_trials_per_block):
        print(f"Block {block+1} - trial {t+1}")

        # Trial prep
        glass, glass_props = create_glass_images.make_glass(circ_here = True, snr_signal_frac_desired = 0.6)
        glass_rendered = pygame.surfarray.make_surface(glass.transpose((1,0,2)))
        glass_size_x, glass_size_y = glass_rendered.get_size()
        glass_center = [(screen_width-glass_size_x) // 2, (screen_height-glass_size_y) // 2]

        screen.fill(blk)

        # Fixation cross to orient
        screen.blit(fix_target_surface, fix_center)
        pygame.display.flip()
        time.sleep(time_fix)

        # Trial stimulus
        screen.fill(blk)
        screen.blit(glass_rendered, glass_center)
        pygame.display.flip()
        trial_time0= timer()

        # Listen for keypress WHILE checking time
        trial_time_adv = timer()
        trial_time_now = trial_time_adv - trial_time0
        while trial_time_now < 4: # < time_show:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            time.sleep(0.01)
            trial_time_adv = timer()
            trial_time_now = trial_time_adv - trial_time0







        # print(glass_props)



        

        t += 1





# Outro
print("Thanks! Done.")


