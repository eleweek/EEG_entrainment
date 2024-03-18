# Eyes-Open Eyes-Closed Experiment Stimulus script
# axs, Mar 2024
#
# Use conda env with Python 3.8, and Psychopy installed

import os
import pygame
import wave, pyaudio

import numpy as np

from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream


# Confirm status
live_run = False
output_device_num = 0  # 0 is default, but might need to pick 4 for multi
audio_channels = 2


# Start PyLSL marker stream
lsl_info = StreamInfo('MarkerSTR','Markers',1,0,'string','EOEC_Marker_String_stream')
lsl_outlet = StreamOutlet(lsl_info)

# colors
blk = (0,0,0)
gry = (128,128,128)
wht = (255,255,255)

# Init, get files
path_here = os.getcwd()
audio_file_path = os.path.join(path_here,'EOEC')

# start PyAudio
p = pyaudio.PyAudio()

# Start PyGame display window, pre-load images
pygame.init()
pygame.display.init() 
screen_width, screen_height = 1000, 1000

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Eyes Open, Eyes Closed')
font = pygame.font.Font(None, 36)

text = font.render('Trial text', True, gry) 
text_rect = text.get_rect(center=(screen_width/4, screen_height/2))


def send_trig(trig_num,trig_long_str):
    lsl_outlet.push_sample([trig_long_str])

def load_sound(filename):
    path_sound = audio_file_path + filename

    s1 = wave.open(path_sound, 'rb')
    s1_n = s1.getnframes()
    s1_sr = s1.getframerate()
    s1_width = int(s1.getsampwidth())

    return s1


# Setup trial struct
n_trials_per_block = 2
n_blocks = 3
audio_filenames = ['/open.wav','/closed.wav']

# make at least n_trials with alternating audio_filenames
trials_structure = audio_filenames * n_blocks

send_trig(0, 'xx_EOEC_begin')

# trial_loop
t_n = n_trials_per_block * n_blocks
t=0
play_time = 30  # seconds

trial_text_list = ["0"] *2
trial_text_list[0] = "Now, please: --  "



user_confirms = input("LSL Stream started. Confirm recording start if needed. [Y]/N \n")


# Trial loop
while t<t_n:

    s1 = load_sound(trials_structure[t])
    mix_stream = p.open(format=p.get_format_from_width(int(s1.getsampwidth())), channels=int(s1.getnchannels()), rate=s1.getframerate(), output=True, output_device_index=output_device_num)

    text_start = f"{t+1} EOEC {trials_structure[t]}"
    send_trig(t+1, text_start)
    print(text_start)

    # Update screen loop
    pt=0
    while pt < play_time:  # while there are still bytes to play, read frames and write to stream
        pt += 1
        

        # Update screen
        pygame.event.get()
        screen.fill(wht)

        trial_text_here = f"Trial {t+1} Sec {pt+1} - Now, please ensure eyes are {trials_structure[t][1:-4]}"
        text = font.render(trial_text_here, True, gry)
        screen.blit(text, text_rect)
        pygame.display.flip()

        if pt==1:  
            splay1 = s1.readframes(s1.getnframes())
            splay1_decode = np.frombuffer(splay1, np.int8)
            mix_stream.write(splay1_decode)
        else:
            pygame.time.delay(1000)

    

    t = t+1

    mix_stream.close()



# Confirm end, clean up
mix_stream.close()
goodbye_text = f"Session EOEC complete with {t} trials"
print(goodbye_text)
p.terminate

