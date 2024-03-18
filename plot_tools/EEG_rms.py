# EEG_rms.py

import pygame, time
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream


print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
stream_idx = 0
inlet = StreamInlet(streams[stream_idx])
EEG_sample, timestamp = inlet.pull_chunk(max_samples=250)

device_info = f"Device {stream_idx}: {streams[stream_idx].name()}"

sr = streams[0].nominal_srate()

# colors
gry = (128,128,128)
wht = (255,255,255)


# Start PyGame display window, pre-load images
pygame.init()
pygame.display.init() 
screen_width, screen_height = 1000, 1000

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('EEG Noise RMS Display')
font = pygame.font.Font(None, 36)

text = font.render('Trial text', True, gry) 
text_rect = text.get_rect(center=(screen_width/4, screen_height/2))
device_info_text = font.render(device_info, True, gry)
device_info_rect = device_info_text.get_rect(center=(screen_width/4, (screen_height/2)-30))

window_length = 1000  #ms
inter_window_delay = 1000  #ms

# trial_loop
t_n = 1000
t=0


#user_confirms = input("LSL Stream started. Confirm recording start if needed. [Y]/N \n")
most_recent_RMS = []

while t<t_n:

    window_step = 0
    while window_step < window_length:
        window_step += 1

         # Get latest second of EEG
        EEG_sample, timestamp = inlet.pull_chunk(max_samples=250)

        # TODO
        # Actually implement RMS
        # Process, filter, ...
        most_recent_RMS = EEG_sample[0][0]

        # Update screen
        pygame.event.get()
        screen.fill(wht)
        screen.blit(device_info_text,device_info_rect)
        trial_text = f"Most recent RMS: {most_recent_RMS}"
        text = font.render(trial_text, True, gry) 
        screen.blit(text, text_rect)
        pygame.display.flip()

        time.sleep(inter_window_delay / 1000)

# Confirm end, clean up
goodbye_text = f"Session complete with {t} trials"
print(goodbye_text)
