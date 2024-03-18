import os
import numpy as np
import pygame
import matplotlib.pyplot as plt
from libs.file_formats import load_openbci_txt
from libs.filters import create_new_raw_with_brainflow_filters_applied

import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg

def plot_to_pygame(fig):
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    return pygame.image.fromstring(raw_data, size, "RGB")


script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_dir = os.path.join(os.path.dirname(script_dir), 'sample_data')
sample_file_path = os.path.join(sample_data_dir, 'sample1-openbci-gui.txt')


# colors
gry = (128,128,128)
wht = (255,255,255)

pygame.init()
pygame.display.init() 
screen_width, screen_height = 1000, 1000

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('EEG Noise RMS Display')
font = pygame.font.Font(None, 36)

text = font.render('Trial text', True, gry) 
text_rect = text.get_rect(center=(screen_width/4, screen_height/2))
device_info_text = font.render("some text", True, gry)
device_info_rect = device_info_text.get_rect(center=(screen_width/4, (screen_height/2)-30))



raw = load_openbci_txt(sample_file_path)
brainflow_raw = create_new_raw_with_brainflow_filters_applied(raw)

raw.filter(l_freq=1.0, h_freq=45.0, method="iir", iir_params=None)
raw.notch_filter(50, notch_widths=4)
raw_plot_fig = raw.plot(duration=20, show=False, show_scrollbars=False, show_scalebars=False, block=False)
print(type(raw_plot_fig), raw_plot_fig)

psd = raw.compute_psd()
psd_plot_fig = psd.plot(average=True, show=False)

brainflow_raw.plot(show=False)
brainflow_psd = brainflow_raw.compute_psd()
brainflow_psd_fig = brainflow_psd.plot(average=True, show=False)


pygame_running = True
while pygame_running:
    for event  in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame_running = False

    screen.fill(wht)
    screen.blit(device_info_text,device_info_rect)
    trial_text = f"Most recent RMS: some_test"
    text = font.render(trial_text, True, gry) 
    screen.blit(text, text_rect)
    psd_plot_pygame_image = plot_to_pygame(psd_plot_fig)

    brainflow_psd_plot_pygame_image = plot_to_pygame(brainflow_psd_fig)

    screen.blit(psd_plot_pygame_image, (0, 0))
    screen.blit(brainflow_psd_plot_pygame_image, (0, psd_plot_pygame_image.get_height() + 20))
    pygame.display.flip()