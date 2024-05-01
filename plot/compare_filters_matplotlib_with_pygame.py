import os
import numpy as np
import pygame
import matplotlib.pyplot as plt
from libs.file_formats import load_openbci_txt
from libs.filters import create_new_raw_with_brainflow_filters_applied
from libs.plot import plot_to_pygame

import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg



script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_dir = os.path.join(os.path.dirname(script_dir), 'sample_data')
sample_file_path = os.path.join(sample_data_dir, 'sample1-openbci-gui.txt')


# colors
gry = (128,128,128)
wht = (255,255,255)

pygame.init()
pygame.display.init() 
SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('EEG Noise RMS Display')
font = pygame.font.Font(None, 36)


raw = load_openbci_txt(sample_file_path)
brainflow_raw = create_new_raw_with_brainflow_filters_applied(raw)

raw.filter(l_freq=1.0, h_freq=45.0)
raw.notch_filter(50, notch_widths=4)
raw.plot(duration=20, show=False, show_scrollbars=False, show_scalebars=False, block=False)

psd = raw.compute_psd()
psd_plot_fig = psd.plot(average=True, show=False)
matplotlib.pyplot.title("MNE with FIR and 4Hz-wide notch filter at 50HZ")


brainflow_raw.plot(show=False)
brainflow_psd = brainflow_raw.compute_psd()
brainflow_psd_fig = brainflow_psd.plot(average=True, show=False)
matplotlib.pyplot.title("Brainflow filters in OpenBCI GUI")


TOP_MARGIN = 20
LEFT_MARGIN = 20

pygame_running = True
while pygame_running:
    for event  in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame_running = False

    screen.fill(wht)
    
    psd_plot_pygame_image = plot_to_pygame(agg, psd_plot_fig)

    brainflow_psd_plot_pygame_image = plot_to_pygame(agg, brainflow_psd_fig)

    screen.blit(psd_plot_pygame_image, (LEFT_MARGIN, TOP_MARGIN))
    screen.blit(brainflow_psd_plot_pygame_image, 
                (LEFT_MARGIN, TOP_MARGIN + psd_plot_pygame_image.get_height() + 20)
                )
    pygame.display.flip()