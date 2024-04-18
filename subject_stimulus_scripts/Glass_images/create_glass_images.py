# Create Glass images, 
# axs, Apr 2024
#

import random
import numpy as np
import matplotlib.pyplot as plt

# Define the size of the image
width = 500
height = 500

# Glass params
circ_here = 1
initial_dot_density = 0.03
dot_offset_px = 6 # Somewhere around 4-8 pixels for 16 arc-min
if circ_here:
    spiral_angle = 90  # 
else:
    # For radial stim, dot twin should be straight line from center
    spiral_angle = 0



# Create a blank image
image = np.zeros((width, height, 3), dtype=np.uint8)

# Set the color of the glass
glass_color = (127, 127, 127)

# Draw dots on the image
for i in range(width):
    for j in range(height):
        if random.random() < initial_dot_density:
            image[i, j] = glass_color

            

# Display the image
plt.imshow(image)
plt.show()