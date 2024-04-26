# Create Glass images, 
# axs, Apr 2024
#

import random
import numpy as np
import matplotlib.pyplot as plt

import time
from math import atan2, degrees, radians, sqrt

# Define the size of the image
width = 500
height = 500

# Glass params
# random.seed(42)
circ_here = True
initial_dot_density = 0.015
dot_twin_frac = 1
dot_offset_px = 6 # Somewhere around 4-8 pixels for 16 arc-min
spiral_b = 10
if circ_here:
    rotation = 4
else:
    # For radial stim, dot twin should be straight line from center
    rotation = 4

max_dots = 250000


# Create a blank image
image = np.zeros((width, height, 3), dtype=np.uint8)

# Set the color of the glass
glass_color = (127, 127, 127)
highlight_color = (250,1,1)

center = (round((width-1)/2), round((height-1)/2))
debug = False

# Draw dots on the image
dot_count = 0
dot_grey_count = 0
dot_red_count = 0
for i in range(width):
    for j in range(height):
        if random.random() < initial_dot_density and dot_count < max_dots:
            dot_count += 1
            image[i, j] = glass_color

            # Draw 'twin' dot IFF
            if random.random() < dot_twin_frac:

                pts_vs_center = (i-center[0], j-center[1])
                theta = (degrees(atan2(pts_vs_center[1],pts_vs_center[0])) + 360.0) % 360.0
                # Center is pt (0,0), and max(i),(max(j/2) has 0 deg theta
                dist_r = sqrt(abs(pts_vs_center[0])**2 + abs(pts_vs_center[1])**2)

                if circ_here:
                    # Imagine a circe / spiral, with radius dist_r, centered on pt 0,0
                    # Draw a single twin dot on this spiral, clockwise by rotation degrees
                    
                    # a is the rotation of an Archemedean spiral
                    # b is the spiral spacing (constant)
                    current_a = dist_r - spiral_b * radians(theta)  
                    new_theta = theta + rotation
                    new_r = current_a + spiral_b * (radians(new_theta))

                    
                    new_x = round(center[0] + new_r * np.cos(radians(new_theta)))
                    new_y = round(center[1] + new_r * np.sin(radians(new_theta)))

                    if 0 < new_x < width:
                        if 0 < new_y < height:
                            image[new_x, new_y] = glass_color
                            dot_red_count +=1
                    


                if debug:
                    debug_str = f"Pts {pts_vs_center} with angle {theta}"
                    print(debug_str)


                
               

# Display the image
plt.imshow(image)
plt.show()

summary_str = f"Found {dot_count} original dots and {dot_red_count} twins, {rotation} deg away"
print(summary_str) 
