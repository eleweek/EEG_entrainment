# Create Glass images, 
# axs, Apr 2024
#

import random
import numpy as np
import matplotlib.pyplot as plt

# Define the size of the image
width = 500
height = 500

# Create a blank image
image = np.zeros((width, height, 3), dtype=np.uint8)

# Set the color of the glass
glass_color = (127, 127, 127)

# Draw dots on the image
for i in range(width):
    for j in range(height):
        if random.random() < 0.5:
            image[i, j] = glass_color

# Display the image
plt.imshow(image)
plt.show()