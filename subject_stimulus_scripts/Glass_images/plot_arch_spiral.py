import numpy as np
import matplotlib.pyplot as plt

def draw_archimedean_spiral(dim=500, a=0, b=10, loops=3):
    # Create a 2D grid of zeros
    grid = np.zeros((dim, dim))
    
    # Center of the grid
    center_x, center_y = dim // 2, dim // 2
    
    # Define the maximum theta to achieve 'loops' loops, 2*pi per loop
    max_theta = 2 * np.pi * loops
    
    # Generate theta values
    theta = np.linspace(0, max_theta, num=1000*loops)
    
    # Calculate r for each theta
    r = a + b * theta
    
    # Convert polar coordinates (r, theta) to Cartesian coordinates (x, y)
    x = center_x + r * np.cos(theta)
    y = center_y + r * np.sin(theta)
    
    # Plotting the spiral
    for i in range(len(x)):
        ix, iy = int(x[i]), int(y[i])
        if 0 <= ix < dim and 0 <= iy < dim:
            grid[ix, iy] = 1  # Mark this point on the grid
    
    return grid

# Generate the spiral grid
spiral_grid = draw_archimedean_spiral()

# Use matplotlib to visualize the spiral
plt.figure(figsize=(10, 10))
plt.imshow(spiral_grid, cmap='binary', origin='lower')
plt.title('Archimedean Spiral with 3 Loops (500x500 grid)')
plt.show()