# Create Glass images, 
# axs, Apr 2024
#

import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

from math import atan2, degrees, radians, sqrt


def make_glass(circ_here, snr_signal_frac_desired):

    # Define the size of the image
    width = 500
    height = 500
    canvas_size = width * height
    max_dot_density = 0.03  # Try 3% of canvas with pixels - 0.03
    max_dots = max_dot_density * canvas_size

    # Glass params
    # random.seed(42)


    dot_twin_frac = snr_signal_frac_desired / 2

    initial_dot_density = max_dot_density - dot_twin_frac*max_dot_density


    dot_offset_px = 6 # Somewhere around 4-8 pixels for 16 arc-min
    spiral_b = 10
    if circ_here:
        rotation = 4
    else:
        # For radial stim, dot twin should be straight line from center
        rotation = 0
        i_mov = round(sqrt(dot_offset_px * dot_offset_px) / 2) # Assume square screen aperature


    # Create a blank image
    image = np.zeros((width, height, 3), dtype=np.uint8)

    # Set the color of the glass
    glass_color = (255, 255, 255)

    center = (round((width - 1) / 2), round((height - 1) / 2))
    debug = False

    # Draw dots on the image
    dot_count = 0
    dot_orig_count = 0
    dot_good_twin_count = 0
    dot_noise_count = 0
    twin_draw_due = False
    frac_twinned_so_far = 0.9

    for i in range(width):
        for j in range(height):
            if random.random() < initial_dot_density and dot_count < max_dots:
                dot_draw_due = True
            else:
                dot_draw_due = False
                    
            if dot_draw_due:
                dot_orig_count += 1
                dot_count += 1

                image[i, j] = glass_color

                # Should we draw a twin of this dot now?
                
                if dot_good_twin_count > 5:
                    frac_twinned_so_far = (2*dot_good_twin_count) / dot_orig_count

                if frac_twinned_so_far < dot_twin_frac:
                    if random.random() < (1.8 * dot_twin_frac):  # 'Catch up' if off-screen draws put us under count
                        twin_draw_due = True
                else:
                    if random.random() < (1.3 * dot_twin_frac):
                        twin_draw_due = True

                # Draw 'twin' dot IFF
                if twin_draw_due:
                    # Pixel shall add to signal
                    twin_draw_due = False

                    pts_vs_center = (i - center[0], j - center[1])
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

                        fits_here = False
                        if 0 < new_x < width:
                            if 0 < new_y < height:
                                fits_here = True
                        
                        if fits_here:
                            image[new_x, new_y] = glass_color
                            dot_good_twin_count += 1
                            dot_count += 1
                        else:
                            dot_noise_count += 1  # can't be twinned here, so this is 'noise'
                    else:
                        # Not circular spiral, so linear here
                        # Get a straight line from center to this pixel.
                        # Continue this line by x pixels, and draw the twin dot there
                        if i < center[0]:
                            new_x = round(i - i_mov)
                            if j < center[1]:
                                new_y = round(j - i_mov)
                            else:
                                new_y = round(j + i_mov)
                        else:
                            new_x = round(i + i_mov)
                            if j < center[1]:
                                new_y = round(j - i_mov)
                            else:
                                new_y = round(j + i_mov)
                        
                        fits_here = False
                        if 0 < new_x < width:
                            if 0 < new_y < height:
                                fits_here = True
                        
                        if fits_here:
                            # Add a linear pixel twin here
                            image[new_x, new_y] = glass_color
                            dot_good_twin_count += 1
                            dot_count += 1
                        else:
                            dot_noise_count += 1  # can't be twinned here, so this is 'noise'


                    if debug:
                        debug_str = f"Pts {pts_vs_center} with angle {theta}"
                        print(debug_str)
                
                else:
                    # The above orig pixel shall be untwinned, and considered noise
                    dot_noise_count += 1

    # Summary strings
    summary_str = f"Found {dot_orig_count} original dots and {dot_good_twin_count} twins, {rotation} deg away"
    print(summary_str) 

    snr_signal_frac_empirical = (2 * dot_good_twin_count) / (dot_noise_count + (2 * dot_good_twin_count))
    snr_signal_frac_empirical = round(snr_signal_frac_empirical,3)
    summary_str2 = f"S - {2*dot_good_twin_count} N - {dot_noise_count}; SNR Signal reqested/actual {snr_signal_frac_desired} / {snr_signal_frac_empirical}"
    print(summary_str2)

    dot_density_empirical = round(dot_count / canvas_size,2)
    summary_str3 = f"Dot density max - {max_dot_density}, measured at {dot_density_empirical}"
    print(summary_str3)

    # End make Glass, return
    glass_props = {
        'circ_here': circ_here,
        'SNR_signal_frac_desired': snr_signal_frac_desired,
        'SNR_signal_frac_empirical': snr_signal_frac_empirical,
    }
    return image, glass_props


def parse_arguments():
    parser = argparse.ArgumentParser(description='Create Glass pattern images with circular or radial patterns')
    
    # Create mutually exclusive group for circular/radial
    pattern_group = parser.add_mutually_exclusive_group(required=True)
    pattern_group.add_argument('--circular', action='store_true', 
                              help='Create circular/spiral glass pattern')
    pattern_group.add_argument('--radial', action='store_true', 
                              help='Create radial glass pattern')
    
    # SNR argument
    parser.add_argument('--snr', type=float, required=True,
                       help='Signal-to-noise ratio (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Validate SNR range
    if not 0.0 <= args.snr <= 1.0:
        parser.error("SNR must be between 0.0 and 1.0")
    
    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Determine pattern type
    circ_here = args.circular  # True if circular, False if radial
    
    # Create the glass pattern
    image, Glass_props = make_glass(circ_here=circ_here, snr_signal_frac_desired=args.snr)
    
    # Plot the image
    plt.imshow(image)
    plt.title(f"{'Circular' if circ_here else 'Radial'} Glass Pattern (SNR: {args.snr})")
    plt.show()

else:
    # Default behavior when imported as a module
    image, Glass_props = make_glass(circ_here=False, snr_signal_frac_desired=0.3)