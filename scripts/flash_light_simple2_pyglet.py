import sys
from math import floor
import pyglet
from pyglet import shapes

frequency = float(sys.argv[1])

WIDTH = 1920
HEIGHT = 1080

def find_target_fps(frequency, min_frequency=48, max_frequency=165):
    result = floor(max_frequency / frequency) * frequency

    if result < min_frequency:
        raise ValueError("Frequency too low")
    
    if result > max_frequency:
        raise ValueError("Frequency too high")
    
    return result

target_fps = find_target_fps(frequency)
interval = 1.0 / target_fps
off_frames_per_each_on = int((target_fps - frequency) / frequency)

# Create the window
window = pyglet.window.Window(WIDTH, HEIGHT, fullscreen=True)

fps_display = pyglet.window.FPSDisplay(window=window)

# Define the white rectangle
rect_width, rect_height = 300, 300
rect_x = (WIDTH - rect_width) // 2
rect_y = (HEIGHT - rect_height) // 2
rectangle = shapes.Rectangle(rect_x, rect_y, rect_width, rect_height, color=(255, 255, 255))

frame_count = 0
rectangle_on = False

def update(dt):
    global frame_count, rectangle_on
    frame_count += 1
    rectangle_on = frame_count % (off_frames_per_each_on + 1) == 0

@window.event
def on_draw():
    window.clear()
    fps_display.draw()
    if rectangle_on:
        rectangle.draw()

@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE:
        window.close()

# Set the update interval based on the target FPS
pyglet.clock.schedule_interval(update, interval)

# Start the Pyglet event loop
pyglet.app.run()