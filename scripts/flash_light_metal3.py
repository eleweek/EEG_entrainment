import objc
from Cocoa import NSApplication, NSApp, NSWindow, NSMakeRect, NSObject
from Quartz import CoreGraphics
import Metal
import MetalKit
from AppKit import NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable, NSView
import time

# Dynamically access the MTKViewDelegate protocol
MTKViewDelegate = objc.protocolNamed('MTKViewDelegate')

class MetalView(NSView, protocols=[MTKViewDelegate]):
    def initWithFrame_(self, frame):
        self = objc.super(MetalView, self).initWithFrame_(frame)
        if self is None:
            return None

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.metal_layer = MetalKit.MTKView.alloc().initWithFrame_device_(frame, self.device)
        self.metal_layer.setDelegate_(self)
        self.metal_layer.setColorPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        self.metal_layer.setFramebufferOnly_(False)
        self.metal_layer.setPaused_(False)
        self.addSubview_(self.metal_layer)
        
        self.command_queue = self.device.newCommandQueue()
        self.flash_on = False
        self.flash_frequency = 10  # Flash frequency in Hz
        self.refresh_rate = 60  # Monitor refresh rate in Hz
        self.frames_per_flash = self.refresh_rate // self.flash_frequency
        self.frame_count = 0

        # Debugging information
        self.frame_times = []

        return self

    def drawInMTKView_(self, view):
        current_time = time.time()
        self.flash_on = (self.frame_count % self.frames_per_flash) == 0
        self.frame_count += 1

        # Log the time a frame was drawn
        self.frame_times.append(current_time)

        # Calculate FPS for the last 60 frames
        if len(self.frame_times) > 60:
            fps_60 = 60 / (self.frame_times[-1] - self.frame_times[-61])
        else:
            fps_60 = 0.0

        # Calculate FPS for the last 300 frames
        if len(self.frame_times) > 300:
            fps_300 = 300 / (self.frame_times[-1] - self.frame_times[-301])
        else:
            fps_300 = 0.0

        # Keep only the last 300 frame times
        if len(self.frame_times) > 300:
            self.frame_times.pop(0)

        print(f"Frame: {self.frame_count}, Time: {current_time}, FPS (60): {fps_60:.2f}, FPS (300): {fps_300:.2f}")

        drawable = self.metal_layer.currentDrawable()
        render_pass_descriptor = self.metal_layer.currentRenderPassDescriptor()
        
        if drawable is not None and render_pass_descriptor is not None:
            color_attachment = render_pass_descriptor.colorAttachments().objectAtIndexedSubscript_(0)
            if self.flash_on:
                color_attachment.setClearColor_(Metal.MTLClearColorMake(1.0, 1.0, 1.0, 1.0))  # White color
            else:
                color_attachment.setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))  # Black color

            command_buffer = self.command_queue.commandBuffer()
            render_encoder = command_buffer.renderCommandEncoderWithDescriptor_(render_pass_descriptor)
            render_encoder.endEncoding()
            command_buffer.presentDrawable_(drawable)
            command_buffer.commit()

    def mtkView_drawableSizeWillChange_(self, view, size):
        pass

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        styleMask = (NSWindowStyleMaskTitled | 
                     NSWindowStyleMaskClosable | 
                     NSWindowStyleMaskResizable)

        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100, 100, 800, 600),
            styleMask,
            CoreGraphics.kCGBackingStoreBuffered,
            False
        )
        self.window.setTitle_("Flashing Square with Metal")
        self.view = MetalView.alloc().initWithFrame_(NSMakeRect(0, 0, 800, 600))
        self.window.setContentView_(self.view)
        self.window.makeKeyAndOrderFront_(None)
        self.window.display()

if __name__ == '__main__':
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    NSApp().setDelegate_(delegate)
    app.run()
