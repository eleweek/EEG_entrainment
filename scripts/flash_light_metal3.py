import objc
from Cocoa import NSApplication, NSApp, NSWindow, NSMakeRect, NSObject
from Quartz import CoreGraphics, CGDisplayCopyAllDisplayModes, CGDisplayCopyDisplayMode, CGDisplayModeGetRefreshRate
import Metal
import MetalKit
from AppKit import NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable, NSView
import time

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
        self.refresh_rate = 60  # Default value, will be updated later
        self.frames_per_flash = int(self.refresh_rate // self.flash_frequency)
        self.frame_count = 0

        self.frame_times = []
        self.last_refresh_check = 0
        self.refresh_check_interval = 1  # Check refresh rate every 1 second

        return self

    def get_display_refresh_rate(self):
        main_display = CoreGraphics.CGMainDisplayID()
        current_mode = CGDisplayCopyDisplayMode(main_display)
        refresh_rate = CGDisplayModeGetRefreshRate(current_mode)
        return refresh_rate

    def drawInMTKView_(self, view):
        current_time = time.time()
        self.flash_on = (self.frame_count % self.frames_per_flash) == 0
        self.frame_count += 1

        self.frame_times.append(current_time)

        if len(self.frame_times) > 60:
            fps_60 = 60 / (self.frame_times[-1] - self.frame_times[-61])
        else:
            fps_60 = 0.0

        # Calculate FPS for the last 300 frames
        if len(self.frame_times) > 300:
            fps_300 = 300 / (self.frame_times[-1] - self.frame_times[-301])
            self.frame_times.pop(0)
        else:
            fps_60 = fps_300 = 0.0

        if current_time - self.last_refresh_check >= self.refresh_check_interval:
            t0 = time.perf_counter()
            self.refresh_rate = self.get_display_refresh_rate()
            self.last_refresh_check = current_time
            print(f"Refresh rate updated: {self.refresh_rate} Hz, took {time.perf_counter() - t0:.6f} seconds")

        print(f"Frame: {self.frame_count}, Time: {current_time:.2f}, "
              f"FPS (60): {fps_60:.2f}, FPS (300): {fps_300:.2f}, "
              f"Refresh Rate: {self.refresh_rate:.2f} Hz")

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

        initial_refresh_rate = self.view.get_display_refresh_rate()
        print(f"Initial display refresh rate: {initial_refresh_rate} Hz")


        self.window.makeKeyAndOrderFront_(None)
        self.window.display()

if __name__ == '__main__':
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    NSApp().setDelegate_(delegate)
    app.run()