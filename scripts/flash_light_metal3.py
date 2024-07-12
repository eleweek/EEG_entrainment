import objc
from Cocoa import NSApplication, NSApp, NSWindow, NSMakeRect, NSObject, NSScreen
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

    def get_all_display_refresh_rates(self):
        refresh_rates = []
        for screen in NSScreen.screens():
            display_id = screen.deviceDescription()['NSScreenNumber']
            current_mode = CGDisplayCopyDisplayMode(display_id)
            refresh_rate = CGDisplayModeGetRefreshRate(current_mode)
            refresh_rates.append((display_id, refresh_rate))
        return refresh_rates

    def drawInMTKView_(self, view):
        current_time = time.time()
        self.flash_on = (self.frame_count % self.frames_per_flash) == 0
        self.frame_count += 1

        self.frame_times.append(current_time)

        if len(self.frame_times) > 60:
            fps_60 = 60 / (self.frame_times[-1] - self.frame_times[-61])
        else:
            fps_60 = 0.0

        if len(self.frame_times) > 300:
            fps_300 = 300 / (self.frame_times[-1] - self.frame_times[-301])
            self.frame_times.pop(0)
        else:
            fps_300 = 0.0

        refresh_rates = self.get_all_display_refresh_rates()

        print(f"FPS: {fps_60:.2f} {fps_300:.2f} flash = {int(self.flash_on)} "
              f"Frame: {self.frame_count}, Time: {current_time:.2f}")
        print(f"Current display refresh rate: {', '.join(str(r[1]) for r in refresh_rates)}")
        print()

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

        initial_refresh_rates = self.view.get_all_display_refresh_rates()
        print(f"Initial display refresh rates: {initial_refresh_rates}")

        self.window.makeKeyAndOrderFront_(None)
        self.window.display()

if __name__ == '__main__':
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    NSApp().setDelegate_(delegate)
    app.run()