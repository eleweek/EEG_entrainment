import objc
from Cocoa import NSApplication, NSApp, NSWindow, NSWindowStyleMask, NSMakeRect, NSView
from Quartz import CoreGraphics
import Metal
import MetalKit

class MetalView(NSView, protocols=[Metal.MTKViewDelegate]):
    def initWithFrame_(self, frame):
        self = objc.super(MetalView, self).initWithFrame_(frame)
        if self is None:
            return None

        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.metal_layer = MetalKit.MTKView.alloc().initWithFrame_device_(frame, self.device)
        self.metal_layer.delegate = self
        self.metal_layer.enableSetNeedsDisplay = True
        self.metal_layer.clearColor = Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        self.metal_layer.colorPixelFormat = Metal.MTLPixelFormatBGRA8Unorm
        self.metal_layer.framebufferOnly = False
        self.addSubview_(self.metal_layer)
        
        self.command_queue = self.device.newCommandQueue()
        self.flash_on = False
        self.flash_frequency = 1  # Flash frequency in Hz
        self.refresh_rate = 60  # Monitor refresh rate in Hz
        self.frames_per_flash = self.refresh_rate // self.flash_frequency
        self.frame_count = 0

        return self

    def drawInMTKView_(self, view):
        self.flash_on = (self.frame_count % self.frames_per_flash) == 0
        self.frame_count += 1

        drawable = self.metal_layer.currentDrawable()
        render_pass_descriptor = self.metal_layer.currentRenderPassDescriptor()
        
        if drawable is not None and render_pass_descriptor is not None:
            command_buffer = self.command_queue.commandBuffer()
            render_encoder = command_buffer.renderCommandEncoderWithDescriptor_(render_pass_descriptor)

            if self.flash_on:
                render_encoder.setClearColor_(Metal.MTLClearColorMake(1.0, 1.0, 1.0, 1.0))  # White color
            else:
                render_encoder.setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))  # Black color

            render_encoder.endEncoding()
            command_buffer.presentDrawable_(drawable)
            command_buffer.commit()

    def mtkView_drawableSizeWillChange_(self, view, size):
        pass

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        styleMask = (NSWindowStyleMask.titled | 
                     NSWindowStyleMask.closable | 
                     NSWindowStyleMask.resizable)

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
