import objc
from Cocoa import NSApplication, NSApp, NSWindow, NSMakeRect, NSObject, NSScreen, NSColor
from Quartz import CoreGraphics
import Metal
import MetalKit
from AppKit import NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable, NSView, NSBackingStoreBuffered
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
        self.flash_frequency = 10.0  # Hz
        self.start_time = time.time()
        self.frame_count = 0

        return self

    def drawInMTKView_(self, view):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Square wave
        phase = (elapsed_time * self.flash_frequency) % 1.0
        flash_on = phase < 0.5
        
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            print(f"Frame: {self.frame_count} | Flash: {flash_on}")
        
        drawable = self.metal_layer.currentDrawable()
        render_pass_descriptor = self.metal_layer.currentRenderPassDescriptor()
        
        if drawable and render_pass_descriptor:
            color_attachment = render_pass_descriptor.colorAttachments().objectAtIndexedSubscript_(0)
            
            if flash_on:
                color_attachment.setClearColor_(Metal.MTLClearColorMake(1.0, 1.0, 1.0, 1.0))
            else:
                color_attachment.setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))
            
            command_buffer = self.command_queue.commandBuffer()
            render_encoder = command_buffer.renderCommandEncoderWithDescriptor_(render_pass_descriptor)
            render_encoder.endEncoding()
            command_buffer.presentDrawable_(drawable)
            command_buffer.commit()

    def mtkView_drawableSizeWillChange_(self, view, size):
        pass

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        screens = NSScreen.screens()
        
        # Debug print
        print(f"Found {len(screens)} screens")
        for i, screen in enumerate(screens):
            print(f"Screen {i}: {screen.frame()}")
        
        # Choose target screen
        if len(screens) > 1:
            target_screen = screens[1]
            print("Using second screen")
        else:
            target_screen = screens[0]
            print("Using main screen")
        
        # Position window on target screen
        screen_frame = target_screen.frame()
        # Start with a smaller window that we'll fullscreen
        initial_rect = NSMakeRect(
            screen_frame.origin.x + 100,
            screen_frame.origin.y + 100,
            800,
            600
        )
        
        # Create window with proper style masks for fullscreen
        style_mask = (NSWindowStyleMaskTitled | 
                     NSWindowStyleMaskClosable | 
                     NSWindowStyleMaskResizable)  # Need resizable for fullscreen!
        
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            initial_rect,
            style_mask,
            NSBackingStoreBuffered,
            False
        )
        
        self.window.setTitle_("Metal Flicker")
        
        # Create Metal view
        self.view = MetalView.alloc().initWithFrame_(NSMakeRect(0, 0, 800, 600))
        self.window.setContentView_(self.view)
        
        # Show window first
        self.window.makeKeyAndOrderFront_(None)
        
        # IMPORTANT: Move window to target screen before fullscreen
        self.window.setFrameOrigin_(screen_frame.origin)
        
        # Now toggle fullscreen
        self.window.toggleFullScreen_(None)
        
        print("Window should now be fullscreen on target monitor")

    def applicationShouldTerminateAfterLastWindowClosed_(self, sender):
        return True

if __name__ == '__main__':
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    NSApp().setDelegate_(delegate)
    app.run()