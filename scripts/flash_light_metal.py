import Cocoa
import Metal
import MetalKit
import time

class Renderer:
    def __init__(self, target_frequency):
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
        self.target_frequency = target_frequency
        self.flash_duration = 1 / (2 * self.target_frequency)
        self.last_switch_time = time.perf_counter()
        self.flash_state = True
        print(f"Renderer initialized with frequency: {target_frequency} Hz")

    def draw(self, view):
        current_time = time.perf_counter()
        if current_time - self.last_switch_time >= self.flash_duration:
            self.flash_state = not self.flash_state
            self.last_switch_time = current_time
            print(f"Flash state changed to: {self.flash_state}")

        drawable = view.currentDrawable()
        render_pass_descriptor = view.currentRenderPassDescriptor()

        if drawable and render_pass_descriptor:
            command_buffer = self.command_queue.commandBuffer()
            render_encoder = command_buffer.renderCommandEncoderWithDescriptor_(render_pass_descriptor)

            color = b'\xff\xff\xff\xff' if self.flash_state else b'\x00\x00\x00\xff'
            render_encoder.setFragmentBytes_length_atIndex_(color, 16, 0)
            render_encoder.drawPrimitives_vertexStart_vertexCount_(Metal.MTLPrimitiveTypeTriangle, 0, 3)

            render_encoder.endEncoding()

            command_buffer.presentDrawable_(drawable)
            command_buffer.commit()

class FlashingView(MetalKit.MTKView):
    def __init__(self, target_frequency):
        super(FlashingView, self).__init__()
        self.renderer = Renderer(target_frequency)
        self.delegate = self
        print("FlashingView initialized")

    def drawInMTKView_(self, view):
        self.renderer.draw(view)

def create_window_and_view(target_frequency):
    app = Cocoa.NSApplication.sharedApplication()
    
    frame = ((100.0, 100.0), (800.0, 600.0))
    window = Cocoa.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame,
        Cocoa.NSWindowStyleMaskTitled | Cocoa.NSWindowStyleMaskClosable | Cocoa.NSWindowStyleMaskMiniaturizable | Cocoa.NSWindowStyleMaskResizable,
        Cocoa.NSBackingStoreBuffered,
        False)
    
    view = FlashingView.alloc().initWithFrame_device_(window.contentView().bounds(), Metal.MTLCreateSystemDefaultDevice())
    view.setAutoresizingMask_(Cocoa.NSViewWidthSizable | Cocoa.NSViewHeightSizable)
    
    view.setClearColor_((0, 0, 0, 1))
    view.setColorPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
    view.setPreferredFramesPerSecond_(60)  # Set to monitor refresh rate
    
    window.contentView().addSubview_(view)
    window.makeKeyAndOrderFront_(None)
    
    print("Window and view created")
    return app, window, view

if __name__ == "__main__":
    target_frequency = 1  # Set to 1 Hz for easier debugging
    app, window, view = create_window_and_view(target_frequency)
    print("Starting application")
    app.run()