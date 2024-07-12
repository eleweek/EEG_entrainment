import Cocoa
import Metal
import MetalKit
import time

class Renderer:
    def __init__(self, device, target_frequency):
        self.device = device
        self.command_queue = self.device.newCommandQueue()
        self.target_frequency = target_frequency
        self.flash_duration = 1 / (2 * self.target_frequency)
        self.last_switch_time = time.time()
        self.flash_state = True

        # Create a simple shader
        self.library = device.newDefaultLibrary()
        self.vertex_function = self.library.newFunctionWithName_('vertex_shader')
        self.fragment_function = self.library.newFunctionWithName_('fragment_shader')

        # Create the render pipeline
        descriptor = Metal.MTLRenderPipelineDescriptor.alloc().init()
        descriptor.vertexFunction = self.vertex_function
        descriptor.fragmentFunction = self.fragment_function
        descriptor.colorAttachments().objectAtIndexedSubscript_(0).pixelFormat = Metal.MTLPixelFormatBGRA8Unorm
        self.pipeline_state = device.newRenderPipelineStateWithDescriptor_error_(descriptor, None)[0]

    def draw(self, view):
        current_time = time.time()
        if current_time - self.last_switch_time >= self.flash_duration:
            self.flash_state = not self.flash_state
            self.last_switch_time = current_time

        drawable = view.currentDrawable()
        render_pass_descriptor = view.currentRenderPassDescriptor()

        if drawable and render_pass_descriptor:
            command_buffer = self.command_queue.commandBuffer()
            render_encoder = command_buffer.renderCommandEncoderWithDescriptor_(render_pass_descriptor)

            render_encoder.setRenderPipelineState_(self.pipeline_state)
            render_encoder.setFragmentBytes_length_atIndex_(b'\x01' if self.flash_state else b'\x00', 1, 0)
            render_encoder.drawPrimitives_vertexStart_vertexCount_(Metal.MTLPrimitiveTypeTriangle, 0, 3)

            render_encoder.endEncoding()
            command_buffer.presentDrawable_(drawable)
            command_buffer.commit()

class FlashingView(MetalKit.MTKView):
    def __init__(self, device, target_frequency):
        super(FlashingView, self).__init__(frame=Cocoa.NSMakeRect(0, 0, 800, 600), device=device)
        self.renderer = Renderer(device, target_frequency)
        self.clearColor = Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        self.colorPixelFormat = Metal.MTLPixelFormatBGRA8Unorm
        self.delegate = self

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
    
    device = Metal.MTLCreateSystemDefaultDevice()
    view = FlashingView.alloc().initWithFrame_device_(window.contentView().bounds(), device)
    view.setAutoresizingMask_(Cocoa.NSViewWidthSizable | Cocoa.NSViewHeightSizable)
    
    window.contentView().addSubview_(view)
    window.makeKeyAndOrderFront_(None)
    
    return app, window, view

if __name__ == "__main__":
    target_frequency = 2  # 2 Hz for easy visual confirmation
    app, window, view = create_window_and_view(target_frequency)
    app.run()