import objc
import time
from Cocoa import NSApplication, NSWindow, NSObject, NSMakeRect, NSView
from Cocoa import NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable, NSWindowStyleMaskMiniaturizable, NSBackingStoreBuffered, NSViewWidthSizable, NSViewHeightSizable
from MetalKit import MTKView
import Metal

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 800, 600),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable,
            NSBackingStoreBuffered,
            False
        )
        self.window.setTitle_("Metal Flashing Square")
        self.window.makeKeyAndOrderFront_(None)

        self.mtkView = MTKView.alloc().initWithFrame_(self.window.contentView().frame())
        self.mtkView.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        self.mtkView.setPreferredFramesPerSecond_(60)

        self.mtkView.setDevice_(Metal.MTLCreateSystemDefaultDevice())
        self.mtkView.setColorPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        self.mtkView.setClearColor_(Metal.MTLClearColorMake(0, 0, 0, 1))

        self.delegate = ViewDelegate.alloc().init()
        self.mtkView.setDelegate_(self.delegate)

        self.window.contentView().addSubview_(self.mtkView)

MTKViewDelegate = objc.protocolNamed('MTKViewDelegate')

class ViewDelegate(NSObject, protocols=[MTKViewDelegate]):
    def init(self):
        self = objc.super(ViewDelegate, self).init()
        if self is None:
            return None
        self.device = Metal.MTLCreateSystemDefaultDevice()
        self.commandQueue = self.device.newCommandQueue()
        self.flickerRate = 7  # Change this value to set the flicker rate
        self.frameCounter = 0
        self.startTime = time.time()
        self.totalFrames = 0
        self.fps = 0.0
        return self

    def mtkView_drawableSizeWillChange_(self, view, size):
        # Handle the view size change if necessary
        pass

    def drawInMTKView_(self, view):
        self.frameCounter += 1
        self.totalFrames += 1

        drawable = view.currentDrawable()
        if drawable is None:
            return

        renderPassDescriptor = view.currentRenderPassDescriptor()
        if renderPassDescriptor is None:
            return

        # Set the clear color based on the frame counter
        if self.frameCounter % self.flickerRate < self.flickerRate // 2:
            renderPassDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setClearColor_(Metal.MTLClearColorMake(1, 1, 1, 1))  # White square
        else:
            renderPassDescriptor.colorAttachments().objectAtIndexedSubscript_(0).setClearColor_(Metal.MTLClearColorMake(0, 0, 0, 1))  # Black background

        commandBuffer = self.commandQueue.commandBuffer()
        renderEncoder = commandBuffer.renderCommandEncoderWithDescriptor_(renderPassDescriptor)
        renderEncoder.endEncoding()
        commandBuffer.presentDrawable_(drawable)
        commandBuffer.commit()

        # Calculate FPS every 60 frames
        if self.totalFrames % 60 == 0:
            endTime = time.time()
            elapsedTime = endTime - self.startTime
            self.fps = 60 / elapsedTime
            print(f"Estimated FPS: {self.fps:.2f}")
            self.startTime = endTime

if __name__ == '__main__':
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()
