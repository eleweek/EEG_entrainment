import time
import objc
from Quartz.CoreGraphics import *
from AppKit import NSApplication, NSApp, NSWindow, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable, NSBackingStoreBuffered, NSColor, NSView, NSTimer, NSRectFill, NSMakeRect, NSObject

class FlashingSquareView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(FlashingSquareView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.flash_on = False
        self.flash_frequency = 1  # Flash frequency in Hz
        self.refresh_rate = 60  # Monitor refresh rate in Hz
        self.frames_per_flash = self.refresh_rate // self.flash_frequency
        self.frame_count = 0
        return self

    def drawRect_(self, rect):
        NSColor.blackColor().set()
        NSRectFill(self.bounds())

        if self.flash_on:
            NSColor.whiteColor().set()
            square_size = 50
            square_rect = NSMakeRect(
                (self.bounds().size.width - square_size) / 2,
                (self.bounds().size.height - square_size) / 2,
                square_size, square_size
            )
            NSRectFill(square_rect)

    def update(self):
        self.flash_on = (self.frame_count % self.frames_per_flash) == 0
        self.frame_count += 1
        self.setNeedsDisplay_(True)

class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        styleMask = (NSWindowStyleMaskTitled | 
                     NSWindowStyleMaskClosable | 
                     NSWindowStyleMaskResizable)
        
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(100, 100, 800, 600),
            styleMask,
            NSBackingStoreBuffered,
            False
        )
        self.window.setTitle_("Flashing Square")
        self.view = FlashingSquareView.alloc().initWithFrame_(NSMakeRect(0, 0, 800, 600))
        self.window.setContentView_(self.view)
        self.window.makeKeyAndOrderFront_(None)
        self.window.display()

        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / self.view.refresh_rate,
            self,
            objc.selector(self.run_update, signature=b'v@:@'),
            None,
            True
        )

    def run_update(self, timer):
        self.view.update()

if __name__ == '__main__':
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    NSApp().setDelegate_(delegate)
    app.run()
