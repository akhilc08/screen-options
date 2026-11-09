// A HiDPI virtual display for the EC2 Mac sandbox, whose only real display is 1024x768. Uses the
// CGVirtualDisplay classes (private, as in DeskPad); the display lives as long as this process runs,
// so it runs as a LaunchAgent. Switch its mode with displayplacer.
//   clang -fobjc-arc -framework Foundation -framework CoreGraphics virtual_display.m -o virtual_display
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

@interface CGVirtualDisplayDescriptor : NSObject
@property (retain) dispatch_queue_t queue;
@property (retain) NSString *name;
@property unsigned int maxPixelsHigh;
@property unsigned int maxPixelsWide;
@property CGSize sizeInMillimeters;
@property unsigned int serialNum;
@property unsigned int productID;
@property unsigned int vendorID;
@property (copy) void (^terminationHandler)(id, id);
@end

@interface CGVirtualDisplayMode : NSObject
- (instancetype)initWithWidth:(unsigned int)width height:(unsigned int)height refreshRate:(double)rate;
@end

@interface CGVirtualDisplaySettings : NSObject
@property unsigned int hiDPI;
@property (retain) NSArray *modes;
@end

@interface CGVirtualDisplay : NSObject
@property (readonly) CGDirectDisplayID displayID;
- (instancetype)initWithDescriptor:(CGVirtualDisplayDescriptor *)descriptor;
- (BOOL)applySettings:(CGVirtualDisplaySettings *)settings;
@end

int main(void) {
    @autoreleasepool {
        CGVirtualDisplayDescriptor *d = [CGVirtualDisplayDescriptor new];
        d.queue = dispatch_get_main_queue();
        d.name = @"rlca virtual display";
        d.maxPixelsWide = 3840;
        d.maxPixelsHigh = 2160;
        d.sizeInMillimeters = CGSizeMake(330, 206);    // about a 15" laptop panel
        d.serialNum = 0x524C4341;
        d.productID = 0x1234;
        d.vendorID = 0x3456;
        CGVirtualDisplay *display = [[CGVirtualDisplay alloc] initWithDescriptor:d];
        CGVirtualDisplaySettings *s = [CGVirtualDisplaySettings new];
        s.hiDPI = 1;
        s.modes = @[
            [[CGVirtualDisplayMode alloc] initWithWidth:2880 height:1800 refreshRate:60],
            [[CGVirtualDisplayMode alloc] initWithWidth:2560 height:1600 refreshRate:60],
            [[CGVirtualDisplayMode alloc] initWithWidth:3360 height:2100 refreshRate:60],
            [[CGVirtualDisplayMode alloc] initWithWidth:3840 height:2160 refreshRate:60],
        ];
        if (![display applySettings:s]) {
            fprintf(stderr, "applySettings failed\n");
            return 1;
        }
        printf("virtual display id %u\n", display.displayID);
        fflush(stdout);
        dispatch_main();
    }
}
