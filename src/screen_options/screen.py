"""Find an app's working window and capture it as an image.

The AX window choice mirrors what a user means by "the window": the focused one,
else the main one, else the first that is not minimized. The capture is the
matching on-screen CG window, so pixel boxes line up with AX frames.
"""
import ctypes
import struct

import objc
import Quartz
from ApplicationServices import AXUIElementCreateApplication
from Foundation import NSData
from PIL import Image

from .ax import attr, frame_of


def ax_window(pid):
    """-> (AX window or None, whether it is minimized)."""
    app = AXUIElementCreateApplication(pid)
    for name in ("AXFocusedWindow", "AXMainWindow"):
        if (w := attr(app, name)) is not None and attr(w, "AXSubrole") != "AXDesktop":
            return w, bool(attr(w, "AXMinimized"))
    wins = [w for w in attr(app, "AXWindows") or [] if attr(w, "AXSubrole") != "AXDesktop"]
    for w in wins:
        if not attr(w, "AXMinimized"):
            return w, False
    return (wins[0], True) if wins else (None, False)


_HI = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
_HI._AXUIElementCreateWithRemoteToken.restype = ctypes.c_void_p
_HI._AXUIElementCreateWithRemoteToken.argtypes = [ctypes.c_void_p]
_HI._AXUIElementGetWindow.restype = ctypes.c_int32
_HI._AXUIElementGetWindow.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]


def all_windows(pid, max_id=3000):
    """Every AX window of `pid`, including ones on other Spaces (AXWindows lists only the current
    Space). Private API, as AltTab does: AX elements are addressable by a remote token of
    (pid, 0, 'coco', element id), and windows carry small element ids."""
    out = []
    for eid in range(max_id):
        tok = struct.pack("<iiiq", pid, 0, 0x636F636F, eid)
        p = _HI._AXUIElementCreateWithRemoteToken(objc.pyobjc_id(NSData.dataWithBytes_length_(tok, len(tok))))
        if p:
            el = objc.objc_object(c_void_p=p)
            if attr(el, "AXRole") == "AXWindow" and attr(el, "AXSubrole") == "AXStandardWindow":
                out.append(el)
    return out


def window_number(ax_window):
    """CG window number behind an AX window (private _AXUIElementGetWindow), or None."""
    n = ctypes.c_uint32(0)
    return n.value if _HI._AXUIElementGetWindow(objc.pyobjc_id(ax_window), ctypes.byref(n)) == 0 else None


def cg_window_id(pid, frame):
    """CG window number of `pid` whose bounds best match the AX frame (x, y, w, h)."""
    infos = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    best, best_err = None, float("inf")
    for w in infos:
        if w.get("kCGWindowOwnerPID") != pid or w.get("kCGWindowLayer") != 0:
            continue
        b = w.get("kCGWindowBounds")
        err = abs(b["X"] - frame[0]) + abs(b["Y"] - frame[1]) + abs(b["Width"] - frame[2]) + abs(b["Height"] - frame[3])
        if err < best_err:
            best, best_err = w["kCGWindowNumber"], err
    return best


def to_pil(cgimage):
    w, h = Quartz.CGImageGetWidth(cgimage), Quartz.CGImageGetHeight(cgimage)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cgimage))
    return Image.frombuffer("RGBA", (w, h), data, "raw", "BGRA", Quartz.CGImageGetBytesPerRow(cgimage), 1).convert("RGB")


def capture_window(pid, window=None):
    """-> dict(window, minimized, frame, image, cgimage, scale) or None when the app has no window.

    `frame` is the window in screen points; `scale` is image pixels per point. Pass `window`
    (an AX window) to capture that one instead of the focused one; it need not be in front.
    """
    if window is not None:
        win, minimized = window, bool(attr(window, "AXMinimized"))
    else:
        win, minimized = ax_window(pid)
    if win is None:
        return None
    frame = frame_of(win)
    out = {"window": win, "minimized": minimized, "frame": frame, "image": None, "cgimage": None, "scale": None}
    if minimized or frame is None:
        return out
    wid = window_number(win) or cg_window_id(pid, frame)
    if wid is None:
        return out
    cg = Quartz.CGWindowListCreateImage(Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow, wid,
                                        Quartz.kCGWindowImageBoundsIgnoreFraming)
    if cg is None:
        return out
    out.update(cgimage=cg, image=to_pil(cg), scale=Quartz.CGImageGetWidth(cg) / frame[2])
    return out
