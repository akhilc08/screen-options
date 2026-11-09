"""Collect and clean actionable elements from the macOS accessibility tree.

Stage 1-2 of the candidate pipeline: walk the frontmost window and the menu bar,
keep elements a user could act on, and describe each with its role, label,
section context and state.
"""
from dataclasses import asdict

from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)

from .elements import Element

ACTIONABLE = {
    "AXButton", "AXMenuItem", "AXTextField", "AXTextArea", "AXCheckBox",
    "AXRadioButton", "AXPopUpButton", "AXComboBox", "AXLink", "AXSlider",
    "AXTab", "AXRow", "AXCell", "AXDisclosureTriangle", "AXSearchField",
    "AXMenuButton", "AXIncrementor", "AXSwitch", "AXToggle",
}
LABEL_ATTRS = ("AXTitle", "AXDescription", "AXValue", "AXHelp", "AXPlaceholderValue")
CONTEXT_ROLES = {"AXGroup", "AXScrollArea", "AXOutline", "AXTable", "AXList", "AXTabGroup",
                 "AXToolbar", "AXSplitGroup", "AXSheet", "AXMenu", "AXMenuBarItem", "AXWebArea"}
MAX_NODES = 20000
MAX_DEPTH = 60
MAX_LABEL = 80


class AXHung(RuntimeError):
    """The app stopped answering AX requests (raised only when HUNG_LIMIT is set)."""


HUNG_LIMIT = None          # opt-in: raise AXHung after this many timeouts in a row (the sandbox collector sets it)
_timeouts = 0


def attr(el, name):
    global _timeouts
    err, val = AXUIElementCopyAttributeValue(el, name, None)
    if err == -25204:      # kAXErrorCannotComplete: the app did not answer within the messaging timeout
        _timeouts += 1
        if HUNG_LIMIT and _timeouts >= HUNG_LIMIT:
            _timeouts = 0
            raise AXHung(name)
    else:
        _timeouts = 0
    return val if err == 0 else None


def _text(v):
    return v.strip()[:MAX_LABEL] if isinstance(v, str) and v.strip() else None


def label_of(el):
    for name in LABEL_ATTRS:
        if t := _text(attr(el, name)):
            return t
    # Rows and cells usually carry their text on a descendant static text
    stack = list(attr(el, "AXChildren") or [])[:8]
    while stack:
        child = stack.pop(0)
        if t := _text(attr(child, "AXValue")) or _text(attr(child, "AXTitle")) or _text(attr(child, "AXDescription")):
            return t
        stack.extend(list(attr(child, "AXChildren") or [])[:4])
    return None


def frame_of(el):
    pos, size = attr(el, "AXPosition"), attr(el, "AXSize")
    if pos is None or size is None:
        return None
    _, p = AXValueGetValue(pos, kAXValueCGPointType, None)
    _, s = AXValueGetValue(size, kAXValueCGSizeType, None)
    return (round(p.x), round(p.y), round(s.width), round(s.height))


MIN_SIDE = 4    # Chromium shrinks scrolled-out web nodes to slivers


def off_bounds(frame, bounds):
    if frame is None or bounds is None:
        return False
    x, y, w, h = frame
    bx, by, bw, bh = bounds
    return w < MIN_SIDE or h < MIN_SIDE or x + w <= bx or y + h <= by or x >= bx + bw or y >= by + bh


def walk(root, source, skip_context=(), bounds=None, keep_unlabeled=False, handles=None):
    """Actionable elements under `root`, each with its ancestor context.

    An actionable element nested in an actionable ancestor with the same label
    (row > cell > button in tab lists) is dropped, keeping the outermost. With
    `bounds` (x, y, w, h), subtrees whose frame lies outside it are skipped, so
    only on-screen elements come back. With `keep_unlabeled`, actionable
    elements without a label are kept (label "") for OCR to name later. A
    `handles` list gets each kept element's live AX handle, in the same order.
    """
    out, seen, nodes = [], set(), 0
    stack = [(root, 0, [], None)]
    while stack and nodes < MAX_NODES:
        el, depth, ctx, parent_label = stack.pop()
        nodes += 1
        frame = frame_of(el) if bounds is not None or source == "window" else None
        if off_bounds(frame, bounds):
            continue
        role = attr(el, "AXRole") or "?"
        label = None
        if role in ACTIONABLE:
            label = label_of(el)
            if (label and label != parent_label) or (keep_unlabeled and not label and frame):
                key = (role, label, tuple(ctx[-2:])) if label else (role, frame)
                if key not in seen:
                    seen.add(key)
                    out.append(Element(
                        id=f"{source[0]}{len(out)}", source=source, role=role, label=label or "",
                        context=ctx[-2:], enabled=attr(el, "AXEnabled") is not False, frame=frame,
                    ))
                    if handles is not None:
                        handles.append(el)
        child_ctx = ctx
        if role in CONTEXT_ROLES or role == "AXMenuItem":   # submenu items get their parent item as context
            t = _text(attr(el, "AXTitle")) or (_text(attr(el, "AXDescription")) if role != "AXMenuItem" else None)
            if t and t not in skip_context:
                child_ctx = ctx + [t[:40]]
        if depth < MAX_DEPTH:
            for child in reversed(attr(el, "AXChildren") or []):
                stack.append((child, depth + 1, child_ctx, label or parent_label))
    return out


def app_menus(menubar):
    """Menu bar items minus the Apple menu, which is the same in every app and lists recent files."""
    items = list(attr(menubar, "AXChildren") or [])
    return [m for m in items if attr(m, "AXTitle") != "Apple"]


def snapshot(pid, app_name):
    """All actionable elements of an app's focused window and menu bar."""
    app = AXUIElementCreateApplication(pid)
    window = attr(app, "AXFocusedWindow") or attr(app, "AXMainWindow")
    menubar = attr(app, "AXMenuBar")
    title = _text(attr(window, "AXTitle")) if window else None
    elements = walk(window, "window", skip_context={title}) if window else []
    for menu in app_menus(menubar) if menubar else []:
        elements += walk(menu, "menubar")
    for i, e in enumerate(elements):
        e.id = f"e{i}"
    return {
        "app": app_name,
        "window_title": title,
        "elements": [asdict(e) for e in elements],
    }
