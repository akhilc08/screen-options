"""Safety policy and primitive actions for unattended exploration of the user's apps.

Exploration only navigates: it presses links, tabs, rows and plain buttons whose labels
pass a deny list, never toggles or sliders, never types, and never opens message rows in
inbox-like sites (that marks them read). Anything denied can still be a labelled target
in the dataset; it is just never pressed.
"""
import re
import subprocess
import time

import Quartz
from ApplicationServices import AXUIElementCopyActionNames, AXUIElementPerformAction

DENY = re.compile(
    r"\b(send|post|publish|reply|respond|comment|like|unlike|follow|unfollow|connect|subscribe|unsubscribe|"
    r"delete|remove|trash|archive|mark|spam|block|report|mute|buy|order|checkout|cart|purchase|pay|payment|"
    r"donate|submit|sign ?out|log ?out|quit|close|uninstall|install|update|share|invite|accept|decline|join|"
    r"leave|upload|attach|move|star|pin|unpin|react|emoji|call|huddle|record|save|apply|confirm|ok|yes|"
    r"done|reset|clear|erase|forget|turn|enable|disable|allow|deny|password|passkey|account|billing|"
    r"restart|shut ?down|sleep|lock|eject|empty|format|rename|duplicate|new|create|add|edit|compose|"
    r"draft|play|pause|shuffle|repeat|download|print|export|import|sync|restore|revert|undo|redo|cut|paste|"
    r"discard|cancel|stop|end|hang ?up|request|approve|reject|vote|rate|review|bid|claim|redeem|transfer|"
    r"commit|push|pull|fetch|stage|unstage|run|debug|deploy|message|start|take|fork|watch|sponsor|merge|"
    r"rebase|terminal|launch|open in|reveal|copy|zoom|full ?screen|minimi[sz]e|maximi[sz]e|resize|hide|"
    r"sidebar|sign ?in|log ?in|mention|format|bold|italic|emoji|attach|gif|reply|customi[sz]e|layout|"
    r"cover|icon|property|template|table|checklist|list|bullet|media|photo|scan|sketch|audio|insert|"
    r"collaborate|style|font|highlight|indent|link|aa)\b",
    re.I,
)
# Link targets that edit, create, configure or pay, whatever the link text says.
DENY_PATH = re.compile(r"/(new|edit|delete|settings|configure|compose|checkout|cart|buy|login|signin|"
                       r"sign_in|logout|signup|join|upload|billing|payment|subscribe|actions/new|premium|upgrade|pricing|"
                       r"plans|admin)(\b|/|$)", re.I)
PRESSABLE = {"AXLink", "AXTab", "AXRadioButton", "AXButton", "AXPopUpButton", "AXMenuButton",
             "AXDisclosureTriangle", "AXRow", "AXCell"}
NEVER = {"AXCheckBox", "AXSwitch", "AXToggle", "AXSlider", "AXIncrementor", "AXTextField", "AXTextArea",
         "AXSearchField", "AXComboBox", "AXMenuItem"}
# Sites and apps where list rows are messages or notifications: opening one changes its state.
MESSAGE_LISTS = ("mail.google.com", "linkedin.com/messaging", "linkedin.com/notifications", "claude.ai")
# Slack channels are allowed: the user OK'd marking them read (2026-10-24).
# Look only: sidebar rows, links and back/forward. Notes and Notion are here because their toolbar
# buttons edit whichever note is open (a 2026-11-05 run inserted a table and a checklist item).
NAV_ONLY_APPS = {"System Settings", "Notes", "Notion"}


def allowed(element, app, url=""):
    """-> (ok, reason). `element` has role, label, context, enabled."""
    if not element.enabled:
        return False, "disabled"
    if element.role in NEVER or element.role not in PRESSABLE:
        return False, f"role {element.role}"
    if not element.label or len(element.label) > 60:
        return False, "label missing or long"
    if DENY.search(element.label):
        return False, "deny list"
    where = f"{app} {url}"
    if element.role in {"AXRow", "AXCell"} and any(m in where for m in MESSAGE_LISTS):
        return False, "message row"
    if app in NAV_ONLY_APPS and element.role not in {"AXRow", "AXCell", "AXLink", "AXRadioButton", "AXTab"} \
            and element.label.lower() not in {"back", "forward"}:
        return False, "look-only app"
    return True, ""


def has_press(ax_element):
    err, actions = AXUIElementCopyActionNames(ax_element, None)
    return err == 0 and bool(actions) and "AXPress" in actions


_last_ours = 0.0
_user_since = time.time() - Quartz.CGEventSourceSecondsSinceLastEventType(
    Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGAnyInputEventType)


def mark_ours():
    """Call after posting synthetic input, so user_idle() does not take it for the user."""
    global _last_ours
    _last_ours = time.time()


def user_idle():
    """Seconds since the user (not us) last touched the keyboard or mouse. The HID idle clock
    counts our own posted events too, so an event within 0.5s after one of ours is ignored."""
    global _user_since
    t_event = time.time() - idle_seconds()
    if t_event > _last_ours + 0.5:
        _user_since = max(_user_since, t_event)
    return time.time() - _user_since


def press(ax_element):
    return AXUIElementPerformAction(ax_element, "AXPress") == 0


def click(x, y):
    """Left mouse click at screen point (x, y)."""
    pt = Quartz.CGPointMake(x, y)
    for kind in (Quartz.kCGEventMouseMoved, Quartz.kCGEventLeftMouseDown, Quartz.kCGEventLeftMouseUp):
        ev = Quartz.CGEventCreateMouseEvent(None, kind, pt, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.05)
    mark_ours()


def activate_element(ax_element, frame):
    """Press through AX when the element offers AXPress; otherwise click its center.

    Chromium web links (Arc) expose only AXShowMenu/AXScrollToVisible yet report success
    for AXPress without doing anything, so the action list decides, not the return code.
    -> "press", "click", or None when neither is possible.
    """
    from .ax import frame_of
    if has_press(ax_element):
        return "press" if press(ax_element) else None
    now = frame_of(ax_element)
    # A click lands wherever the element is now; if the page moved since the capture, the
    # point may sit on a different (unvetted) control, so skip.
    if frame and now and max(abs(a - b) for a, b in zip(frame, now)) <= 4:
        click(now[0] + now[2] / 2, now[1] + now[3] / 2)
        return "click"
    return None


def unminimize(ax_window):
    from ApplicationServices import AXUIElementSetAttributeValue
    return AXUIElementSetAttributeValue(ax_window, "AXMinimized", False) == 0


def key(code, cmd=False):
    """Post one key press (virtual key code) to the frontmost app."""
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(None, code, down)
        if cmd:
            Quartz.CGEventSetFlags(ev, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
        time.sleep(0.02)
    mark_ours()


ESCAPE, LEFT_BRACKET = 53, 33


def go_back():
    key(LEFT_BRACKET, cmd=True)


def idle_seconds():
    """Seconds since the user last touched the keyboard or mouse (our synthetic keys count too)."""
    return Quartz.CGEventSourceSecondsSinceLastEventType(Quartz.kCGEventSourceStateHIDSystemState,
                                                         Quartz.kCGAnyInputEventType)


def mouse_in_corner(margin=5):
    """The user's stop signal: mouse pushed into the top-left corner."""
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return loc.x <= margin and loc.y <= margin


def screen_locked():
    d = Quartz.CGSessionCopyCurrentDictionary() or {}
    return bool(d.get("CGSSessionScreenIsLocked"))


def osascript(script, timeout=10):
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def link_url(ax_element):
    from .ax import attr
    url = attr(ax_element, "AXURL")
    if url is None:
        return None
    return str(url.absoluteString()) if hasattr(url, "absoluteString") else str(url)


def link_target_ok(ax_element, start_domain):
    """Links carry their destination in AXURL: refuse other sites and edit/settings/pay paths."""
    from urllib.parse import urlparse
    url = link_url(ax_element)
    if url is None:
        return True
    host = urlparse(url).hostname or ""
    if host and ".".join(host.split(".")[-2:]) != start_domain:
        return False
    return not DENY_PATH.search(urlparse(url).path)


SAFE_URL = re.compile(r"https://[\w.-]+(/[\w./?=&%#:@~+,;!*()$-]*)?")


class ArcWindow:
    """An Arc window of our own, driven without focus or the mouse so the user can keep working.

    One working tab, navigated by setting its URL (links are followed by their AXURL; Chromium
    ignores AXPress on links anyway). AX and capture reach the window on any Space. Arc shares a
    Space's tabs across its windows, so our tab shows in the user's sidebar but is never active there.
    """

    def __init__(self, pid, window_id=None):
        from .screen import all_windows
        self.pid = pid
        if window_id is None:
            before = {self._eid(w) for w in all_windows(pid)}
            osascript('tell application "Arc" to make new window')
            time.sleep(1.5)
            window_id = osascript('tell application "Arc" to get id of front window')
            new = [w for w in all_windows(pid) if self._eid(w) not in before]
            self.ax = new[0] if new else None
        self.id = window_id
        self.tab = None
        if getattr(self, "ax", None) is None:      # reusing a window we made earlier
            self.tab = self.active_tab() or None
            self.ax = self._find_ax()

    @staticmethod
    def _eid(w):
        from .screen import window_number
        return window_number(w)

    def _find_ax(self):
        """Our AX window, found by loading a marker URL in our tab and looking for the web area
        that shows it. Titles are no good: the user's window can show the same page."""
        import uuid
        from .screen import all_windows
        from .ax import attr
        marker = uuid.uuid4().hex[:12]
        self.open(f"https://www.google.com/?rlca={marker}")
        for _ in range(20):
            time.sleep(0.5)
            for w in all_windows(self.pid):
                stack, n = [w], 0
                while stack and n < 400:
                    el = stack.pop(); n += 1
                    if attr(el, "AXRole") == "AXWebArea":
                        u = attr(el, "AXURL")
                        if u is not None and marker in str(u.absoluteString() if hasattr(u, "absoluteString") else u):
                            return w
                        break
                    stack.extend(attr(el, "AXChildren") or [])
        raise RuntimeError("could not find our Arc window")

    def _q(self, cmd):
        return osascript(f'tell application "Arc" to tell window id "{self.id}" to {cmd}')

    def alive(self):
        return self.id in osascript('tell application "Arc" to get id of windows').split(", ")

    def url(self):
        return self._q("get URL of active tab")

    def active_tab(self):
        return self._q("get id of active tab")

    def loading(self):
        return self._q("get loading of active tab") == "true"

    def open(self, url):
        if not SAFE_URL.fullmatch(url):
            raise ValueError(f"refusing to open {url!r}")
        if self.tab is not None and self.tab not in self.tab_ids():
            print(f"our tab {self.tab} is gone (closed from the sidebar?); opening a new one", flush=True)
            self.tab = None
        if self.tab is None:
            self._q(f'make new tab with properties {{URL:"{url}"}}')
            time.sleep(0.5)
            self.tab = self.active_tab()
        else:
            self._q(f'set URL of tab id "{self.tab}" to "{url}"')

    def back(self):
        self._q(f'tell tab id "{self.tab}" to go back')

    def tab_ids(self):
        return set(self._q("get id of tabs").split(", "))

    def keep_one_tab(self, before):
        """After a press: if our window switched tabs, go back to ours, and close the other tab only
        if it is new (not in `before`, the tab ids from just before the press), since the Space's
        existing tabs are the user's. -> whether a tab was closed."""
        now = self.active_tab()
        if self.tab is None or now == self.tab:
            return False
        closed = bool(now) and now not in before
        if closed:
            self._q(f'close tab id "{now}"')
        self._q(f'tell tab id "{self.tab}" to select')
        return closed

    def close(self):
        if self.tab is not None:
            self._q(f'close tab id "{self.tab}"')
            self.tab = None
        osascript(f'tell application "Arc" to close window id "{self.id}"')


def arc_tabs():
    """{tab id: window id} over every Arc window."""
    out = {}
    for wid in osascript('tell application "Arc" to get id of windows').split(", "):
        if wid:
            for tid in osascript(f'tell application "Arc" to get id of tabs of window id "{wid}"').split(", "):
                if tid:
                    out.setdefault(tid, wid)
    return out


def close_new_arc_tabs(before):
    """Close Arc tabs that appeared since `before` (from arc_tabs()); a native app opened a link."""
    new = {t: w for t, w in arc_tabs().items() if t not in before}
    for tid, wid in new.items():
        osascript(f'tell application "Arc" to close tab id "{tid}" of window id "{wid}"')
    return len(new)


def set_muted(muted):
    """-> previous muted state. Feeds autoplay video overnight; restored when the run ends."""
    before = osascript("output muted of (get volume settings)") == "true"
    osascript(f"set volume output muted {'true' if muted else 'false'}")
    return before
