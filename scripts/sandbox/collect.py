"""Collect screens on a throwaway sandbox Mac (EC2 Mac seeded by scripts/vm/seed_content.sh). Nothing on
it is real, so unlike explore.py this launches apps, takes focus, opens menus and dialogs, types fake
text, and presses irreversible controls (send, delete) so the dataset has real examples of them.

Built for the dataset rules: breadth over depth (each app gets a round until it stops yielding new
screens, then the next app), light/dark and several window sizes, menus and dialogs on purpose, and
logged-out websites in Safari. The only things it refuses are what would break the machine or the
collection (sharing, network, users, privacy panes, power, sign-in) and web forms that would reach
real people (contact, sign-up, newsletter).

Writes explore.py's run layout (states.jsonl, transitions.jsonl, shots/) so replay_sources.py and
label_prep.py work on it. Each state's url is sandbox://<app>?mode=<light|dark>&size=WxH (or the web
page's url plus the same query), so look and size count toward a distinct screen.

  RLCA_SANDBOX=1 python scripts/sandbox/collect.py --hours 20
"""
import argparse
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import Quartz
from AppKit import NSApplicationActivateIgnoringOtherApps, NSWorkspace
from Foundation import NSDate, NSRunLoop
from ApplicationServices import AXUIElementCreateApplication, AXUIElementCreateSystemWide, AXUIElementPerformAction, \
    AXUIElementSetAttributeValue, AXUIElementSetMessagingTimeout, AXValueCreate, kAXValueCGPointType, kAXValueCGSizeType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/data"))
import explore as E  # noqa: E402
from screen_options import explore as X  # noqa: E402
from screen_options import ax as rlca_ax  # noqa: E402
from screen_options.ax import attr, frame_of  # noqa: E402
from screen_options.screen import all_windows  # noqa: E402

APPS = ["Finder", "Notes", "Calendar", "Reminders", "Contacts", "System Settings", "Preview", "TextEdit", "Maps",
        "Weather", "Calculator", "Dictionary", "Font Book", "Activity Monitor", "Disk Utility", "Console",
        "Script Editor", "Automator", "Photos", "Music", "TV", "Podcasts", "Books", "News", "Stocks", "Clock",
        "Freeform", "Journal", "Home", "Chess", "Image Capture", "Shortcuts", "Voice Memos", "Stickies", "Tips",
        "Grapher", "Digital Color Meter", "Mail", "Messages", "FaceTime", "Safari",
        # installed by setup_mac.sh when available
        "Visual Studio Code", "LibreOffice", "Firefox", "VLC", "GIMP", "Inkscape", "Obsidian", "Blender"]
SITES = ["https://en.wikipedia.org/wiki/Special:Random", "https://www.openstreetmap.org/", "https://news.ycombinator.com/",
         "https://www.bbc.com/news", "https://www.npr.org/", "https://github.com/explore", "https://stackoverflow.com/questions",
         "https://www.python.org/", "https://developer.mozilla.org/en-US/", "https://www.weather.gov/",
         "https://www.allrecipes.com/", "https://www.imdb.com/chart/top/", "https://www.goodreads.com/",
         "https://www.reddit.com/r/AskHistorians/", "https://www.ikea.com/us/en/", "https://www.target.com/",
         "https://www.bestbuy.com/", "https://www.etsy.com/", "https://www.booking.com/", "https://www.kayak.com/",
         "https://www.amtrak.com/", "https://www.nasa.gov/", "https://www.nytimes.com/", "https://www.espn.com/",
         "https://www.khanacademy.org/", "https://arxiv.org/", "https://www.gutenberg.org/", "https://www.archive.org/",
         "https://www.zillow.com/", "https://www.craigslist.org/", "https://www.usps.com/", "https://www.irs.gov/",
         "https://www.cdc.gov/", "https://www.wikihow.com/", "https://www.youtube.com/", "https://www.apple.com/",
         "https://www.microsoft.com/", "https://www.wolframalpha.com/", "https://www.duckduckgo.com/",
         "https://www.speedtest.net/"]
# Protects the machine and the run: anything that cuts SSH/VNC, signs in, changes users or privacy
# grants, powers off, or installs. Everything else, including delete and send, is fair game here.
MACHINE_DENY = re.compile(
    r"\b(shut ?down|restart|log ?out|sign ?out|sleep|lock screen|erase|reset|startup disk|recovery|"
    r"sharing|remote|screen sharing|file sharing|firewall|network|wi-?fi|ethernet|vpn|bluetooth|users?|groups?|"
    r"password|passwords|passkey|filevault|privacy|security|accessibility|login items|software update|update|"
    r"install|uninstall|migration|transfer|time machine|sign ?in|log ?in|apple ?id|apple account|icloud|"
    r"energy|battery|screen time|profiles?|keychain|developer|terminal|quit|"
    # Disk Utility and Activity Monitor: never touch disks or kill processes (sshd, the virtual display)
    r"partition|mount|unmount|eject|first aid|restore|format|raid|force|kill|stop|sample|signal)\b", re.I)
# Web forms that would reach real people or create accounts.
WEB_DENY = re.compile(r"\b(contact|subscribe|newsletter|sign ?up|register|join|create account|send|submit|"
                      r"apply|checkout|buy|add to cart|order|book now|reserve|donate|report|feedback|chat)\b", re.I)
SAFE_PRESS = {"AXButton", "AXLink", "AXTab", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXDisclosureTriangle",
              "AXRow", "AXCell", "AXCheckBox", "AXSwitch", "AXToggle", "AXComboBox", "AXIncrementor"}
TYPABLE = {"AXTextField", "AXSearchField", "AXTextArea", "AXComboBox"}
FAKE_TEXT = ["meeting notes", "grocery list", "Lisbon", "budget 2025", "Sam Rivera", "pasta recipe", "weekly plan",
             "hello world", "project ideas", "weather tomorrow", "jazz", "Ithaca", "birthday party", "tax checklist"]
# Screen sizes of the HiDPI virtual display (scripts/sandbox/virtual_display.m), cycled per round.
SIZES = [(1440, 900), (1280, 800), (1680, 1050), (1920, 1080)]
VDISPLAY = os.environ.get("RLCA_VDISPLAY", "983D324D-A931-4414-8898-7E1951B77892")
DISPLAYPLACER = "/opt/homebrew/bin/displayplacer"
ESC, CMD_LBRACKET = 53, 33


# Apps never explored: they change the machine or the run (terminals, accessibility and sharing tools, migration),
# or are covered elsewhere (Safari runs the site list). Everything else installed is fair game: more apps, more data.
SKIP_APPS = {"Terminal", "iTerm", "iTerm2", "Migration Assistant", "Boot Camp Assistant", "Keychain Access", "Passwords",
             "VoiceOver Utility", "Accessibility Inspector", "Directory Utility", "Bluetooth File Exchange",
             "Wireless Diagnostics", "Feedback Assistant", "Ticket Viewer", "Screen Sharing", "System Information",
             "Python Launcher", "Setup Assistant", "Install macOS", "App Store", "Apps", "iPhone Mirroring", "Safari"}


def discover_apps():
    """APPS plus every .app installed in /Applications and /System/Applications (one level, and Utilities)."""
    found = set(APPS)
    for d in ("/Applications", "/System/Applications", "/System/Applications/Utilities", "/Applications/Utilities"):
        for p in Path(d).glob("*.app"):
            found.add(p.stem)
    return sorted(a for a in found if a not in SKIP_APPS and not a.startswith(("ENA_", "Install ", "Uninstall")))


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def set_dark(dark):
    sh(f"osascript -e 'tell application \"System Events\" to tell appearance preferences to set dark mode to {str(dark).lower()}'")


def running(name):
    # runningApplications only refreshes when the run loop turns; a plain script never turns it,
    # so apps launched after the first call would never show up.
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        if a.localizedName() == name:
            return a
    return None


def launch(name):
    if not running(name):
        if subprocess.run(["open", "-a", name], capture_output=True).returncode != 0:
            return None
        time.sleep(4)
    app = running(name)
    if app:
        app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        time.sleep(1)
    return app


KEEP_RUNNING = {"Finder"}
DIALOG_HELPERS = ["UserNotificationCenter", "Problem Reporter", "AddPrinter", "CoreServicesUIAgent", "storeuid"]


def sweep():
    """Close what clicking left behind between apps: other apps it opened (Add Printer, Transmission...), crash
    reports, permission prompts and the Force Quit window. They pile up over hours, cover the screen and stall AX."""
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.2))
    for a in NSWorkspace.sharedWorkspace().runningApplications():
        name, exe = a.localizedName(), str(a.executableURL().lastPathComponent() if a.executableURL() else "")
        if a.processIdentifier() == os.getpid():          # this script is a Python.app, so it counts as a regular app
            continue
        if (a.activationPolicy() == 0 and name not in KEEP_RUNNING) or name in DIALOG_HELPERS or exe in DIALOG_HELPERS:
            a.forceTerminate()                                                   # policy 0 = regular (Dock) app
    for name in DIALOG_HELPERS:
        sh(["pkill", "-9", "-x", name])
    sh(["pkill", "-9", "-f", "/storeuid.app/"])          # App Store sign-in sheet; ignores terminate
    # Persistent notifications ("Time Machine can back up...", "X added an extension") stack up in the corner of
    # every screenshot; dropping the delivered-notifications store and restarting the daemons clears them.
    for f in (Path.home() / "Library/Group Containers/group.com.apple.usernoted/db2").glob("db*"):
        f.unlink(missing_ok=True)
    sh("pkill -x usernoted; pkill -x NotificationCenter")
    # Finder windows and the Force Quit window (loginwindow) can't be quit; press their close buttons over AX
    # (AppleScript to Finder would raise an Automation permission prompt).
    wins = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, 0) or []
    pids = {w["kCGWindowOwnerPID"] for w in wins if w.get("kCGWindowLayer", 0) <= 8 and (
        w.get("kCGWindowOwnerName") == "Finder" or w.get("kCGWindowName") == "Force Quit Applications")}
    for _ in range(4):                                   # closing one Finder window can reveal another
        for pid in pids:
            for aw in attr(AXUIElementCreateApplication(pid), "AXWindows") or []:
                if (btn := attr(aw, "AXCloseButton")) is not None:
                    AXUIElementPerformAction(btn, "AXPress")
        time.sleep(0.3)
    time.sleep(1)


def quit_app(name):
    sh(f"osascript -e 'tell application \"{name}\" to quit saving no'", timeout=10)
    time.sleep(1)
    if (a := running(name)):
        a.forceTerminate()


def main_window(pid):
    # all_windows finds windows on every Space but misses some (Notes, Calendar...); AXWindows has those.
    app = AXUIElementCreateApplication(pid)
    cands = list(all_windows(pid)) + [w for w in attr(app, "AXWindows") or [] if attr(w, "AXSubrole") == "AXStandardWindow"]
    wins = [w for w in cands if (f := frame_of(w)) and f[2] > 200 and f[3] > 150 and not attr(w, "AXMinimized")]
    focused = attr(app, "AXFocusedWindow")
    return focused if focused is not None and focused in wins else (wins[0] if wins else None)


def set_screen(size):
    """Switch the virtual display to a resolution; it stays the main display at (0, 0)."""
    sh([DISPLAYPLACER, f"id:{VDISPLAY} res:{size[0]}x{size[1]} scaling:on origin:(0,0) degree:0"])
    time.sleep(2)


def resize(win, screen):
    """A window size people actually use: full screen, or a random 60-95% of it, below the menu bar."""
    sw, sh_ = screen[0], screen[1] - 25 - 80          # menu bar and dock
    frac = 1.0 if random.random() < 0.35 else random.uniform(0.6, 0.95)
    w, h = int(sw * frac), int(sh_ * frac)
    x, y = random.randint(0, sw - w), 25 + random.randint(0, sh_ - h)
    AXUIElementSetAttributeValue(win, "AXPosition", AXValueCreate(kAXValueCGPointType, Quartz.CGPointMake(x, y)))
    AXUIElementSetAttributeValue(win, "AXSize", AXValueCreate(kAXValueCGSizeType, Quartz.CGSizeMake(w, h)))


def scroll(win, dy=-6):
    f = frame_of(win)
    if f:
        ev = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine, 1, dy)
        Quartz.CGEventSetLocation(ev, Quartz.CGPointMake(f[0] + f[2] / 2, f[1] + f[3] / 2))
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def web_url(win):
    wa = E.find_role(win, "AXWebArea")
    u = attr(wa, "AXURL") if wa is not None else None
    return str(u.absoluteString()) if u is not None and hasattr(u, "absoluteString") else (str(u) if u else "")


# Pixel clicks: AX-poor apps (Blender, GIMP, Inkscape...) expose almost nothing to press, so the explorer also clicks
# GPA-GUI-Detector boxes. Kev loses most on icon-only targets that only the detector finds, so these screens are the
# training data it lacks. Never in apps where an unlabeled toggle could break the machine.
AX_THIN = 8
NO_PIXEL_APPS = {"System Settings", "Disk Utility", "Activity Monitor", "Console", "Terminal", "Script Editor"}
GPA_COREML = Path.home() / ".cache/rlca/coreml/gpa_640.mlpackage"


class Collector:
    def __init__(self, out, per_app_new, patience):
        self.run = E.Run(Path(out), dry=False)
        self.per_app_new, self.patience = per_app_new, patience
        self.dark, self.size = False, SIZES[0]
        self.sigs = set()
        self.gpa = None

    def pixel_boxes(self, win, sig):
        """GPA boxes (conf >= 0.15) on the screenshot just saved for `sig`, in screen points; [] if unavailable."""
        if not GPA_COREML.exists():
            return []
        from PIL import Image
        if self.gpa is None:
            from ultralytics import YOLO
            self.gpa = YOLO(str(GPA_COREML), task="detect")
        f = frame_of(win)
        if not f:
            return []
        img = Image.open(self.run.out / self.run.seen[sig]).convert("RGB")
        s = img.width / f[2]
        r = self.gpa.predict(img, imgsz=640, conf=0.15, device="cpu", verbose=False)[0]
        return [(f[0] + x1 / s, f[1] + y1 / s, (x2 - x1) / s, (y2 - y1) / s) for x1, y1, x2, y2 in r.boxes.xyxy.tolist()]

    def mode(self):
        return f"mode={'dark' if self.dark else 'light'}&size={self.size[0]}x{self.size[1]}"

    def capture(self, target, app_name, win, url):
        got = self.run.capture(target, app_name, window=win, url=url)
        if got is None:
            return None, False
        new = got[3] not in self.sigs
        self.sigs.add(got[3])
        return got, new

    def act(self, app_name, win, elements, handles, web, sig):
        """One action; -> (kind, element or None, pixel box or None)."""
        if not web and app_name not in NO_PIXEL_APPS and random.random() < (0.7 if len(elements) < AX_THIN else 0.15):
            try:
                boxes = self.pixel_boxes(win, sig)
            except Exception as ex:            # a detector failure shouldn't stop collection
                print(f"gpa failed: {ex}", flush=True)
                boxes = []
            f = frame_of(win)
            boxes = [b for b in boxes if f and f[0] <= b[0] + b[2] / 2 <= f[0] + f[2] and f[1] + 28 <= b[1] + b[3] / 2 <= f[1] + f[3]]
            if boxes:
                b = random.choice(boxes)
                X.click(b[0] + b[2] / 2, b[1] + b[3] / 2)
                return "pixel_click", None, [round(v, 1) for v in b]
        kind, e = self._act(app_name, win, elements, handles, web)
        return kind, e, None

    def _act(self, app_name, win, elements, handles, web):
        """One AX-driven action; -> (kind, element or None)."""
        r = random.random()
        if r < 0.08:
            X.key(ESC)
            return "escape", None
        if r < 0.14:
            scroll(win, random.choice([-8, 8]))
            return "scroll", None
        pairs = list(zip(elements, handles))
        typable = [(e, h) for e, h in pairs if e.role in TYPABLE and e.enabled]
        if r < 0.30 and typable:
            e, h = random.choice(typable)
            if not (web and WEB_DENY.search(" ".join(e.context) + " " + e.label)):
                AXUIElementSetAttributeValue(h, "AXFocused", True)
                AXUIElementSetAttributeValue(h, "AXValue", random.choice(FAKE_TEXT))
                return "type", e
        cands = [(e, h) for e, h in pairs if e.enabled and e.role in SAFE_PRESS and e.label
                 and not MACHINE_DENY.search(e.label) and not MACHINE_DENY.search(" ".join(e.context))
                 and not (web and WEB_DENY.search(e.label))]
        if web:
            cands = [(e, h) for e, h in cands if e.frame and web and E.inside(e.frame, web)]
        if not cands:
            X.go_back()
            return "back", None
        e, h = E.pick(self.run, app_name, cands)
        how = X.activate_element(h, e.frame)
        return how or "noop", e

    def explore(self, target, app_name, start_url=None, max_steps=150):
        app = launch(app_name)
        if app is None:
            return "not installed"
        rlca_ax._timeouts = 0              # the hung-app count starts fresh for each app
        pid = app.processIdentifier()
        if start_url:
            sh(["open", "-a", app_name, start_url])
            time.sleep(4)
        new_total, since_new = 0, 0
        for step in range(max_steps):
            if X.mouse_in_corner():
                return "mouse in corner"
            if NSWorkspace.sharedWorkspace().frontmostApplication().localizedName() != app_name:
                app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
                time.sleep(0.8)
            win = main_window(pid)
            if win is None:
                # A reopen (clicking the Dock icon) brings back the main window; Cmd-N as a fallback.
                subprocess.run(["open", "-a", app_name], capture_output=True)
                time.sleep(2)
                if (win := main_window(pid)) is None:
                    X.key(45, cmd=True)
                    time.sleep(2)
                if (win := main_window(pid)) is None:
                    return "no window"
            f = frame_of(win)
            if step == 0 or (f and f[0] >= self.size[0]):      # new windows can open on the 1024x768 display
                resize(win, self.size)
                time.sleep(0.8)
            url = web_url(win) if start_url else ""
            if start_url and url and E.domain(url) != E.domain(start_url):
                sh(["open", "-a", app_name, start_url])
                time.sleep(4)
                continue
            base = url.split("#")[0] if url else f"sandbox://{app_name.replace(' ', '_')}"
            got, new = self.capture(target, app_name, win, f"{base}{'&' if '?' in base else '?'}{self.mode()}")
            if got is None:
                return "no capture"
            new_total += new
            since_new = 0 if new else since_new + 1
            if new_total >= self.per_app_new or since_new >= self.patience:
                return f"done: {new_total} new"
            sid, elements, handles, sig, _, web = got
            t0 = time.time()
            kind, e, box = self.act(app_name, win, elements, handles, web if start_url else None, sig)
            time.sleep(1.5 if start_url else 1.0)
            self.run.log(before=sid, target=target, app=app_name, kind=kind, element=asdict(e) if e else None,
                         box=box, secs=round(time.time() - t0, 2), mode=self.mode())
        return f"steps out: {new_total} new"


def main():
    if os.environ.get("RLCA_SANDBOX") != "1":
        sys.exit("refusing: this presses irreversible controls; run only on a sandbox Mac with RLCA_SANDBOX=1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=20)
    ap.add_argument("--per_app_new", type=int, default=25, help="new screens per app per round before moving on")
    ap.add_argument("--patience", type=int, default=20, help="steps without a new screen before moving on")
    ap.add_argument("--reseed_every", type=int, default=4, help="rounds between content reseeds")
    ap.add_argument("--out", default=str(ROOT / "data/sandbox" / datetime.now().strftime("%Y%m%d-%H%M%S")))
    args = ap.parse_args()
    # A hung app otherwise blocks an AX call forever (seen: 40+ min stall); set on system-wide it applies globally.
    # 3 s was too short: heavy apps (Blender, Krita, OBS) and big pages in Safari answer slowly while loading on an M1.
    AXUIElementSetMessagingTimeout(AXUIElementCreateSystemWide(), 8.0)
    rlca_ax.HUNG_LIMIT = 3     # and three timeouts in a row abandon the app instead of walking its whole tree
    c = Collector(args.out, args.per_app_new, args.patience)
    deadline, rnd = time.time() + args.hours * 3600, 0
    while time.time() < deadline:
        c.dark = rnd % 2 == 1
        c.size = SIZES[(rnd // 2) % len(SIZES)]
        set_dark(c.dark)
        set_screen(c.size)
        if rnd and rnd % args.reseed_every == 0:
            sh(f"RLCA_SANDBOX=1 {ROOT}/scripts/vm/seed_content.sh", timeout=600)
        jobs = [("app", a) for a in discover_apps()] + [("site", s) for s in SITES]
        random.shuffle(jobs)
        for kind, what in jobs:
            if time.time() >= deadline:
                break
            try:
                if kind == "app":
                    why = c.explore(what.lower().replace(" ", "_"), what)
                    quit_app(what)
                else:
                    why = c.explore("web:" + E.domain(what), "Safari", start_url=what, max_steps=60)
            except Exception as ex:          # one bad app shouldn't end a 20-hour run
                why = f"error {type(ex).__name__}: {str(ex)[:100]}"
                if kind == "app" and isinstance(ex, rlca_ax.AXHung):
                    quit_app(what)
            print(f"{datetime.now():%H:%M} round {rnd} {c.mode()} {what}: {why} (states {c.run.n}, distinct {len(c.sigs)})",
                  flush=True)
            if why == "mouse in corner":
                return
            sweep()
        quit_app("Safari")
        rnd += 1


if __name__ == "__main__":
    main()
