#!/bin/zsh
# One-time setup of a fresh EC2 Mac as a data-collection sandbox. Run over SSH as ec2-user:
#   RLCA_SANDBOX=1 VNC_PASS=... zsh scripts/sandbox/setup_mac.sh
# Gives ec2-user a password, turns on screen sharing (for the one-time permission grants over an
# SSH tunnel) and auto-login (the collector needs a real GUI session), keeps the Mac awake, installs
# Python deps and a few free no-sign-in apps, and seeds fake content.
set -e
[[ "$RLCA_SANDBOX" == 1 ]] || { echo "refusing: sandbox Mac only"; exit 1; }
[[ -n "$VNC_PASS" ]] || { echo "set VNC_PASS"; exit 1; }
cd "$(dirname "$0")/../.."

sudo dscl . -passwd /Users/ec2-user "$VNC_PASS"
sudo launchctl enable system/com.apple.screensharing
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.screensharing.plist 2>/dev/null || true

# Auto-login: loginwindow reads the password from /etc/kcpassword, XORed with a fixed key.
python3 - "$VNC_PASS" <<'EOF'
import sys, subprocess
key = [0x7D, 0x89, 0x52, 0x23, 0xD2, 0xBC, 0xDD, 0xEA, 0xA3, 0xB9, 0x1F]
pw = sys.argv[1].encode()
pw += b"\0" * (12 - len(pw) % 12)
enc = bytes(b ^ key[i % len(key)] for i, b in enumerate(pw))
subprocess.run(["sudo", "tee", "/etc/kcpassword"], input=enc, stdout=subprocess.DEVNULL, check=True)
EOF
sudo chmod 600 /etc/kcpassword
sudo defaults write /Library/Preferences/com.apple.loginwindow autoLoginUser ec2-user

sudo pmset -a sleep 0 displaysleep 0 disksleep 0
defaults -currentHost write com.apple.screensaver idleTime 0
defaults write com.apple.screensaver askForPassword 0 || true

export HOMEBREW_NO_AUTO_UPDATE=1
brew install python@3.12 >/dev/null
PY=$(brew --prefix python@3.12)/bin/python3.12
$PY -m venv .venv
.venv/bin/pip install -q pyobjc-core pyobjc-framework-Quartz pyobjc-framework-Cocoa \
  pyobjc-framework-ApplicationServices pyobjc-framework-Vision pillow
# GPA-GUI-Detector for pixel clicks in AX-poor apps; collect.py loads ~/.cache/rlca/coreml/gpa_640.mlpackage (copy it in).
.venv/bin/pip install -q ultralytics coremltools
# Free apps that work without signing in. Weighted toward pro/creative tools whose AX trees are thin (Kev's weakest
# screens) plus Electron, Java and Qt apps; collect.py explores every installed app, so more here means more data.
CASKS=(visual-studio-code libreoffice firefox vlc gimp inkscape obsidian blender krita audacity obs darktable freecad
  musescore shotcut kdenlive qgis godot scribus zotero anki calibre thunderbird joplin drawio arduino-ide sublime-text
  handbrake transmission keka dbeaver-community gnucash openscad iina logseq zettlr marktext texstudio lmms cyberduck
  filezilla bruno intellij-idea-ce pycharm-ce)
for cask in "${CASKS[@]}"; do
  brew install --cask "$cask" >/dev/null 2>&1 && echo "installed $cask" || echo "skipped $cask"
done
# Without this each app's first launch stops on a "downloaded from the internet" prompt.
sudo xattr -dr com.apple.quarantine /Applications/*.app 2>/dev/null || true
echo "setup done; reboot for auto-login"
