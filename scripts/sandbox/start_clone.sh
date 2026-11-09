#!/bin/zsh
# First boot of a sandbox Mac cloned from the configured AMI: make the virtual display main, drop the
# data copied from the source Mac, reseed (content is randomised per hostname), and start a run.
# Run over SSH as ec2-user:  RLCA_SANDBOX=1 zsh scripts/sandbox/start_clone.sh [hours]
set -e
[[ "$RLCA_SANDBOX" == 1 ]] || { echo "refusing: sandbox Mac only"; exit 1; }
cd "$(dirname "$0")/../.."
HOURS=${1:-22}
GUI=(sudo launchctl asuser 501 sudo -u ec2-user)
until pgrep -x Finder >/dev/null; do sleep 5; done
launchctl print gui/501/rlca.vdisplay >/dev/null 2>&1 || launchctl bootstrap gui/501 ~/Library/LaunchAgents/rlca.vdisplay.plist
sleep 5
VID=$($GUI /opt/homebrew/bin/displayplacer list | awk '/Persistent screen id/{id=$4} /15 inch/{print id}')
REAL=$($GUI /opt/homebrew/bin/displayplacer list | awk '/Persistent screen id/{id=$4} /24 inch/{print id}')
$GUI /opt/homebrew/bin/displayplacer "id:$VID res:1440x900 scaling:on origin:(0,0) degree:0" "id:$REAL res:1024x768 origin:(1440,0) degree:0"
rm -rf data/sandbox
$GUI env RLCA_SANDBOX=1 zsh scripts/vm/seed_content.sh
OUT=$PWD/data/sandbox/$(hostname -s)-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
($GUI env RLCA_SANDBOX=1 RLCA_VDISPLAY="$VID" nohup .venv/bin/python scripts/sandbox/collect.py --hours "$HOURS" --out "$OUT" > "$OUT/collect.log" 2>&1 &)
echo "started $OUT (virtual display $VID)"
