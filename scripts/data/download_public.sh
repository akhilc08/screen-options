#!/bin/zsh
# Download public GUI grounding/agent datasets into data/public/ (git-ignored). Resumable: rerun to continue.
# Local (default): the screenshot-only sets, whose element lists come from Apple Vision OCR (macOS only).
# --big: Multimodal-Mind2Web and AndroidControl, which carry their own HTML/accessibility trees; meant to
# run on an AWS box next to training rather than on the laptop.
#   scripts/data/download_public.sh [--big]
set -e
cd "$(dirname "$0")/../.."
OUT=data/public
mkdir -p $OUT
PY=.venv/bin/python

hf() {   # hf <repo> <dir> [allow patterns...]
  $PY - "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, d, *allow = sys.argv[1:]
snapshot_download(repo, repo_type="dataset", local_dir=f"data/public/{d}", allow_patterns=allow or None, max_workers=4)
print("done", repo, flush=True)
EOF
}

if [[ "$1" != "--big" ]]; then
  hf rootsautomation/ScreenSpot screenspot
  hf OS-Copilot/ScreenSpot-v2 screenspot_v2
  hf likaixin/ScreenSpot-Pro screenspot_pro
  hf Writer/omniact omniact
  hf OS-Copilot/OS-Atlas-data os_atlas "desktop_domain/macos*" "desktop_domain/linux*" "README.md" "prompts.json"
  echo "local sets done"
  exit 0
fi

hf osunlp/Multimodal-Mind2Web mind2web

# AndroidControl lives on Google Cloud Storage, not the Hub.
mkdir -p $OUT/android_control
for f in $(curl -s "https://storage.googleapis.com/storage/v1/b/gresearch/o?prefix=android_control/&fields=items(name)" \
           | $PY -c "import json,sys; [print(i['name']) for i in json.load(sys.stdin)['items'] if not i['name'].endswith('/')]"); do
  curl -sSL -C - -o "$OUT/$f" "https://storage.googleapis.com/gresearch/$f"
  echo "done $f"
done
echo "all done"
