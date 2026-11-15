"""Sample OS-Atlas desktop screenshots straight out of the remote split zip, without downloading all of it.

OS-Atlas's Windows desktop images are one ~71 GB zip split into byte chunks (windows_image_aa .. _ad) on the
Hugging Face hub. HTTP range requests over the chunks give a seekable file, so zipfile reads the central
directory from the end and pulls only the sampled members.

  python scripts/data/os_atlas_remote.py --plat windows --n 3000
  -> data/public/os_atlas/windows_images/<img_filename>, data/public/os_atlas/desktop_domain/windows_splited.json
"""
import argparse
import io
import json
import random
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
REPO = "OS-Copilot/OS-Atlas-data"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/desktop_domain/"
PARTS = {"windows": ["windows_image_aa", "windows_image_ab", "windows_image_ac", "windows_image_ad"]}


class RangeFile(io.RawIOBase):
    """Read-only seekable view over byte-split remote files, fetched with HTTP range requests."""

    def __init__(self, names):
        self.s = requests.Session()
        self.urls, self.starts, pos = [], [], 0
        for n in names:
            r = self.s.head(BASE + n, allow_redirects=True, timeout=60)
            size = int(r.headers["Content-Length"])
            self.urls.append(r.url)
            self.starts.append((pos, size))
            pos += size
        self.size, self.pos = pos, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, start, n):
        out = b""
        for url, (p0, sz) in zip(self.urls, self.starts):
            if n <= 0:
                break
            if start < p0 + sz and start + n > p0:
                a = max(start, p0) - p0
                b = min(start + n, p0 + sz) - p0
                for _ in range(5):
                    try:
                        r = self.s.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=120)
                        r.raise_for_status()
                        out += r.content
                        break
                    except requests.RequestException:
                        continue
                else:
                    raise IOError(f"range fetch failed {url} {a}-{b}")
        return out

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        data = self._get(self.pos, n)
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plat", default="windows", choices=PARTS)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    d = ROOT / "data/public/os_atlas"
    (d / "desktop_domain").mkdir(parents=True, exist_ok=True)
    js = Path(hf_hub_download(REPO, f"desktop_domain/{args.plat}_splited.json", repo_type="dataset",
                              local_dir=str(d)))
    screens = json.load(open(js))
    names = sorted({s["img_filename"] for s in screens})
    random.Random(0).shuffle(names)
    want = names[:args.n]
    out = d / f"{args.plat}_images"
    todo = [n for n in want if not (out / n).exists()]
    print(f"{len(screens)} screens / {len(names)} images, sampling {len(want)}, {len(todo)} to fetch", flush=True)

    zf = zipfile.ZipFile(io.BufferedReader(RangeFile(PARTS[args.plat]), buffer_size=1 << 20))
    index = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
    print(f"zip has {len(index)} files", flush=True)

    base = RangeFile(PARTS[args.plat])
    local = threading.local()

    def fetch(name):
        """One member: its local header, then the compressed bytes, in two range requests."""
        info = index.get(Path(name).name)
        if info is None:
            return 0
        if not hasattr(local, "s"):
            local.s = requests.Session()
        rf = RangeFile.__new__(RangeFile)
        rf.s, rf.urls, rf.starts, rf.size = local.s, base.urls, base.starts, base.size
        head = rf._get(info.header_offset, 30)
        n_name, n_extra = struct.unpack("<HH", head[26:30])
        raw = rf._get(info.header_offset + 30 + n_name + n_extra, info.compress_size)
        data = zlib.decompress(raw, -15) if info.compress_type == zipfile.ZIP_DEFLATED else raw
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return 1

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for k, ok in enumerate(pool.map(fetch, todo)):
            done += ok
            if k % 200 == 0:
                print(f"{k}/{len(todo)} fetched {done}", flush=True)
    print(f"done: {done} images", flush=True)


if __name__ == "__main__":
    main()
