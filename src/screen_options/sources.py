"""Merge the three element sources into one candidate list, each item tagged with where it came from.

- ax:  controls the app declares (role, label, frame). Unlabelled ones take the OCR text inside them.
- ocr: text lines no AX control covers (links and buttons in apps with thin trees).
- det: boxes the pixel detector finds. One that lands on an AX or OCR item only adds its tag;
       one that lands on nothing is an icon the other two missed, and needs a caption.
"""
from dataclasses import dataclass, field


@dataclass
class Item:
    box: tuple                       # (x, y, w, h) in screen points
    label: str = ""
    role: str = ""
    sources: set = field(default_factory=set)
    context: list = field(default_factory=list)
    ax_index: int | None = None      # index into the AX element list, for pressing through AX
    conf: float = 0.0                # highest detector confidence among the detections merged into this item

    def describe(self):
        tag = "+".join(sorted(self.sources))
        role = self.role.removeprefix("AX").lower() or "item"
        ctx = f" · in: {' › '.join(self.context)}" if self.context else ""
        return f'{role} "{self.label}"{ctx} [{tag}]'


def center(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def contains(outer, pt, pad=2):
    return outer[0] - pad <= pt[0] <= outer[0] + outer[2] + pad and outer[1] - pad <= pt[1] <= outer[1] + outer[3] + pad


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def area(b):
    return b[2] * b[3]


def smallest_containing(items, pt, max_area=None):
    best = None
    for i, it in enumerate(items):
        if contains(it.box, pt) and (max_area is None or area(it.box) <= max_area):
            if best is None or area(it.box) < area(items[best].box):
                best = i
    return best


def merge(ax_elements, ocr, det, window_frame):
    """-> (items, icon indices that still need a caption)."""
    win_area = area(window_frame)
    items = [Item(box=e.frame, label=e.label, role=e.role, sources={"ax"}, context=e.context, ax_index=i)
             for i, e in enumerate(ax_elements) if e.frame]
    n_ax = len(items)
    # Controls bigger than a third of the window are containers in practice; text inside them stays its own item.
    for line in ocr:
        hit = smallest_containing(items[:n_ax], center(line["box"]), max_area=win_area / 3)
        if hit is not None:
            it = items[hit]
            it.sources.add("ocr")
            if not it.label:
                it.label = line["text"]
            continue
        items.append(Item(box=line["box"], label=line["text"], role="text", sources={"ocr"}))
    need_caption = []
    # Item boxes as arrays (grown as icons are added), so each detection is matched against all items at once:
    # the same tests as iou() / contains() / area() below, vectorised (hundreds of boxes per screen).
    import numpy as np
    B = np.zeros((len(items) + len(det), 4))
    for i, it in enumerate(items):
        B[i] = it.box
    n = len(items)
    for d in det:
        b = d["box"]
        c = d.get("conf", 0.0)
        X, Y, Wd, Ht = B[:n, 0], B[:n, 1], B[:n, 2], B[:n, 3]
        if n:
            ix = np.maximum(0, np.minimum(X + Wd, b[0] + b[2]) - np.maximum(X, b[0]))
            iy = np.maximum(0, np.minimum(Y + Ht, b[1] + b[3]) - np.maximum(Y, b[1]))
            inter = ix * iy
            union = Wd * Ht + b[2] * b[3] - inter
            ious = np.divide(inter, union, out=np.zeros(n), where=union != 0)
            best = int(np.argmax(ious))
            if ious[best] > 0.3:
                items[best].sources.add("det")
                items[best].conf = max(items[best].conf, c)
                continue
            cx, cy = X + Wd / 2, Y + Ht / 2
            ok = ((b[0] - 2 <= cx) & (cx <= b[0] + b[2] + 2) & (b[1] - 2 <= cy) & (cy <= b[1] + b[3] + 2)
                  & (Wd * Ht <= area(b) * 1.5))
            inside = np.flatnonzero(ok).tolist()
        else:
            inside = []
        if inside:
            for i in inside:
                items[i].sources.add("det")
                items[i].conf = max(items[i].conf, c)
            continue
        hit = smallest_containing(items[:n_ax], center(b), max_area=area(b) * 4)
        if hit is not None:
            items[hit].sources.add("det")
            items[hit].conf = max(items[hit].conf, c)
            continue
        items.append(Item(box=b, role="icon", sources={"det"}, conf=c))
        B[n] = b
        n += 1
        need_caption.append(len(items) - 1)
    return items, need_caption
