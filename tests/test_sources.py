"""Merge rules for AX + OCR + detector options (screen_options.sources.merge)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from screen_options.elements import Element  # noqa: E402
from screen_options.sources import iou, merge  # noqa: E402

WINDOW = (0, 0, 1000, 800)


def ax(label, frame, role="AXButton"):
    return Element(id=label or "e", source="window", role=role, label=label, frame=frame)


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_ocr_inside_ax_control_tags_it_and_fills_missing_label():
    items, _ = merge([ax("", (10, 10, 100, 30))], [{"box": (20, 15, 50, 20), "text": "Save"}], [], WINDOW)
    assert len(items) == 1
    assert items[0].label == "Save"
    assert items[0].sources == {"ax", "ocr"}


def test_ocr_outside_ax_becomes_its_own_option():
    items, _ = merge([ax("Save", (10, 10, 100, 30))], [{"box": (500, 500, 60, 20), "text": "Pricing"}], [], WINDOW)
    assert [it.label for it in items] == ["Save", "Pricing"]
    assert items[1].sources == {"ocr"}


def test_detector_box_on_ax_control_only_adds_tag():
    items, need = merge([ax("Star", (10, 10, 24, 24))], [], [{"box": (11, 11, 23, 23)}], WINDOW)
    assert len(items) == 1 and items[0].sources == {"ax", "det"}
    assert need == []


def test_detector_box_on_nothing_is_a_new_icon_needing_caption():
    items, need = merge([ax("Save", (10, 10, 100, 30))], [], [{"box": (600, 400, 24, 24)}], WINDOW)
    assert items[-1].role == "icon" and items[-1].sources == {"det"}
    assert need == [len(items) - 1]


def test_pixel_only_merge_without_ax():
    items, need = merge([], [{"box": (20, 15, 50, 20), "text": "Inbox"}], [{"box": (600, 400, 24, 24)}], WINDOW)
    assert {it.label for it in items} == {"Inbox", ""}
    assert len(need) == 1
