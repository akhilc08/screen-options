"""Element records shared by the collector (macOS-only) and the model side (any platform)."""
from dataclasses import dataclass, field


@dataclass
class Element:
    id: str
    source: str            # "window" or "menubar"
    role: str
    label: str
    context: list = field(default_factory=list)   # nearest labelled ancestors, outermost first
    enabled: bool = True
    frame: tuple | None = None                     # (x, y, w, h) in screen points

    def describe(self):
        ctx = " › ".join(self.context)
        role = self.role.removeprefix("AX").lower()
        state = "" if self.enabled else " · disabled"
        return f'{role} "{self.label}"' + (f" · in: {ctx}" if ctx else "") + state
