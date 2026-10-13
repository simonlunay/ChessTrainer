"""Hide a window from screen capture so ChessTrainer never reads its own arrows.

Windows 10 version 2004+ supports ``SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)``:
the window stays visible on the monitor but is left out of screenshots, screen recordings and
GDI/DXGI captures (including mss). Controlled by ``Settings.hide_overlay_from_capture``; turn
it off to record demo GIFs. Other platforms: no-op (returns False).
"""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

WDA_NONE = 0x00
WDA_EXCLUDEFROMCAPTURE = 0x11


def set_excluded_from_capture(hwnd: int, excluded: bool) -> bool:
    """Apply or remove capture exclusion for a native window handle. True on success."""
    if sys.platform != "win32":
        return False
    user32 = ctypes.windll.user32
    user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    user32.SetWindowDisplayAffinity.restype = ctypes.c_bool
    ok = bool(
        user32.SetWindowDisplayAffinity(
            ctypes.c_void_p(hwnd), WDA_EXCLUDEFROMCAPTURE if excluded else WDA_NONE
        )
    )
    if not ok:
        log.warning("SetWindowDisplayAffinity failed (error %d)", ctypes.GetLastError())
    return ok


def apply_capture_setting(widget: object, hide: bool) -> bool:
    """Apply the setting to a Qt widget. Call after it is shown (it needs a native handle)."""
    win_id = getattr(widget, "winId", None)
    if win_id is None:
        return False
    return set_excluded_from_capture(int(win_id()), hide)
