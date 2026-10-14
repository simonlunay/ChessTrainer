"""System-wide hotkeys (Windows ``RegisterHotKey``), working while another window has focus.

A dedicated thread registers the hotkeys and runs its own message loop; each ``WM_HOTKEY`` calls
``on_hotkey(action)`` on that thread (the app forwards it to the GUI thread with a Qt signal).
Nothing is injected or hooked: Windows simply reserves the key combinations while the app runs.
A combination already taken by another program cannot be registered; that is reported in
``failed`` (action -> combination) instead of raising. On other platforms nothing is registered.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from collections.abc import Callable, Mapping
from ctypes import wintypes

log = logging.getLogger(__name__)

#: Default combinations (``Settings.hotkeys`` overrides them; empty string = disabled).
DEFAULT_HOTKEYS: dict[str, str] = {
    "toggle_overlay": "Ctrl+Alt+O",
    "toggle_explanations": "Ctrl+Alt+E",
    "pause": "Ctrl+Alt+P",
    "flip": "Ctrl+Alt+F",
    "switch_mode": "Ctrl+Alt+M",
}

ACTION_TEXT = {
    "toggle_overlay": "show / hide the arrows",
    "toggle_explanations": "explanations on / off",
    "pause": "pause / resume",
    "flip": "flip the board",
    "switch_mode": "switch mode (analysis, trainer, review)",
}

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
_MODS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT,
         "win": MOD_WIN}  # fmt: skip
_NAMED_KEYS = {
    "space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22, "insert": 0x2D, "delete": 0x2E, "left": 0x25, "up": 0x26,
    "right": 0x27, "down": 0x28, "pause": 0x13,
}  # fmt: skip


def parse_hotkey(text: str) -> tuple[int, int]:
    """``"Ctrl+Alt+O"`` -> (modifiers, virtual-key code). Raises ValueError."""
    parts = [p.strip().lower() for p in text.split("+") if p.strip()]
    if not parts:
        raise ValueError("empty hotkey")
    *mods, key = parts
    flags = 0
    for m in mods:
        if m not in _MODS:
            raise ValueError(f"unknown modifier {m!r} in {text!r}")
        flags |= _MODS[m]
    if len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        vk = 0x70 + int(key[1:]) - 1
    elif key in _NAMED_KEYS:
        vk = _NAMED_KEYS[key]
    else:
        raise ValueError(f"unknown key {key!r} in {text!r}")
    if not flags & (MOD_CONTROL | MOD_ALT | MOD_WIN) and not (0x70 <= vk <= 0x87):
        raise ValueError(f"{text!r} needs Ctrl, Alt or Win (or an F key) to be system-wide")
    return flags, vk


def effective_hotkeys(overrides: Mapping[str, str] | None) -> dict[str, str]:
    keys = dict(DEFAULT_HOTKEYS)
    for action, combo in (overrides or {}).items():
        if action in keys:
            keys[action] = combo
    return {a: c for a, c in keys.items() if c.strip()}


class GlobalHotkeys:
    def __init__(self, bindings: Mapping[str, str], on_hotkey: Callable[[str], None]) -> None:
        self.bindings = dict(bindings)
        self.on_hotkey = on_hotkey
        self.registered: dict[str, str] = {}
        self.failed: dict[str, str] = {}  # action -> "Ctrl+Alt+O (reason)"
        self._ids: dict[int, str] = {}
        self._thread_id = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)

    @property
    def supported(self) -> bool:
        return sys.platform == "win32"

    def start(self, timeout: float = 2.0) -> None:
        """Register the hotkeys; returns once registration is done (see ``failed``)."""
        if not self.supported:
            self.failed = dict.fromkeys(self.bindings, "not supported on this platform")
            return
        self._thread.start()
        self._ready.wait(timeout)

    def stop(self) -> None:
        if self._thread.is_alive() and self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=2)

    def post_test_message(self, action: str) -> bool:
        """Deliver a WM_HOTKEY for ``action`` to the loop, as Windows would (tests only)."""
        for hid, a in self._ids.items():
            if a == action:
                ok: int = ctypes.windll.user32.PostThreadMessageW(
                    self._thread_id, WM_HOTKEY, hid, 0
                )
                return bool(ok)
        return False

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = int(kernel32.GetCurrentThreadId())
        msg = wintypes.MSG()
        # Create this thread's message queue before anyone posts to it.
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        for i, (action, combo) in enumerate(self.bindings.items(), start=1):
            try:
                mods, vk = parse_hotkey(combo)
            except ValueError as err:
                self.failed[action] = f"{combo} ({err})"
                continue
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                self._ids[i] = action
                self.registered[action] = combo
            else:
                self.failed[action] = f"{combo} (already used by another program)"
        if self.failed:
            log.warning("hotkeys not registered: %s", self.failed)
        self._ready.set()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    hit = self._ids.get(int(msg.wParam))
                    if hit is not None:
                        try:
                            self.on_hotkey(hit)
                        except Exception:  # noqa: BLE001 - never kill the loop
                            log.exception("hotkey handler failed")
        finally:
            for hid in self._ids:
                user32.UnregisterHotKey(None, hid)
