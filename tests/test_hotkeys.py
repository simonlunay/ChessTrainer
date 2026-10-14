"""Global hotkeys: parsing, and registration + dispatch on Windows. No keys are pressed: the
test registers an unused combination (Ctrl+Alt+Shift+F23) and posts the WM_HOTKEY message that
Windows would send to the hotkey thread."""

from __future__ import annotations

import sys
import threading

import pytest

from chesstrainer.overlay.hotkeys import (
    DEFAULT_HOTKEYS,
    MOD_ALT,
    MOD_CONTROL,
    MOD_SHIFT,
    GlobalHotkeys,
    effective_hotkeys,
    parse_hotkey,
)


def test_parse_hotkey() -> None:
    assert parse_hotkey("Ctrl+Alt+O") == (MOD_CONTROL | MOD_ALT, ord("O"))
    assert parse_hotkey("ctrl + shift + f9") == (MOD_CONTROL | MOD_SHIFT, 0x78)
    assert parse_hotkey("F8") == (0, 0x77)  # function keys may stand alone
    assert parse_hotkey("Alt+Left") == (MOD_ALT, 0x25)
    for bad in ("", "O", "Shift+O", "Ctrl+Hyper+O", "Ctrl+Alt+Banana"):
        with pytest.raises(ValueError):
            parse_hotkey(bad)
    for combo in DEFAULT_HOTKEYS.values():
        parse_hotkey(combo)


def test_effective_hotkeys_overrides_and_disables() -> None:
    keys = effective_hotkeys({"flip": "Ctrl+Alt+B", "pause": "", "unknown": "Ctrl+Alt+U"})
    assert keys["flip"] == "Ctrl+Alt+B" and "pause" not in keys and "unknown" not in keys
    assert keys["toggle_overlay"] == DEFAULT_HOTKEYS["toggle_overlay"]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hotkeys")
def test_register_dispatch_and_conflict() -> None:
    got: list[str] = []
    event = threading.Event()

    def handler(action: str) -> None:
        got.append(action)
        event.set()

    combo = "Ctrl+Alt+Shift+F23"
    hk = GlobalHotkeys({"flip": combo, "pause": "Ctrl+Nope+P"}, handler)
    hk.start()
    try:
        assert hk.registered == {"flip": combo}
        assert "pause" in hk.failed
        other = GlobalHotkeys({"flip": combo}, lambda a: None)
        other.start()
        try:
            assert "already used" in other.failed["flip"]
        finally:
            other.stop()
        assert hk.post_test_message("flip")
        assert event.wait(2) and got == ["flip"]
    finally:
        hk.stop()
    # Unregistered on stop: the combination can be taken again.
    again = GlobalHotkeys({"flip": combo}, lambda a: None)
    again.start()
    try:
        assert again.registered == {"flip": combo}
    finally:
        again.stop()
