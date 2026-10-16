"""Background station logger: STATION_ON at start, STATION_OFF at shutdown/logoff.

Started at user logon from the Startup folder (scripts/install_station_logger.ps1):

    .venv311\\Scripts\\pythonw.exe -m src.events.station_logger

A hidden top-level window receives WM_ENDSESSION when Windows shuts down or the
user logs off; a timer writes a heartbeat used to recover after a power loss.
"""

from __future__ import annotations

import ctypes
import logging
import sys

from src.events import station_log

log = logging.getLogger(__name__)

HEARTBEAT_MS = 30_000
MUTEX_NAME = "Local\\SmartSenseStationLogger"
WINDOW_CLASS = "SmartSenseStationLogger"

WM_DESTROY = 0x0002
WM_QUERYENDSESSION = 0x0011
WM_ENDSESSION = 0x0016
WM_TIMER = 0x0113
ERROR_ALREADY_EXISTS = 183


def _safe(func) -> None:
    try:
        func()
    except Exception:
        log.exception("%s failed", func.__name__)


def _run_message_loop() -> None:
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.DefWindowProcW.restype = LRESULT
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
    user32.SetTimer.restype = ctypes.c_size_t
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = wintypes.BOOL
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE

    def wndproc(hwnd, msg, wparam, lparam):
        if msg == WM_TIMER:
            _safe(station_log.heartbeat)
            return 0
        if msg == WM_QUERYENDSESSION:
            return 1
        if msg == WM_ENDSESSION:
            if wparam:
                _safe(station_log.station_off)
            return 0
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    callback = WNDPROC(wndproc)  # must stay referenced for the window's lifetime
    hinstance = kernel32.GetModuleHandleW(None)
    wc = WNDCLASSW(lpfnWndProc=callback, hInstance=hinstance, lpszClassName=WINDOW_CLASS)
    if not user32.RegisterClassW(ctypes.byref(wc)):
        raise ctypes.WinError(ctypes.get_last_error())
    # Top-level (never shown) window - message-only windows don't get WM_ENDSESSION.
    hwnd = user32.CreateWindowExW(0, WINDOW_CLASS, WINDOW_CLASS, 0, 0, 0, 0, 0, None, None, hinstance, None)
    if not hwnd:
        raise ctypes.WinError(ctypes.get_last_error())
    user32.SetTimer(hwnd, 1, HEARTBEAT_MS, None)

    msg = wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))


def main() -> int:
    if sys.platform != "win32":
        print("The station logger runs on Windows only.", file=sys.stderr)
        return 1

    log_dir = station_log.state_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "station_logger.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    mutex = kernel32.CreateMutexW(None, False, MUTEX_NAME)  # noqa: F841 - held until exit
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        log.info("Station logger already running, exiting")
        return 0

    try:
        station_log.station_on()
        _run_message_loop()
    except Exception:
        log.exception("Station logger crashed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
