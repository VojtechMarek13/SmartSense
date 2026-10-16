"""Station event log for the Power BI efficiency dashboard.

Writes one CSV per day into <Desktop>/Home/sf6/events:

    station_id,event_type,ts,boot_id,seq
    SF6,STATION_ON,2026-09-25T07:33:45,142,1

Events:
    STATION_ON   PC started (station logger launched at user logon)
    STATION_OFF  PC shut down / user logged off
    CYCLE_START  first SmartSense application (desktop or web) started
    CYCLE_DONE   last SmartSense application closed

boot_id is persistent and incremented on every station start; seq restarts at 1
on every station start and continues across the daily file rotation.

The station logger (src.events.station_logger) and the SmartSense app share
the counters through a JSON state file guarded by a file lock. The logger
records a heartbeat, so after a power loss the missing CYCLE_DONE /
STATION_OFF are written on the next start with the last heartbeat time.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

STATION_ID = os.environ.get("SMARTSENSE_STATION_ID", "SF6")
HEADER = "station_id,event_type,ts,boot_id,seq\n"
TS_FORMAT = "%Y-%m-%dT%H:%M:%S"

STATION_ON = "STATION_ON"
STATION_OFF = "STATION_OFF"
CYCLE_START = "CYCLE_START"
CYCLE_DONE = "CYCLE_DONE"

_DEFAULT_STATE = {
    "boot_id": 0,
    "seq": 0,
    "station_on": False,
    "last_alive": None,
    "cycle_pids": [],
}


def _desktop_dir() -> Path:
    if sys.platform == "win32":
        import ctypes

        # CSIDL_DESKTOPDIRECTORY - follows OneDrive / redirected desktops
        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0:
            return Path(buf.value)
    return Path.home() / "Desktop"


def events_dir() -> Path:
    override = os.environ.get("SMARTSENSE_EVENTS_DIR")
    return Path(override) if override else _desktop_dir() / "Home" / "sf6" / "events"


def state_dir() -> Path:
    # Kept outside the events folder so Power BI only sees the CSV files.
    override = os.environ.get("SMARTSENSE_EVENTS_STATE_DIR")
    if override:
        return Path(override)
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "SmartSense"


def _now() -> str:
    return datetime.now().strftime(TS_FORMAT)


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock(f) -> None:
    f.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock(f) -> None:
    f.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_state() -> Iterator[dict]:
    directory = state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "events_state.json"
    with open(directory / "events.lock", "a+b") as lock:
        _lock(lock)
        try:
            state = dict(_DEFAULT_STATE)
            if path.exists():
                state.update(json.loads(path.read_text(encoding="utf-8")))
            state["cycle_pids"] = list(state["cycle_pids"])
            try:
                yield state
            finally:
                # Saved even on failure so an already written seq is never reused.
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
                os.replace(tmp, path)
        finally:
            _unlock(lock)


def _emit(state: dict, event_type: str, ts: str | None = None) -> None:
    ts = ts or _now()
    state["seq"] += 1
    path = events_dir() / f"{STATION_ID}_events_{ts[:10]}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        if is_new:
            f.write(HEADER)
        f.write(f"{STATION_ID},{event_type},{ts},{state['boot_id']},{state['seq']}\n")
    log.info("%s boot_id=%s seq=%s ts=%s", event_type, state["boot_id"], state["seq"], ts)


def _close_cycle(state: dict, ts: str | None = None) -> None:
    if state["cycle_pids"]:
        _emit(state, CYCLE_DONE, ts)
        state["cycle_pids"] = []


def _prune_dead_apps(state: dict) -> None:
    """Close the cycle if every SmartSense app that held it crashed or was killed."""
    alive = [pid for pid in state["cycle_pids"] if _pid_alive(pid)]
    if not alive:
        _close_cycle(state)
    state["cycle_pids"] = alive


# --- station (called by the station logger) ---------------------------------


def station_on() -> None:
    with _locked_state() as state:
        if state["station_on"]:
            # Previous run ended without STATION_OFF (power loss, crash).
            last_alive = state["last_alive"] or _now()
            _close_cycle(state, last_alive)
            _emit(state, STATION_OFF, last_alive)
        state["boot_id"] += 1
        state["seq"] = 0
        state["station_on"] = True
        state["last_alive"] = _now()
        _emit(state, STATION_ON)


def station_off() -> None:
    with _locked_state() as state:
        _close_cycle(state)
        if state["station_on"]:
            _emit(state, STATION_OFF)
            state["station_on"] = False


def heartbeat() -> None:
    with _locked_state() as state:
        if not state["station_on"]:
            return
        state["last_alive"] = _now()
        _prune_dead_apps(state)


# --- cycle (called by the SmartSense app) ------------------------------------


def cycle_start(pid: int | None = None) -> bool:
    """Register a running SmartSense app; CYCLE_START is written only by the first one.

    Returns False if this pid is already registered.
    """
    pid = pid or os.getpid()
    with _locked_state() as state:
        _prune_dead_apps(state)
        if pid in state["cycle_pids"]:
            return False
        if not state["cycle_pids"]:
            if not state["station_on"]:
                log.warning("CYCLE_START without STATION_ON - is the station logger running?")
            _emit(state, CYCLE_START)
        state["cycle_pids"].append(pid)
        return True


def cycle_done(pid: int | None = None) -> None:
    """Unregister a SmartSense app; CYCLE_DONE is written when the last one closes."""
    pid = pid or os.getpid()
    with _locked_state() as state:
        if pid not in state["cycle_pids"]:
            return
        state["cycle_pids"] = [p for p in state["cycle_pids"] if p != pid and _pid_alive(p)]
        if not state["cycle_pids"]:
            _emit(state, CYCLE_DONE)


def app_started() -> bool:
    """cycle_start() for the apps; event logging errors never stop SmartSense."""
    try:
        return cycle_start()
    except Exception:
        log.exception("Failed to record CYCLE_START")
        return False


def app_stopped() -> None:
    try:
        cycle_done()
    except Exception:
        log.exception("Failed to record CYCLE_DONE")


@contextmanager
def smartsense_cycle() -> Iterator[None]:
    """Wrap the desktop app run."""
    started = app_started()
    try:
        yield
    finally:
        if started:
            app_stopped()
