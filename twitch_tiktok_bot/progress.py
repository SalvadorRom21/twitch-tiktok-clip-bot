"""User-facing progress feedback for long pipeline steps."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


@contextmanager
def step(
    label: str,
    *,
    heartbeat_sec: float = 15.0,
    detail: str = "",
) -> Iterator[Callable[[str], None]]:
    """
    Print step start, optional heartbeat while running, and elapsed time on exit.

    Yields an `update(detail)` callable for inline status lines.
    """
    prefix = f"       " if label.startswith("  ") else ""
    header = f"{label}..." if not label.endswith("...") else label
    if detail:
        header = f"{header[:-3]} — {detail}..."
    print(header, flush=True)
    started = time.monotonic()
    stop = threading.Event()

    def _heartbeat() -> None:
        while not stop.wait(heartbeat_sec):
            elapsed = fmt_elapsed(time.monotonic() - started)
            print(f"{prefix}... still running ({elapsed})", flush=True)

    thread: threading.Thread | None = None
    if heartbeat_sec > 0:
        thread = threading.Thread(target=_heartbeat, daemon=True)
        thread.start()

    def update(msg: str) -> None:
        elapsed = fmt_elapsed(time.monotonic() - started)
        print(f"{prefix}... {msg} ({elapsed})", flush=True)

    try:
        yield update
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=0.2)
        elapsed = fmt_elapsed(time.monotonic() - started)
        print(f"{prefix}done ({elapsed})", flush=True)


def run_ffmpeg(
    cmd: list[str],
    *,
    label: str = "encoding",
    duration_sec: float | None = None,
) -> None:
    """Run FFmpeg and stream progress when duration is known."""
    full_cmd = list(cmd)
    if duration_sec and duration_sec > 0:
        insert_at = len(full_cmd) - 1
        extra = ["-progress", "pipe:1", "-nostats", "-loglevel", "error"]
        full_cmd = full_cmd[:insert_at] + extra + full_cmd[insert_at:]

    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE if duration_sec else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    stderr_chunks: list[str] = []

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        for chunk in iter(proc.stderr.readline, ""):
            stderr_chunks.append(chunk)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    last_pct = -1
    if duration_sec and proc.stdout is not None:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("out_time_ms="):
                continue
            try:
                out_ms = int(line.split("=", 1)[1])
            except ValueError:
                continue
            pct = min(100, int(out_ms / (duration_sec * 1_000_000) * 100))
            if pct != last_pct and pct % 5 == 0:
                last_pct = pct
                elapsed = out_ms / 1_000_000
                sys.stdout.write(
                    f"\r       {label}: {pct:3d}% "
                    f"({fmt_elapsed(elapsed)} / {fmt_elapsed(duration_sec)})"
                )
                sys.stdout.flush()
        if last_pct >= 0:
            sys.stdout.write("\n")
            sys.stdout.flush()

    code = proc.wait()
    stderr_thread.join(timeout=2.0)
    stderr = "".join(stderr_chunks)
    if code != 0:
        raise RuntimeError(f"ffmpeg failed (exit {code}):\n{stderr}")
