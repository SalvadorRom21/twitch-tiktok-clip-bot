"""Launch cursor-sdk-bridge on Windows without selector-based stderr polling."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from cursor_sdk import Client
from cursor_sdk._bridge import (
    Bridge,
    BridgeEndpoint,
    _bridge_subprocess_env,
    _terminate_process,
    parse_discovery_line,
)
from cursor_sdk._vendor import resolve_bridge_path
from cursor_sdk.errors import CursorSDKError

if TYPE_CHECKING:
    from os import PathLike


def _launch_bridge_process(
    workspace: str | PathLike[str],
    timeout: float,
    *,
    cursor_api_key: str | None = None,
) -> tuple[Bridge, subprocess.Popen[str]]:
    argv = [resolve_bridge_path(), "--workspace", str(workspace)]
    env = dict(_bridge_subprocess_env())
    if cursor_api_key:
        env["CURSOR_API_KEY"] = cursor_api_key
    process = subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stderr is None:
        _terminate_process(process)
        raise CursorSDKError("Bridge process stderr is unavailable")

    discovery: dict | None = None
    stderr_lines: list[str] = []
    done = threading.Event()

    def read_stderr() -> None:
        nonlocal discovery
        for line in process.stderr:
            stderr_lines.append(line)
            parsed = parse_discovery_line(line)
            if parsed is not None:
                discovery = dict(parsed)
                done.set()
                return

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()

    deadline = time.monotonic() + timeout
    while not done.is_set():
        exit_code = process.poll()
        if exit_code is not None:
            done.wait(timeout=0.5)
            if discovery is None:
                _terminate_process(process)
                raise CursorSDKError(
                    f"Bridge exited before discovery with status {exit_code}: "
                    + "".join(stderr_lines)
                )
            break
        if time.monotonic() >= deadline:
            _terminate_process(process)
            raise CursorSDKError(
                "Timed out waiting for bridge discovery: " + "".join(stderr_lines)
            )
        time.sleep(0.05)

    reader.join(timeout=1.0)
    if discovery is None:
        _terminate_process(process)
        raise CursorSDKError(
            "Bridge did not emit discovery JSON: " + "".join(stderr_lines)
        )

    bridge = Bridge(BridgeEndpoint.from_discovery(discovery), process)
    return bridge, process


@contextmanager
def launch_sdk_client(
    workspace: str | PathLike[str],
    *,
    timeout: float = 90,
    cursor_api_key: str | None = None,
) -> Iterator[Client]:
    if sys.platform == "win32":
        bridge, _process = _launch_bridge_process(
            workspace, timeout, cursor_api_key=cursor_api_key
        )
        client = Client(bridge.endpoint, allow_api_key_env_fallback=True)
        client._owned_bridge = bridge
        try:
            yield client
        finally:
            client.close()
        return

    if cursor_api_key:
        os.environ["CURSOR_API_KEY"] = cursor_api_key
    with Client.launch_bridge(workspace=workspace, timeout=timeout) as client:
        yield client
