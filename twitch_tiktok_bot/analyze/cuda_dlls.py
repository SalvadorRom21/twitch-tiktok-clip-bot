"""Load NVIDIA CUDA DLLs from pip packages on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_DIRS_ADDED = False


def ensure_cuda_dll_paths() -> None:
    """Add nvidia-*-cu12 pip package bin dirs so ctranslate2 can load on Windows."""
    global _DLL_DIRS_ADDED
    if _DLL_DIRS_ADDED or sys.platform != "win32":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_packages.exists():
        return

    path_prefix: list[str] = []
    for bin_dir in sorted(site_packages.rglob("bin")):
        if not bin_dir.is_dir():
            continue
        bin_str = str(bin_dir)
        path_prefix.append(bin_str)
        try:
            os.add_dll_directory(bin_str)
        except OSError:
            pass

    if path_prefix:
        os.environ["PATH"] = os.pathsep.join(path_prefix + [os.environ.get("PATH", "")])

    _DLL_DIRS_ADDED = True
