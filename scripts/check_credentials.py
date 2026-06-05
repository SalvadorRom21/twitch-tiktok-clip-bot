#!/usr/bin/env python3
"""Verify Twitch credentials are loaded."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twitch_tiktok_bot.config import load_config


def main() -> int:
    config = load_config(project_root=Path.cwd())
    t = config.twitch
    checks = {
        "client_id": bool(t.client_id.strip()),
        "client_secret": bool(t.client_secret.strip()),
        "broadcaster_id": bool(t.broadcaster_id.strip()),
    }
    print("Credential check:")
    for name, ok in checks.items():
        print(f"  {name}: {'OK' if ok else 'MISSING'}")

    if all(checks.values()):
        print("\nAll Twitch credentials loaded.")
        return 0

    print("\nFix: edit config.local.yaml or .env in the project root.")
    print("  copy config.local.yaml.example config.local.yaml")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
