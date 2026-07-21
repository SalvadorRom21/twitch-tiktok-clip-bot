"""CLI: import Merica hard-neg + dual-fight labels, optionally retrain ML."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twitch_tiktok_bot.labels.elden_ml.hardneg_packs import (  # noqa: E402
    apply_merica_hardneg_pack,
)
from twitch_tiktok_bot.labels.elden_ml.infer import clear_model_cache  # noqa: E402
from twitch_tiktok_bot.labels.elden_ml.train import train_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train", action="store_true", help="Retrain model after import")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()

    def prog(fields: dict) -> None:
        print(fields.get("message") or fields)

    result = apply_merica_hardneg_pack(
        args.data_dir,
        force=args.force,
        extract_video=False,
        on_progress=prog,
    )
    print(result)
    if args.train and not result.get("skipped"):
        metrics = train_model(
            args.data_dir,
            epochs=args.epochs,
            on_progress=prog,
        )
        clear_model_cache()
        print("train", metrics)


if __name__ == "__main__":
    main()
