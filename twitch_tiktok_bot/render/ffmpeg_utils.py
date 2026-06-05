"""FFmpeg filter string helpers (Windows-safe)."""

from __future__ import annotations

from pathlib import Path

from twitch_tiktok_bot.render.crop import crop_x_expression


def escape_expr_commas(expr: str) -> str:
    """Escape commas inside a filter expression so -vf chains parse correctly."""
    return expr.replace(",", r"\,")


def crop_filter(face_center_x: float | None) -> str:
    x = escape_expr_commas(crop_x_expression(face_center_x))
    return f"crop=ih*9/16:ih:{x}:0"


def ass_filter_path(path: Path) -> str:
    """Format an ASS subtitle path for FFmpeg filters on Windows and Unix."""
    posix = path.resolve().as_posix()
    if len(posix) > 1 and posix[1] == ":":
        posix = posix[0] + r"\:" + posix[2:]
    return f"ass={posix}"


def join_video_filters(filters: list[str]) -> str:
    return ",".join(filters)
