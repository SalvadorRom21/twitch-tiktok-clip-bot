"""Helpers for vertical crop positioning."""

from __future__ import annotations


def crop_x_expression(face_center_x: float | None) -> str:
    """
    FFmpeg crop x-offset expression for 9:16 vertical crop.
    Centers on detected face when available, otherwise frame center.
    """
    if face_center_x is None:
        return "(iw-ih*9/16)/2"
    # Clamp face-centered crop within frame bounds
    return (
        f"max(0,min(iw-ih*9/16,iw*{face_center_x:.4f}-ih*9/32))"
    )
