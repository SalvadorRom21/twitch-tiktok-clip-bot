"""Face-cam on top + gameplay below (TikTok streamer layout)."""

from __future__ import annotations

from twitch_tiktok_bot.analyze.face_cam import FaceCamRegion
from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.models import EditEffect


def gameplay_crop_exprs(region: FaceCamRegion | None) -> tuple[str, str, str, str]:
    """
    Return (w, h, x, y) FFmpeg expressions for the gameplay panel.
    Crops center gameplay, avoiding the face-cam overlay when known.
    """
    if region is None:
        return ("iw*0.82", "ih*0.92", "iw*0.09", "ih*0.02")

    cx = region.x + region.w / 2
    cy = region.y + region.h / 2
    top = cy < 0.5
    left = cx < 0.5

    if top and left:
        return ("iw*0.82", "ih*0.88", "iw*0.12", "ih*0.06")
    if top and not left:
        return ("iw*0.82", "ih*0.88", "iw*0.06", "ih*0.06")
    if not top and left:
        return ("iw*0.78", "ih*0.90", "iw*0.14", "ih*0.03")
    return ("iw*0.78", "ih*0.90", "iw*0.08", "ih*0.03")


def _zoom_on_gameplay(
    _effects: list[EditEffect],
    _out_w: int,
    _game_h: int,
    _zoom_scale: float,
    _zoom_dur: float,
) -> str:
    # zoompan causes judder on montage output — disabled for smooth playback.
    return ""


def face_panel_height(out_w: int, out_h: int, panel_ratio: float) -> int:
    """Top panel height — 16:9 at full output width, capped to leave room for gameplay."""
    aspect_h = int(round(out_w * 9 / 16))
    ratio_h = int(out_h * panel_ratio)
    return min(aspect_h, ratio_h, int(out_h * 0.42))


def build_stacked_filter(
    config: AppConfig,
    region: FaceCamRegion,
    effects: list[EditEffect],
    *,
    video_in: str = "[0:v]",
) -> str:
    """FFmpeg filter_complex chain: face panel on top, gameplay below."""
    render = config.render
    out_w = render.width
    out_h = render.height
    face_h = face_panel_height(out_w, out_h, render.face_panel_ratio)
    game_h = out_h - face_h

    fx = f"iw*{region.x:.4f}"
    fy = f"ih*{region.y:.4f}"
    fw = f"iw*{region.w:.4f}"
    fh = f"ih*{region.h:.4f}"

    gw, gh, gx, gy = gameplay_crop_exprs(region)
    zoom_filter = _zoom_on_gameplay(
        effects,
        out_w,
        game_h,
        config.editing.zoom_scale,
        config.editing.zoom_duration_sec,
    )

    # Fit the full webcam crop inside the panel (letterbox) — no zoom-to-fill cropping
    # of neon signs, chair backdrop, or chin. Gameplay bleed is excluded at detect time.
    face_chain = (
        f"crop={fw}:{fh}:{fx}:{fy},"
        f"scale={out_w}:{face_h}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{face_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    game_chain = (
        f"crop={gw}:{gh}:{gx}:{gy},"
        f"scale={out_w}:{game_h}:flags=lanczos:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{game_h}"
        f"{zoom_filter}"
    )

    return (
        f"{video_in}split=2[fv][gv];"
        f"[fv]{face_chain}[face];"
        f"[gv]{game_chain}[game];"
        f"[face][game]vstack=inputs=2,setsar=1,format=yuv420p[vout]"
    )
