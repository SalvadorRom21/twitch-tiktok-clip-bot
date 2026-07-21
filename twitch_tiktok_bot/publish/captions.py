"""Platform-specific caption / title formatting."""

from __future__ import annotations

import re

_HASHTAG_RE = re.compile(r"#(\w+)")


def _split_caption(caption: str) -> tuple[str, list[str]]:
    text = (caption or "").strip()
    tags = [m.group(1) for m in _HASHTAG_RE.finditer(text)]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    body_lines: list[str] = []
    for ln in lines:
        if ln.strip().startswith("#") and all(
            tok.startswith("#") for tok in ln.split()
        ):
            continue
        body_lines.append(ln)
    body = "\n".join(body_lines).strip()
    if body_lines:
        last = body_lines[-1]
        cleaned = _HASHTAG_RE.sub("", last).strip()
        if cleaned != last:
            body_lines[-1] = cleaned
            body = "\n".join(ln for ln in body_lines if ln.strip()).strip()
    return body, tags


def ensure_hashtags(tags: list[str], extras: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in [*tags, *extras]:
        t = tag.lstrip("#").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def tiktok_caption(caption: str, *, title: str = "") -> str:
    body, tags = _split_caption(caption)
    tags = ensure_hashtags(tags, ["fyp", "eldenring", "gaming"])
    hook = body.split("\n")[0].strip() if body else (title or "Boss fight").strip()
    rest = "\n".join(body.splitlines()[1:]).strip() if body else ""
    parts = [hook]
    if rest:
        parts.append(rest)
    parts.append("")
    parts.append(" ".join(f"#{t}" for t in tags[:12]))
    return "\n".join(parts).strip() + "\n"


def instagram_caption(caption: str, *, title: str = "") -> str:
    body, tags = _split_caption(caption)
    tags = ensure_hashtags(tags, ["reels", "eldenring", "gaming", "bossfight"])
    hook = body.split("\n")[0].strip() if body else (title or "Boss fight").strip()
    rest = "\n".join(body.splitlines()[1:]).strip() if body else ""
    parts = [hook]
    if rest:
        parts.append("")
        parts.append(rest)
    parts.append("")
    parts.append(" ".join(f"#{t}" for t in tags[:20]))
    return "\n".join(parts).strip() + "\n"


def youtube_title(caption: str, *, title: str = "") -> str:
    body, _ = _split_caption(caption)
    hook = body.split("\n")[0].strip() if body else (title or "Boss fight").strip()
    if "#shorts" not in hook.lower():
        if len(hook) > 90:
            hook = hook[:90].rstrip()
        hook = f"{hook} #Shorts"
    return hook[:100]


def youtube_description(caption: str, *, title: str = "") -> str:
    body, tags = _split_caption(caption)
    tags = ensure_hashtags(tags, ["Shorts", "EldenRing", "Gaming"])
    parts: list[str] = []
    if body:
        parts.append(body)
    elif title:
        parts.append(title)
    parts.append("")
    parts.append(" ".join(f"#{t}" for t in tags[:15]))
    return "\n".join(parts).strip() + "\n"
