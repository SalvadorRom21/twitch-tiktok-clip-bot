"""Cut dead air inside a full fight — keep the whole arc, drop boring middles."""

from __future__ import annotations

from pathlib import Path

from twitch_tiktok_bot.config import AppConfig
from twitch_tiktok_bot.labels.fights import load_fight_labels
from twitch_tiktok_bot.labels.literacy import load_literacy_store
from twitch_tiktok_bot.models import ClipAnalysis, EditPlan, EditSegment
from twitch_tiktok_bot.plan.action import (
    _peaks_in_range,
    peak_cluster_is_gunfire,
    segment_action_score,
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(len(ordered) * pct / 100.0)
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


def _silent_fraction(analysis: ClipAnalysis, start: float, end: float) -> float:
    span = max(end - start, 0.1)
    silent = 0.0
    for gap in analysis.silence_ranges:
        if gap.end < start or gap.start > end:
            continue
        silent += max(0.0, min(gap.end, end) - max(gap.start, start))
    return silent / span


def _window_action_score(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    *,
    apex: bool,
    fight_start: float = 0.0,
    fight_span: float = 1.0,
) -> float:
    score = segment_action_score(
        analysis, start, end, require_combat_for_apex=apex
    )
    peaks = _peaks_in_range(analysis.loud_peaks, start, end)
    if len(peaks) >= 3 and peak_cluster_is_gunfire(peaks):
        score += 4.0
    elif len(peaks) >= 1:
        score += min(2.0, len(peaks) * 0.6)
    score -= _silent_fraction(analysis, start, end) * 8.0

    if fight_span > 1.0:
        rel = (start - fight_start) / fight_span
        # Early push / first contact often scores quieter than the tail wipe.
        if rel < 0.4:
            score += 3.0 - rel * 5.0

    return score


def _merge_ranges(
    ranges: list[tuple[float, float]],
    *,
    merge_gap: float,
) -> list[tuple[float, float]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda item: item[0])
    merged: list[list[float]] = [[sorted_ranges[0][0], sorted_ranges[0][1]]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(item[0], item[1]) for item in merged]


def _score_windows_in_range(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    config: AppConfig,
    *,
    apex: bool,
    fight_start: float,
    fight_span: float,
) -> list[tuple[float, float, float]]:
    editing = config.editing
    step = editing.action_cut_scan_step_sec
    win = editing.action_cut_window_sec
    windows: list[tuple[float, float, float]] = []
    t = start
    while t < end - 0.5:
        w_end = min(end, t + win)
        if w_end - t < editing.action_cut_min_sec * 0.5:
            break
        score = _window_action_score(
            analysis,
            t,
            w_end,
            apex=apex,
            fight_start=fight_start,
            fight_span=fight_span,
        )
        windows.append((t, w_end, score))
        t += step
    return windows


def _split_long_span_at_cold_gaps(
    analysis: ClipAnalysis,
    start: float,
    end: float,
    config: AppConfig,
    *,
    apex: bool,
    fight_start: float,
    fight_span: float,
) -> list[tuple[float, float]]:
    """Break a long active span at internal low-action valleys (not tail-picking)."""
    editing = config.editing
    if end - start <= editing.action_cut_split_span_sec:
        return [(start, end)]

    windows = _score_windows_in_range(
        analysis,
        start,
        end,
        config,
        apex=apex,
        fight_start=fight_start,
        fight_span=fight_span,
    )
    if len(windows) < 3:
        return [(start, end)]

    scores = [item[2] for item in windows]
    cold_threshold = _percentile(scores, 30) - 1.0
    cold_gap = editing.action_cut_cold_gap_sec
    min_chunk = editing.action_cut_min_sec

    chunks: list[tuple[float, float]] = []
    chunk_start = start
    cold_start: float | None = None

    for win_start, win_end, score in windows:
        if score < cold_threshold:
            if cold_start is None:
                cold_start = win_start
        else:
            if cold_start is not None:
                gap_len = win_start - cold_start
                if gap_len >= cold_gap and win_start - chunk_start >= min_chunk:
                    chunks.append((chunk_start, cold_start))
                    chunk_start = win_end
                cold_start = None

    if chunk_start < end and end - chunk_start >= min_chunk * 0.75:
        chunks.append((chunk_start, end))

    return chunks if chunks else [(start, end)]


def _coverage_span(segments: list[EditSegment]) -> float:
    return sum(seg.end - seg.start for seg in segments)


def _fight_id_for_segment(
    work_dir: Path | None,
    fight: EditSegment,
) -> str:
    if work_dir is None:
        return ""
    store = load_fight_labels(work_dir)
    if not store:
        return ""
    mid = (fight.start + fight.end) / 2
    best_id = ""
    best_overlap = 0.0
    for label in store.fights:
        overlap = max(
            0.0,
            min(fight.end, label.end_sec) - max(fight.start, label.start_sec),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = label.id
    return best_id


def _literacy_spine_segments(
    work_dir: Path | None,
    fight: EditSegment,
    config: AppConfig,
) -> list[EditSegment] | None:
    """
    Build the edit from curator literacy beats (start → peaks → wipe),
    dropping uncaptured gaps between labeled moments.
    """
    if work_dir is None:
        return None

    store = load_literacy_store(work_dir)
    if not store:
        return None

    fight_id = _fight_id_for_segment(work_dir, fight)
    if not fight_id:
        return None

    moments = [
        moment
        for moment in store.moments
        if moment.answered
        and moment.fight_id == fight_id
        and moment.clip_worthy != "skip"
    ]
    if len(moments) < 3:
        return None

    editing = config.editing
    pad_before = editing.action_cut_pad_before_sec
    pad_after = editing.action_cut_pad_after_sec
    min_cut = editing.action_cut_min_sec
    merge_gap = editing.action_cut_merge_gap_sec

    raw: list[EditSegment] = []
    for moment in sorted(moments, key=lambda item: item.timestamp_sec):
        if moment.context_start_sec is not None and moment.context_end_sec is not None:
            start = max(fight.start, moment.context_start_sec)
            end = min(fight.end, moment.context_end_sec)
        else:
            start = max(fight.start, moment.timestamp_sec - pad_before)
            end = min(fight.end, moment.timestamp_sec + pad_after)

        if moment.event_type == "knock_self":
            span = min(end - start, 6.0)
            center = moment.timestamp_sec
            start = max(fight.start, center - span * 0.35)
            end = min(fight.end, start + span)

        if end - start < min_cut * 0.4:
            continue

        raw.append(
            EditSegment(
                start=start,
                end=end,
                reason="literacy_spine",
            )
        )

    if not raw:
        return None

    raw.sort(key=lambda seg: seg.start)
    merged: list[EditSegment] = []
    for seg in raw:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg.start <= prev.end + merge_gap:
            merged[-1] = EditSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                reason="literacy_spine",
            )
        else:
            merged.append(seg)

    if merged:
        merged = _bridge_spine_gaps_to_finale(
            merged, fight, config, work_dir=work_dir
        )

    return merged if merged else None


def _literacy_fight_end_cutoff(
    work_dir: Path | None,
    fight: EditSegment,
    config: AppConfig,
) -> float | None:
    """End the clip shortly after wipe / death box — before looting."""
    if work_dir is None:
        return None

    store = load_literacy_store(work_dir)
    if not store:
        return None

    fight_id = _fight_id_for_segment(work_dir, fight)
    wipe_types = {"team_wipe", "death_box", "fight_end"}
    death_box_times: list[float] = []
    wipe_times: list[float] = []

    for moment in store.moments:
        if not moment.answered:
            continue
        if fight_id and moment.fight_id and moment.fight_id != fight_id:
            continue
        if moment.timestamp_sec < fight.start or moment.timestamp_sec > fight.end:
            continue
        if moment.event_type == "death_box":
            death_box_times.append(moment.timestamp_sec)
        elif moment.event_type in wipe_types:
            wipe_times.append(moment.timestamp_sec)

    cue = max(death_box_times) if death_box_times else (
        max(wipe_times) if wipe_times else None
    )
    if cue is None:
        return None

    pad = config.editing.action_cut_post_wipe_pad_sec
    return min(fight.end, cue + pad)


def _trim_segments_at_fight_end(
    segments: list[EditSegment],
    cutoff: float,
    config: AppConfig,
) -> list[EditSegment]:
    min_len = config.editing.action_cut_min_sec * 0.5
    trimmed: list[EditSegment] = []
    for seg in segments:
        if seg.start >= cutoff - 0.05:
            continue
        end = min(seg.end, cutoff)
        if end - seg.start < min_len:
            continue
        trimmed.append(
            EditSegment(start=seg.start, end=end, reason=seg.reason)
        )
    return trimmed if trimmed else segments


def _apply_fight_end_trim(
    segments: list[EditSegment],
    fight: EditSegment,
    config: AppConfig,
    work_dir: Path | None,
) -> list[EditSegment]:
    cutoff = _literacy_fight_end_cutoff(work_dir, fight, config)
    if cutoff is None or cutoff >= fight.end - 0.25:
        return segments
    trimmed = _trim_segments_at_fight_end(segments, cutoff, config)
    if trimmed != segments:
        print(
            f"  [plan] post-wipe trim: ending at {cutoff:.1f}s "
            f"(skip looting after fight)"
        )
    return trimmed


def _remerge_close_segments(
    segments: list[EditSegment],
    merge_gap: float,
) -> list[EditSegment]:
    if not segments:
        return segments
    segments = sorted(segments, key=lambda seg: seg.start)
    merged: list[EditSegment] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg.start <= prev.end + merge_gap:
            merged[-1] = EditSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                reason=prev.reason,
            )
        else:
            merged.append(seg)
    return merged


def _bridge_spine_gaps_to_finale(
    segments: list[EditSegment],
    fight: EditSegment,
    config: AppConfig,
    *,
    work_dir: Path | None,
) -> list[EditSegment]:
    """
    After knock / mid-fight beats, keep teammate cleanup → last enemy → wipe.
    Bridges large gaps with action-aware cuts instead of dropping them entirely.
    """
    if len(segments) < 2 or work_dir is None:
        return segments

    store = load_literacy_store(work_dir)
    fight_id = _fight_id_for_segment(work_dir, fight)
    finale_types = {"team_wipe", "death_box", "fight_end", "kill"}
    has_finale_moment = False
    if store and fight_id:
        has_finale_moment = any(
            moment.answered
            and moment.fight_id == fight_id
            and moment.event_type in finale_types
            and moment.clip_worthy in {"must_include", "nice"}
            for moment in store.moments
        )

    editing = config.editing
    merge_gap = editing.action_cut_merge_gap_sec
    min_gap = merge_gap + 4.0
    result: list[EditSegment] = []
    index = 0

    while index < len(segments):
        current = segments[index]
        if index + 1 >= len(segments):
            result.append(current)
            break

        nxt = segments[index + 1]
        gap = nxt.start - current.end
        near_fight_end = nxt.end >= fight.end - 8.0
        should_bridge = gap >= min_gap and (
            has_finale_moment and near_fight_end
        )

        if not should_bridge:
            result.append(current)
            index += 1
            continue

        # Connect through cleanup / last enemy / wipe — trim only quiet valleys.
        bridge = EditSegment(
            start=max(fight.start, current.end),
            end=min(fight.end, max(nxt.end, fight.end)),
            reason="literacy_bridge",
        )
        result.append(current)
        result.append(bridge)
        index += 2

    return result


def _literacy_anchor_windows(
    work_dir: Path | None,
    fight: EditSegment,
    config: AppConfig,
) -> list[tuple[float, float, str, str]]:
    """Curator literacy moments → time windows the edit must cover."""
    if work_dir is None:
        return []

    store = load_literacy_store(work_dir)
    if not store:
        return []

    fight_id = _fight_id_for_segment(work_dir, fight)
    editing = config.editing
    pad_before = editing.action_cut_pad_before_sec
    pad_after = editing.action_cut_pad_after_sec
    anchors: list[tuple[float, float, str, str]] = []

    for moment in store.moments:
        if not moment.answered or moment.clip_worthy == "skip":
            continue
        if fight_id and moment.fight_id and moment.fight_id != fight_id:
            continue
        if moment.timestamp_sec < fight.start - 2 or moment.timestamp_sec > fight.end + 2:
            continue

        if moment.context_start_sec is not None and moment.context_end_sec is not None:
            start = max(fight.start, moment.context_start_sec)
            end = min(fight.end, moment.context_end_sec)
        else:
            start = max(fight.start, moment.timestamp_sec - pad_before)
            end = min(fight.end, moment.timestamp_sec + pad_after)

        if moment.event_type == "knock_self":
            # Keep knock impact, cut long downed crawl (literacy rule).
            span = min(end - start, 6.0)
            center = moment.timestamp_sec
            start = max(fight.start, center - span * 0.35)
            end = min(fight.end, start + span)

        if end - start < editing.action_cut_min_sec * 0.5:
            continue

        anchors.append((start, end, moment.event_type, moment.clip_worthy))

    return anchors


def _segments_overlap(seg: EditSegment, start: float, end: float) -> bool:
    return seg.end > start and seg.start < end


def _anchor_covered(
    segments: list[EditSegment],
    start: float,
    end: float,
    *,
    min_fraction: float = 0.45,
) -> bool:
    anchor_len = max(end - start, 0.1)
    covered = 0.0
    for seg in segments:
        if not _segments_overlap(seg, start, end):
            continue
        covered += max(
            0.0,
            min(seg.end, end) - max(seg.start, start),
        )
    return covered / anchor_len >= min_fraction


def _merge_literacy_anchors(
    segments: list[EditSegment],
    anchors: list[tuple[float, float, str, str]],
    fight: EditSegment,
    config: AppConfig,
) -> list[EditSegment]:
    """Patch in literacy beats the scorer missed — without re-merging the whole fight."""
    if not anchors:
        return segments

    editing = config.editing
    merge_gap = editing.action_cut_merge_gap_sec
    merged = list(segments)

    for start, end, event_type, worthy in anchors:
        if worthy not in {"must_include", "nice"}:
            continue
        if _anchor_covered(merged, start, end):
            continue
        merged.append(
            EditSegment(start=start, end=end, reason="literacy_anchor")
        )

    merged.sort(key=lambda seg: seg.start)
    combined: list[EditSegment] = []
    for seg in merged:
        if not combined:
            combined.append(seg)
            continue
        prev = combined[-1]
        if seg.start <= prev.end + merge_gap:
            combined[-1] = EditSegment(
                start=prev.start,
                end=max(prev.end, seg.end),
                reason=prev.reason,
            )
        else:
            combined.append(seg)

    finale_types = {"team_wipe", "death_box", "fight_end"}
    for start, end, event_type, worthy in anchors:
        if event_type in finale_types and worthy == "must_include":
            if combined and end > combined[-1].end - 3:
                combined[-1] = EditSegment(
                    start=combined[-1].start,
                    end=max(combined[-1].end, min(fight.end, end + 1.5)),
                    reason=combined[-1].reason,
                )

    return combined


def _split_oversized_segments(
    analysis: ClipAnalysis,
    segments: list[EditSegment],
    fight: EditSegment,
    config: AppConfig,
    *,
    apex: bool,
    max_span: float = 28.0,
) -> list[EditSegment]:
    """Break long stitched spans at internal cold valleys."""
    out: list[EditSegment] = []
    for seg in segments:
        if seg.end - seg.start <= max_span:
            out.append(seg)
            continue
        chunks = _split_long_span_at_cold_gaps(
            analysis,
            seg.start,
            seg.end,
            config,
            apex=apex,
            fight_start=fight.start,
            fight_span=fight.end - fight.start,
        )
        for start, end in chunks:
            if end - start >= config.editing.action_cut_min_sec * 0.75:
                out.append(
                    EditSegment(start=start, end=end, reason=seg.reason)
                )
    out.sort(key=lambda seg: seg.start)
    return out if out else segments


_DETECTED_FIGHT_LITERACY_GAP_SEC = 18.0
_OPENER_MIN_GAP_SEC = 12.0


def _literacy_spine_starts_too_late(
    spine: list[EditSegment],
    fight: EditSegment,
) -> bool:
    """Literacy beats can start mid-window — don't replace a detected fight opener."""
    if fight.reason != "detected_fight":
        return False
    first = min(seg.start for seg in spine)
    return first > fight.start + _DETECTED_FIGHT_LITERACY_GAP_SEC


def _ensure_fight_opener(
    analysis: ClipAnalysis,
    segments: list[EditSegment],
    fight: EditSegment,
    config: AppConfig,
    *,
    apex: bool,
) -> list[EditSegment]:
    """
    When the first kept beat starts well into the labeled window, prepend the
    opening contact arc (detected start → first beat) instead of jumping late.
    """
    if not segments:
        return segments

    fight_start = fight.start
    first_start = min(seg.start for seg in segments)
    gap = first_start - fight_start
    if gap < _OPENER_MIN_GAP_SEC:
        return segments

    span = max(fight.end - fight_start, 0.1)
    if gap / span < 0.1:
        return segments

    merge_gap = config.editing.action_cut_merge_gap_sec
    prefix_end = max(fight_start, first_start - merge_gap)
    if prefix_end - fight_start < config.editing.action_cut_min_sec * 0.75:
        return segments

    prefix_fight = EditSegment(
        start=fight_start,
        end=prefix_end,
        reason=fight.reason or "detected_fight",
    )
    prefix_cuts = _cut_by_removing_cold_gaps(
        analysis, prefix_fight, config, apex=apex
    )
    if not prefix_cuts:
        return segments

    combined = prefix_cuts + list(segments)
    combined.sort(key=lambda seg: seg.start)
    return _remerge_close_segments(combined, merge_gap)


def _is_tail_only(
    segments: list[EditSegment],
    fight_start: float,
    fight_end: float,
) -> bool:
    if not segments:
        return True
    span = max(fight_end - fight_start, 0.1)
    first_rel = (segments[0].start - fight_start) / span
    union_start = min(seg.start for seg in segments)
    return first_rel > 0.55 or union_start > fight_start + span * 0.45


def _build_segments_from_hot_ranges(
    analysis: ClipAnalysis,
    fight: EditSegment,
    config: AppConfig,
    *,
    percentile: float,
    apex: bool,
) -> list[EditSegment]:
    editing = config.editing
    fight_start = fight.start
    fight_end = fight.end
    span = fight_end - fight_start
    step = editing.action_cut_scan_step_sec
    win = editing.action_cut_window_sec
    merge_gap = editing.action_cut_merge_gap_sec
    pad_before = editing.action_cut_pad_before_sec
    pad_after = editing.action_cut_pad_after_sec
    min_cut = editing.action_cut_min_sec

    windows: list[tuple[float, float, float]] = []
    t = fight_start
    while t < fight_end - min_cut * 0.4:
        w_end = min(fight_end, t + win)
        score = _window_action_score(
            analysis,
            t,
            w_end,
            apex=apex,
            fight_start=fight_start,
            fight_span=span,
        )
        windows.append((t, w_end, score))
        t += step

    if not windows:
        return [fight]

    scores = [item[2] for item in windows]
    rel_threshold = _percentile(scores, percentile)
    abs_floor = editing.action_cut_min_score * 0.45
    threshold = max(abs_floor, rel_threshold)

    hot_ranges: list[tuple[float, float]] = []
    for win_start, win_end, score in windows:
        rel = (win_start - fight_start) / max(span, 0.1)
        local_threshold = threshold - (2.0 if rel < 0.38 else 0.0)
        if score >= local_threshold:
            hot_ranges.append((win_start, win_end))

    if not hot_ranges:
        return [fight]

    merged = _merge_ranges(hot_ranges, merge_gap=merge_gap)
    raw_chunks: list[tuple[float, float]] = []
    for start, end in merged:
        raw_chunks.extend(
            _split_long_span_at_cold_gaps(
                analysis,
                start,
                end,
                config,
                apex=apex,
                fight_start=fight_start,
                fight_span=span,
            )
        )

    segments: list[EditSegment] = []
    for start, end in raw_chunks:
        padded_start = max(fight_start, start - pad_before)
        padded_end = min(fight_end, end + pad_after)
        if padded_end - padded_start < min_cut:
            continue
        segments.append(
            EditSegment(
                start=padded_start,
                end=padded_end,
                reason="action_cut",
            )
        )

    segments.sort(key=lambda seg: seg.start)
    return segments


def _cut_by_removing_cold_gaps(
    analysis: ClipAnalysis,
    fight: EditSegment,
    config: AppConfig,
    *,
    apex: bool,
) -> list[EditSegment]:
    """
    Inverse strategy: walk the full fight, remove only sustained cold gaps,
    keep everything else — preserves start-to-finish story.
    """
    editing = config.editing
    fight_start = fight.start
    fight_end = fight.end
    span = fight_end - fight_start
    pad_before = editing.action_cut_pad_before_sec
    pad_after = editing.action_cut_pad_after_sec
    min_cut = editing.action_cut_min_sec
    cold_gap = editing.action_cut_cold_gap_sec

    windows = _score_windows_in_range(
        analysis,
        fight_start,
        fight_end,
        config,
        apex=apex,
        fight_start=fight_start,
        fight_span=span,
    )
    if not windows:
        return [fight]

    scores = [item[2] for item in windows]
    cold_threshold = _percentile(scores, 28)

    segments: list[EditSegment] = []
    chunk_start = fight_start
    cold_start: float | None = None

    for win_start, win_end, score in windows:
        if score < cold_threshold:
            if cold_start is None:
                cold_start = win_start
        else:
            if cold_start is not None:
                gap_len = win_start - cold_start
                if gap_len >= cold_gap:
                    end = min(fight_end, cold_start + pad_after)
                    start = max(fight_start, chunk_start - pad_before)
                    if end - start >= min_cut:
                        segments.append(
                            EditSegment(start=start, end=end, reason="action_cut")
                        )
                    chunk_start = win_start
                cold_start = None

    final_start = max(fight_start, chunk_start - pad_before)
    final_end = fight_end
    if final_end - final_start >= min_cut:
        segments.append(
            EditSegment(
                start=final_start,
                end=final_end,
                reason="action_cut",
            )
        )

    segments.sort(key=lambda seg: seg.start)
    return segments if segments else [fight]


def cut_fight_to_action_segments(
    analysis: ClipAnalysis,
    fight: EditSegment,
    config: AppConfig,
    *,
    work_dir: Path | None = None,
) -> list[EditSegment]:
    """
    Keep beats across the full labeled fight (start → end), remove low-action
    gaps in the middle, pad each beat, stitch chronologically.
    """
    editing = config.editing
    span = fight.end - fight.start
    if span < editing.action_cut_min_fight_span_sec:
        return [fight]

    apex = editing.game_profile.lower() == "apex"
    percentile = editing.action_cut_percentile

    spine = _literacy_spine_segments(work_dir, fight, config)
    if spine and _literacy_spine_starts_too_late(spine, fight):
        print(
            f"  [plan] literacy spine skipped: first beat at "
            f"{min(seg.start for seg in spine):.0f}s is "
            f"{min(seg.start for seg in spine) - fight.start:.0f}s after "
            f"fight start {fight.start:.0f}s"
        )
        spine = None

    if spine:
        apex_flag = apex
        bridged: list[EditSegment] = []
        end_cutoff = _literacy_fight_end_cutoff(work_dir, fight, config) or fight.end
        for seg in spine:
            if seg.reason == "literacy_bridge":
                # Keep teammate cleanup → last enemy → wipe as one arc.
                bridged.append(
                    EditSegment(
                        start=seg.start,
                        end=end_cutoff,
                        reason="literacy_bridge",
                    )
                )
            else:
                bridged.append(seg)
        if not bridged:
            bridged = spine
        bridged = _apply_fight_end_trim(bridged, fight, config, work_dir)
        bridged = _ensure_fight_opener(
            analysis, bridged, fight, config, apex=apex
        )
        print(
            f"  [plan] literacy spine: {len(bridged)} beat(s), "
            f"{_coverage_span(bridged):.0f}s from fight "
            f"{fight.start:.0f}–{fight.end:.0f}s"
        )
        return bridged

    segments = _build_segments_from_hot_ranges(
        analysis,
        fight,
        config,
        percentile=percentile,
        apex=apex,
    )

    if not segments or segments == [fight]:
        return segments if segments else [fight]

    if _is_tail_only(segments, fight.start, fight.end):
        segments = _cut_by_removing_cold_gaps(
            analysis, fight, config, apex=apex
        )

    if not segments:
        return [fight]

    # Never drop early beats for score — merge if over limit instead of tail bias.
    max_segments = editing.action_cut_max_segments
    while max_segments > 0 and len(segments) > max_segments:
        best_merge_idx = 0
        best_gap = 1e9
        for index in range(len(segments) - 1):
            gap = segments[index + 1].start - segments[index].end
            if gap < best_gap:
                best_gap = gap
                best_merge_idx = index
        left = segments[best_merge_idx]
        right = segments[best_merge_idx + 1]
        merged = EditSegment(
            start=left.start,
            end=right.end,
            reason="action_cut",
        )
        segments = (
            segments[:best_merge_idx]
            + [merged]
            + segments[best_merge_idx + 2 :]
        )

    coverage = _coverage_span(segments) / max(span, 0.1)
    if coverage < 0.12 and len(segments) <= 2:
        segments = [fight]

    segments = _apply_fight_end_trim(segments, fight, config, work_dir)

    anchors = _literacy_anchor_windows(work_dir, fight, config)
    if anchors:
        before = len(segments)
        segments = _merge_literacy_anchors(segments, anchors, fight, config)
        print(
            f"  [plan] literacy anchors: {len(anchors)} beat(s), "
            f"{before}->{len(segments)} segment(s) for fight "
            f"{fight.start:.0f}–{fight.end:.0f}s"
        )

    segments = _split_oversized_segments(
        analysis, segments, fight, config, apex=apex
    )

    segments = _ensure_fight_opener(
        analysis, segments, fight, config, apex=apex
    )

    return segments


def apply_action_cuts_to_plan(
    plan: EditPlan,
    analysis: ClipAnalysis,
    config: AppConfig,
    *,
    work_dir: Path | None = None,
) -> EditPlan:
    """Replace full-fight segments with action-only micro-cuts."""
    if not config.editing.action_cut_enabled or not plan.segments:
        return plan

    cut_reasons = {
        "gunfight",
        "labeled_fight",
        "detected_fight",
        "action_cut",
    }
    new_segments: list[EditSegment] = []
    for seg in plan.segments:
        if seg.reason in cut_reasons:
            new_segments.extend(
                cut_fight_to_action_segments(
                    analysis, seg, config, work_dir=work_dir
                )
            )
        else:
            new_segments.append(seg)

    if not new_segments:
        return plan

    new_segments.sort(key=lambda seg: seg.start)
    return EditPlan(
        target_duration_sec=sum(s.end - s.start for s in new_segments),
        segments=new_segments,
        effects=plan.effects,
        hook_text=plan.hook_text,
        hashtags=plan.hashtags,
        caption_style=plan.caption_style,
    )
