"""Live progress for Elden Ring scan / retune jobs."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}

# Wall-clock-weighted scan phases (monotonic global %). Tuned for long VODs
# where extract + ML dominate; probes are secondary.
SCAN_PHASE_RANGES: dict[str, tuple[float, float]] = {
    "starting": (0.0, 1.0),
    "sampling": (1.0, 40.0),
    "gap_fill": (40.0, 45.0),
    "scoring": (45.0, 55.0),
    "terminal_probes": (55.0, 65.0),
    "start_probes": (65.0, 75.0),
    "end_probes": (75.0, 82.0),
    "ml_blend": (82.0, 96.0),
    "infer": (82.0, 96.0),  # alias used by ML callback
    "clustering": (96.0, 99.5),
    "reanalyze": (10.0, 55.0),
    "done": (100.0, 100.0),
    "failed": (0.0, 100.0),
}

_ETA_EMA = 0.35


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def scan_phase_pct(phase: str, frac: float) -> float:
    """Map 0..1 progress within a named phase → global scan %."""
    lo, hi = SCAN_PHASE_RANGES.get(phase, (0.0, 100.0))
    frac = max(0.0, min(1.0, float(frac)))
    return round(lo + (hi - lo) * frac, 2)


def phase_end_pct(phase: str) -> float:
    return SCAN_PHASE_RANGES.get(phase, (0.0, 100.0))[1]


def start_job(job_id: str, *, kind: str = "scan", message: str = "Starting…") -> dict:
    payload = {
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "phase": "starting",
        "message": message,
        "current": 0,
        "total": 0,
        "pct": 0.0,
        "eta_sec": None,
        "eta_phase_sec": None,
        "elapsed_sec": 0.0,
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "_t0": time.monotonic(),
        "_phase_t0": time.monotonic(),
        "_phase_name": "starting",
        "_phase_work0": 0.0,
        "_eta_smooth": None,
        "error": "",
    }
    with _lock:
        _jobs[job_id] = payload
    return public_progress(job_id)


def _resolve_pct(job: dict[str, Any], fields: dict[str, Any]) -> float:
    """Compute candidate pct from phase_frac / current/total / raw pct."""
    phase = str(fields.get("phase") or job.get("phase") or "")
    frac = fields.get("phase_frac")
    if frac is None:
        cur = fields.get("current", job.get("current"))
        total = fields.get("total", job.get("total"))
        try:
            cur_f = float(cur) if cur is not None else 0.0
            tot_f = float(total) if total is not None else 0.0
        except (TypeError, ValueError):
            cur_f, tot_f = 0.0, 0.0
        if tot_f > 0:
            frac = cur_f / tot_f
    if frac is not None and phase in SCAN_PHASE_RANGES:
        return scan_phase_pct(phase, float(frac))
    if "pct" in fields and fields["pct"] is not None:
        return float(fields["pct"])
    return float(job.get("pct") or 0.0)


def _compute_eta(job: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return (total_eta_sec, phase_eta_sec) using phase rate + later-phase budget."""
    elapsed = float(job.get("elapsed_sec") or 0.0)
    pct = float(job.get("pct") or 0.0)
    if elapsed < 1.5 or pct < 0.5:
        return None, None

    phase = str(job.get("phase") or "")
    cur = float(job.get("current") or 0.0)
    total = float(job.get("total") or 0.0)
    phase_elapsed = max(0.25, time.monotonic() - float(job.get("_phase_t0") or job["_t0"]))

    eta_phase: float | None = None
    lo, hi = SCAN_PHASE_RANGES.get(phase, (0.0, 100.0))
    span = max(0.5, hi - lo)
    within = max(0.0, min(span, pct - lo))
    frac = within / span

    # Prefer in-phase % rate — stable across extract→score substeps that reset
    # current/total mid-phase.
    if frac > 0.02:
        rate_frac = frac / phase_elapsed
        if rate_frac > 1e-6:
            eta_phase = (1.0 - frac) / rate_frac

    # Refine with item throughput when the counter is advancing meaningfully.
    if total > cur >= 1 and (cur / total) >= 0.05:
        rate = cur / phase_elapsed
        if rate > 1e-6:
            item_eta = (total - cur) / rate
            if eta_phase is None:
                eta_phase = item_eta
            else:
                # Blend — items help when ML/probe counters are honest.
                eta_phase = 0.55 * eta_phase + 0.45 * item_eta

    later_weight = max(0.0, 100.0 - hi)
    sec_per_pct = elapsed / max(pct, 1.0)
    # Conservative pad on unknown later phases (ML often slower than extract).
    later_pad = 1.25 if later_weight >= 10.0 else 1.05
    eta_later = sec_per_pct * later_weight * later_pad

    if eta_phase is None:
        eta_total = sec_per_pct * max(0.0, 100.0 - pct)
    else:
        eta_total = float(eta_phase) + eta_later

    return max(0.0, eta_total), (max(0.0, eta_phase) if eta_phase is not None else None)


def update_job(job_id: str, **fields: Any) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return {}

        prev_phase = str(job.get("_phase_name") or job.get("phase") or "")
        new_phase = str(fields.get("phase") or prev_phase)

        # Strip helper-only keys before merge; keep phase_frac for pct resolve.
        phase_frac = fields.pop("phase_frac", None)
        raw_fields = {k: v for k, v in fields.items() if not str(k).startswith("_")}
        if phase_frac is not None:
            raw_fields["phase_frac"] = phase_frac

        candidate_pct = _resolve_pct(job, raw_fields)
        # Never move the bar backwards (phase overlap / ML remap used to jump).
        prev_pct = float(job.get("pct") or 0.0)
        pct = max(prev_pct, candidate_pct)

        raw_fields.pop("phase_frac", None)
        job.update(raw_fields)
        job["pct"] = round(pct, 2)
        job["phase"] = new_phase

        t0 = float(job.get("_t0") or time.monotonic())
        job["elapsed_sec"] = round(time.monotonic() - t0, 1)

        if new_phase != prev_phase:
            job["_phase_t0"] = time.monotonic()
            job["_phase_name"] = new_phase
            job["_phase_work0"] = float(job.get("current") or 0.0)
        else:
            # Sub-step within same phase reset the item counter (extract→score).
            prev_cur = float(job.get("_last_current") or 0.0)
            new_cur = float(job.get("current") or 0.0)
            if prev_cur >= 8 and new_cur < prev_cur * 0.35:
                job["_phase_work0"] = 0.0
                # Keep phase clock — overall phase rate still valid via pct.
        job["_last_current"] = float(job.get("current") or 0.0)

        eta_total, eta_phase = _compute_eta(job)
        # Prefer caller-provided ETA (e.g. FFmpeg out_time rate) when present.
        if "eta_sec" in raw_fields and raw_fields["eta_sec"] is not None:
            try:
                job["eta_sec"] = round(float(raw_fields["eta_sec"]), 0)
                job["_eta_smooth"] = float(raw_fields["eta_sec"])
            except (TypeError, ValueError):
                pass
        elif eta_total is not None:
            prev = job.get("_eta_smooth")
            if prev is None:
                smooth = eta_total
            else:
                smooth = _ETA_EMA * eta_total + (1.0 - _ETA_EMA) * float(prev)
            job["_eta_smooth"] = smooth
            job["eta_sec"] = round(smooth, 0)
        if "eta_phase_sec" in raw_fields and raw_fields["eta_phase_sec"] is not None:
            try:
                job["eta_phase_sec"] = round(float(raw_fields["eta_phase_sec"]), 0)
            except (TypeError, ValueError):
                job["eta_phase_sec"] = (
                    round(eta_phase, 0) if eta_phase is not None else None
                )
        else:
            job["eta_phase_sec"] = (
                round(eta_phase, 0) if eta_phase is not None else None
            )

        job["updated_at"] = _now_iso()
        return {k: v for k, v in job.items() if not str(k).startswith("_")}


def finish_job(job_id: str, *, status: str = "done", message: str = "Done", error: str = "") -> dict:
    with _lock:
        existing = _jobs.get(job_id) or {}
        pct = 100.0 if status == "done" else float(existing.get("pct") or 0)
        current = existing.get("total") or existing.get("current") or 0
    return update_job(
        job_id,
        status=status,
        phase="done" if status == "done" else "failed",
        message=message,
        pct=pct,
        eta_sec=0,
        eta_phase_sec=0,
        error=error,
        current=current,
    )


def public_progress(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return {
                "job_id": job_id,
                "status": "idle",
                "phase": "idle",
                "message": "",
                "pct": 0.0,
                "eta_sec": None,
                "eta_phase_sec": None,
                "elapsed_sec": 0.0,
            }
        return {k: v for k, v in job.items() if not str(k).startswith("_")}


def clear_job(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
