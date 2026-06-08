"""FastAPI web app for previewing and approving TikTok shorts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twitch_tiktok_bot.config import AppConfig, apply_game_profile, load_config
from twitch_tiktok_bot.labels.fights import (
    FightLabelStore,
    find_cached_vods,
    load_fight_labels,
    new_fight_label,
    resolve_vod_video,
    save_fight_labels,
)
from twitch_tiktok_bot.labels.literacy import (
    CLIP_WORTHY_OPTIONS,
    EVENT_TYPES,
    LiteracyMoment,
    ensure_literacy_store,
    load_literacy_store,
    save_literacy_store,
    sync_literacy_to_reference,
)
from twitch_tiktok_bot.plan.game_profiles import GAME_PROFILE_OPTIONS
from twitch_tiktok_bot.status import ClipStatus, list_jobs, load_job
from twitch_tiktok_bot.web.jobs import set_status, start_job, update_caption

STATIC_DIR = Path(__file__).parent / "static"


class ProcessRequest(BaseModel):
    clip_url: str
    title: str = ""
    clip_id: str = ""
    game_profile: str = ""


class CaptionUpdate(BaseModel):
    caption: str


class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected|ready)$")


class FightLabelCreate(BaseModel):
    start_sec: float
    end_sec: float
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    quality: str = "good"
    use_for_clips: bool = True
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    notes: str = ""


class FightLabelUpdate(BaseModel):
    start_sec: float | None = None
    end_sec: float | None = None
    description: str | None = None
    tags: list[str] | None = None
    quality: str | None = None
    use_for_clips: bool | None = None
    clip_start_sec: float | None = None
    clip_end_sec: float | None = None
    notes: str | None = None


class GuideUpdate(BaseModel):
    guide: str = ""


class LiteracyMomentCreate(BaseModel):
    timestamp_sec: float
    title: str = "Custom moment"
    bot_question: str = ""
    category: str = "unknown"


class LiteracyMomentUpdate(BaseModel):
    event_type: str | None = None
    what_happening: str | None = None
    visual_cues: str | None = None
    audio_cues: str | None = None
    clip_worthy: str | None = None
    teaches_bot: str | None = None
    answered: bool | None = None


class GeneralLiteracyUpdate(BaseModel):
    answer: str = ""


def create_app(config: AppConfig | None = None) -> FastAPI:
    app = FastAPI(title="Twitch TikTok Clip Bot", version="0.2.0")
    app.state.config = config or load_config()

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="UI not found")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/label", response_class=HTMLResponse)
    def label_page() -> HTMLResponse:
        label_path = STATIC_DIR / "label.html"
        if not label_path.exists():
            raise HTTPException(status_code=500, detail="Label UI not found")
        return HTMLResponse(label_path.read_text(encoding="utf-8"))

    @app.get("/literacy", response_class=HTMLResponse)
    def literacy_page() -> HTMLResponse:
        literacy_path = STATIC_DIR / "literacy.html"
        if not literacy_path.exists():
            raise HTTPException(status_code=500, detail="Literacy UI not found")
        return HTMLResponse(literacy_path.read_text(encoding="utf-8"))

    @app.get("/api/game-profiles")
    def get_game_profiles() -> dict:
        cfg: AppConfig = app.state.config
        return {
            "options": GAME_PROFILE_OPTIONS,
            "default": cfg.editing.game_profile,
        }

    @app.get("/api/clips")
    def get_clips() -> list[dict]:
        cfg: AppConfig = app.state.config
        return [job.to_dict() for job in list_jobs(cfg)]

    @app.get("/api/clips/{clip_id}")
    def get_clip(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        job = load_job(cfg, clip_id)
        if not job:
            raise HTTPException(status_code=404, detail="Clip not found")
        return job.to_dict()

    @app.post("/api/process")
    def process_clip(body: ProcessRequest) -> dict:
        cfg: AppConfig = app.state.config
        if not body.clip_url.strip():
            raise HTTPException(status_code=400, detail="clip_url is required")
        job = start_job(
            cfg,
            clip_url=body.clip_url.strip(),
            title=body.title.strip(),
            clip_id=body.clip_id.strip() or None,
            game_profile=body.game_profile.strip() or None,
        )
        return job.to_dict()

    @app.get("/api/clips/{clip_id}/video")
    def get_video(clip_id: str) -> FileResponse:
        cfg: AppConfig = app.state.config
        job = load_job(cfg, clip_id)
        if not job or not job.output_video:
            raise HTTPException(status_code=404, detail="Video not found")
        video_path = Path(job.output_video)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file missing")
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/clips/{clip_id}/caption")
    def get_caption(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        job = load_job(cfg, clip_id)
        if not job:
            raise HTTPException(status_code=404, detail="Clip not found")
        caption_path = Path(job.caption_file) if job.caption_file else None
        text = ""
        if caption_path and caption_path.exists():
            text = caption_path.read_text(encoding="utf-8")
        return {"caption": text, "hook_text": job.hook_text, "hashtags": job.hashtags}

    @app.put("/api/clips/{clip_id}/caption")
    def put_caption(clip_id: str, body: CaptionUpdate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = update_caption(cfg, clip_id, body.caption)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/clips/{clip_id}/approve")
    def approve_clip(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = set_status(cfg, clip_id, ClipStatus.APPROVED)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/clips/{clip_id}/reject")
    def reject_clip(clip_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = set_status(cfg, clip_id, ClipStatus.REJECTED)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/clips/{clip_id}/status")
    def update_status(clip_id: str, body: StatusUpdate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            job = set_status(cfg, clip_id, ClipStatus(body.status))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return job.to_dict()

    @app.get("/api/label/vods")
    def list_label_vods() -> list[dict]:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        return find_cached_vods(data_dir)

    @app.get("/api/label/{vod_id}")
    def get_label_store(vod_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        return {
            **store.to_dict(),
            "video_url": f"/api/label/{vod_id}/video",
            "video_name": video_path.name,
        }

    @app.get("/api/label/{vod_id}/video")
    def get_label_video(vod_id: str) -> FileResponse:
        cfg: AppConfig = app.state.config
        try:
            _work_dir, video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"},
        )

    @app.put("/api/label/{vod_id}/guide")
    def put_label_guide(vod_id: str, body: GuideUpdate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        store.guide = body.guide.strip()
        save_fight_labels(work_dir, store)
        return store.to_dict()

    @app.post("/api/label/{vod_id}/fights")
    def add_fight_label(vod_id: str, body: FightLabelCreate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        fight = new_fight_label(
            start_sec=body.start_sec,
            end_sec=body.end_sec,
            description=body.description,
            tags=body.tags,
            quality=body.quality,
            use_for_clips=body.use_for_clips,
            notes=body.notes,
        )
        if body.clip_start_sec is not None:
            fight.clip_start_sec = round(body.clip_start_sec, 2)
        if body.clip_end_sec is not None:
            fight.clip_end_sec = round(body.clip_end_sec, 2)
        store.fights.append(fight)
        store.fights.sort(key=lambda item: item.start_sec)
        save_fight_labels(work_dir, store)
        return fight.to_dict()

    @app.put("/api/label/{vod_id}/fights/{fight_id}")
    def update_fight_label(
        vod_id: str, fight_id: str, body: FightLabelUpdate
    ) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        fight = next((item for item in store.fights if item.id == fight_id), None)
        if fight is None:
            raise HTTPException(status_code=404, detail="Fight label not found")
        updates = body.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(fight, key, value)
        if fight.end_sec < fight.start_sec:
            fight.start_sec, fight.end_sec = fight.end_sec, fight.start_sec
        store.fights.sort(key=lambda item: item.start_sec)
        save_fight_labels(work_dir, store)
        return fight.to_dict()

    @app.delete("/api/label/{vod_id}/fights/{fight_id}")
    def delete_fight_label(vod_id: str, fight_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_fight_labels(work_dir) or FightLabelStore(vod_id=vod_id)
        before = len(store.fights)
        store.fights = [item for item in store.fights if item.id != fight_id]
        if len(store.fights) == before:
            raise HTTPException(status_code=404, detail="Fight label not found")
        save_fight_labels(work_dir, store)
        return store.to_dict()

    @app.get("/api/literacy")
    def literacy_api_index() -> dict:
        return {
            "ui": "/literacy",
            "vods": "/api/literacy/vods",
            "store": "/api/literacy/{vod_id}",
            "hint": "Start server: python main.py --literacy 2788855626",
        }

    @app.get("/api/literacy/vods")
    def list_literacy_vods() -> list[dict]:
        cfg: AppConfig = app.state.config
        data_dir = cfg.resolve_path(cfg.paths.data_dir)
        vods = find_cached_vods(data_dir)
        for vod in vods:
            work_dir = data_dir / vod["vod_id"]
            literacy = load_literacy_store(work_dir)
            vod["moment_count"] = len(literacy.moments) if literacy else 0
        return vods

    @app.get("/api/literacy/{vod_id}")
    def get_literacy_store(vod_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        save_literacy_store(work_dir, store)
        answered = sum(1 for moment in store.moments if moment.answered)
        return {
            **store.to_dict(),
            "video_url": f"/api/label/{vod_id}/video",
            "event_types": EVENT_TYPES,
            "clip_worthy_options": CLIP_WORTHY_OPTIONS,
            "progress": {
                "answered": answered,
                "total": len(store.moments),
            },
        }

    @app.post("/api/literacy/{vod_id}/moments")
    def add_literacy_moment(vod_id: str, body: LiteracyMomentCreate) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        moment = LiteracyMoment(
            id=uuid.uuid4().hex[:10],
            timestamp_sec=round(body.timestamp_sec, 2),
            title=body.title.strip() or "Custom moment",
            bot_question=body.bot_question.strip()
            or "What is happening here?",
            category=body.category,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        store.moments.append(moment)
        store.moments.sort(key=lambda item: item.timestamp_sec)
        save_literacy_store(work_dir, store)
        return moment.to_dict()

    @app.put("/api/literacy/{vod_id}/moments/{moment_id}")
    def update_literacy_moment(
        vod_id: str, moment_id: str, body: LiteracyMomentUpdate
    ) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        moment = next(
            (item for item in store.moments if item.id == moment_id), None
        )
        if moment is None:
            raise HTTPException(status_code=404, detail="Moment not found")
        for key, value in body.model_dump(exclude_unset=True).items():
            setattr(moment, key, value)
        save_literacy_store(work_dir, store)
        return moment.to_dict()

    @app.put("/api/literacy/{vod_id}/general/{question_id}")
    def update_general_literacy(
        vod_id: str, question_id: str, body: GeneralLiteracyUpdate
    ) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = ensure_literacy_store(work_dir, vod_id, cfg.project_root)
        item = next((g for g in store.general if g.id == question_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="Question not found")
        item.answer = body.answer.strip()
        save_literacy_store(work_dir, store)
        return item.to_dict()

    @app.post("/api/literacy/{vod_id}/sync")
    def sync_literacy(vod_id: str) -> dict:
        cfg: AppConfig = app.state.config
        try:
            work_dir, _video_path = resolve_vod_video(cfg, vod_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        store = load_literacy_store(work_dir)
        if store is None:
            raise HTTPException(status_code=404, detail="No literacy labels yet")
        out_path = sync_literacy_to_reference(
            store,
            cfg.resolve_path("data/reference/apex_game_literacy.yaml"),
            cfg.project_root,
        )
        answered = sum(1 for moment in store.moments if moment.answered)
        return {
            "synced_to": str(out_path),
            "moments_answered": answered,
            "moments_total": len(store.moments),
        }

    return app


def run_server(config: AppConfig | None = None) -> None:
    import uvicorn

    cfg = config or load_config()
    app = create_app(cfg)
    base = f"http://{cfg.web.host}:{cfg.web.port}"
    print(f"  Preview UI     {base}/")
    print(f"  Fight labeler  {base}/label")
    print(f"  Apex trainer   {base}/literacy")
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="info")


if __name__ == "__main__":
    run_server()
