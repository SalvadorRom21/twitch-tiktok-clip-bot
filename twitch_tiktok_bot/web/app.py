"""FastAPI web app for previewing and approving TikTok shorts."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from twitch_tiktok_bot.config import AppConfig, load_config
from twitch_tiktok_bot.status import ClipStatus, list_jobs, load_job
from twitch_tiktok_bot.web.jobs import set_status, start_job, update_caption

STATIC_DIR = Path(__file__).parent / "static"


class ProcessRequest(BaseModel):
    clip_url: str
    title: str = ""
    clip_id: str = ""


class CaptionUpdate(BaseModel):
    caption: str


class StatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(approved|rejected|ready)$")


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
        return FileResponse(video_path, media_type="video/mp4")

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

    return app


def run_server(config: AppConfig | None = None) -> None:
    import uvicorn

    cfg = config or load_config()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.web.host, port=cfg.web.port, log_level="info")


if __name__ == "__main__":
    run_server()
