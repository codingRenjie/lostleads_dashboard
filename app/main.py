import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import PROJECT_ROOT, settings
from app.day_views import build_day_groups, load_day_view
from app.db import Batch, BatchInsight, FileError, Loss, init_db
from app.db import SessionLocal as DbSessionLocal
from app.pipeline import process_batch_files, safe_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIO_EXT = {
    ".mp3",
    ".wav",
    ".m4a",
    ".webm",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".oga",
    ".ogg",
    ".flac",
}

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="老船长", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")


def _require_keys() -> None:
    if not settings.llm_api_key.strip():
        raise HTTPException(
            status_code=503,
            detail="未配置 LLM_API_KEY（Kimi / Moonshot），请在项目根目录复制 .env.example 为 .env 并填写。",
        )
    if settings.iflytek_configured():
        return
    if not settings.effective_asr_key().strip():
        raise HTTPException(
            status_code=503,
            detail="未配置语音识别：请在 .env 填写科大讯飞 IFLYTEK_*，或配置 ASR_API_KEY 使用 Whisper 兼容接口。",
        )


@app.get("/logo.png")
async def logo_png():
    path = PROJECT_ROOT / "logo.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="logo.png not found")
    return FileResponse(path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    day: str | None = None,
    section: str = "analysis",
):
    """
    day: YYYY-MM-DD（上海时区，按批次 created_at）；缺省为最新有数据的一天。
    section: analysis | strategy | data | history
    """
    if section not in {"analysis", "strategy", "data", "history"}:
        section = "analysis"

    db = DbSessionLocal()
    try:
        groups = build_day_groups(db)
        latest_day = groups["latest_day"]
        day_summaries = groups["day_summaries"]
        by_day = groups["by_day"]

        if not latest_day:
            return templates.TemplateResponse(
                request,
                "dashboard.html",
                {
                    "section": section,
                    "latest_day": None,
                    "view_day": None,
                    "is_today": True,
                    "day_label": "",
                    "history_summaries": [],
                    "loss_rows": [],
                    "insight": None,
                    "consultant_counts_json": "{}",
                    "theme_counts_json": "{}",
                    "vehicle_model_distinct": 0,
                    "consultant_count": 0,
                    "batch_count": 0,
                    "batch_ids": [],
                    "status": "empty",
                    "export_day": None,
                },
            )

        # 历史列表中选中某日后进入三栏；点「当日*」则回到最新日
        if section == "history":
            view_day = latest_day
            is_today = True
        elif day and day in by_day:
            view_day = day
            is_today = day == latest_day
        else:
            view_day = latest_day
            is_today = True

        view = load_day_view(db, view_day, by_day)
        history_summaries = [s for s in day_summaries if not s["is_latest"]]

        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "section": section,
                "latest_day": latest_day,
                "view_day": view_day,
                "is_today": is_today,
                "day_label": view["day_label"],
                "history_summaries": history_summaries,
                "loss_rows": view["loss_rows"],
                "insight": view["insight"],
                "consultant_counts_json": json.dumps(
                    view["consultant_counts"], ensure_ascii=False
                ),
                "theme_counts_json": json.dumps(view["theme_counts"], ensure_ascii=False),
                "vehicle_model_distinct": view["vehicle_model_distinct"],
                "consultant_count": len(view["consultant_counts"]),
                "batch_count": view["batch_count"],
                "batch_ids": view["batch_ids"],
                "status": view["status"],
                "export_day": view_day,
            },
        )
    finally:
        db.close()


@app.get("/api/batches")
async def list_batches():
    db = DbSessionLocal()
    try:
        rows = db.execute(select(Batch).order_by(Batch.id.desc()).limit(100)).scalars().all()
        out = []
        for b in rows:
            n = db.execute(select(Loss).where(Loss.batch_id == b.id)).scalars().all()
            out.append(
                {
                    "id": b.id,
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                    "status": b.status,
                    "loss_count": len(n),
                    "error_message": b.error_message,
                }
            )
        return out
    finally:
        db.close()


@app.get("/api/batches/{batch_id}")
async def get_batch(batch_id: int):
    db = DbSessionLocal()
    try:
        batch = db.execute(
            select(Batch)
            .where(Batch.id == batch_id)
            .options(
                selectinload(Batch.losses),
                selectinload(Batch.file_errors),
            )
        ).scalar_one_or_none()
        if not batch:
            raise HTTPException(404, "batch not found")
        losses = [
            {
                "loss_id": x.loss_id,
                "date": x.date,
                "consultant": x.consultant,
                "vehicle_model": x.vehicle_model,
                "customer_tags": json.loads(x.customer_tags or "[]"),
                "loss_description": x.loss_description,
                "source_filename": x.source_filename,
            }
            for x in sorted(batch.losses, key=lambda r: r.loss_id)
        ]
        errors = [{"filename": e.filename, "message": e.message} for e in batch.file_errors]
        insight = None
        bi = db.get(BatchInsight, batch_id)
        if bi:
            try:
                insight = json.loads(bi.payload_json)
            except json.JSONDecodeError:
                insight = None
        return {
            "id": batch.id,
            "created_at": batch.created_at.isoformat() if batch.created_at else None,
            "status": batch.status,
            "error_message": batch.error_message,
            "losses": losses,
            "file_errors": errors,
            "insight": insight,
            "asr_provider": "iflytek" if settings.iflytek_configured() else "whisper",
        }
    finally:
        db.close()


@app.post("/api/process")
async def process_upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(default_factory=list),
    meta: str | None = Form(None),
):
    _require_keys()
    if not files:
        raise HTTPException(400, "请选择包含音频文件的文件夹")

    meta_map: dict = {}
    if meta:
        try:
            meta_map = json.loads(meta)
        except json.JSONDecodeError:
            meta_map = {}

    filtered: list[UploadFile] = []
    for f in files:
        name = f.filename or ""
        ext = Path(name.lower()).suffix
        if ext in AUDIO_EXT:
            filtered.append(f)
    if not filtered:
        raise HTTPException(
            400,
            f"未找到支持的音频格式，允许的后缀：{', '.join(sorted(AUDIO_EXT))}",
        )

    db = DbSessionLocal()
    batch = Batch(status="pending")
    db.add(batch)
    db.commit()
    db.refresh(batch)
    batch_id = batch.id
    db.close()

    upload_dir = PROJECT_ROOT / "data" / "uploads" / str(batch_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict] = []
    for uf in filtered:
        raw_name = uf.filename or "unknown"
        dest_name = safe_filename(raw_name)
        dest = upload_dir / dest_name
        content = await uf.read()
        dest.write_bytes(content)
        lm = meta_map.get(raw_name)
        if lm is None:
            for k, v in meta_map.items():
                if k.endswith(dest_name) or dest_name.endswith(Path(k).name):
                    lm = v
                    break
        lm_int = int(lm) if isinstance(lm, (int, float)) else None
        tasks.append(
            {
                "path": dest,
                "logical_name": raw_name,
                "last_modified_ms": lm_int,
            }
        )

    background_tasks.add_task(process_batch_files, batch_id, tasks)

    return {
        "batch_id": batch_id,
        "status": "pending",
        "file_count": len(tasks),
        "asr_provider": "iflytek" if settings.iflytek_configured() else "whisper",
    }


def _losses_to_csv_response(rows: list[Loss], filename: str) -> StreamingResponse:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["战败ID", "日期", "顾问名称", "车型", "客户关键词", "战败描述"])
    for r in rows:
        tags = json.loads(r.customer_tags or "[]")
        tag_str = ",".join(tags) if isinstance(tags, list) else str(tags)
        w.writerow(
            [
                r.loss_id,
                r.date,
                r.consultant,
                r.vehicle_model,
                tag_str,
                r.loss_description,
            ]
        )
    data = "\ufeff" + buf.getvalue()
    return StreamingResponse(
        iter([data.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/batches/{batch_id}/export.csv")
async def export_csv(batch_id: int):
    db = DbSessionLocal()
    try:
        batch = db.get(Batch, batch_id)
        if not batch:
            raise HTTPException(404, "batch not found")
        rows = (
            db.execute(select(Loss).where(Loss.batch_id == batch_id).order_by(Loss.loss_id))
            .scalars()
            .all()
        )
        return _losses_to_csv_response(rows, f"losses_batch_{batch_id}.csv")
    finally:
        db.close()


@app.get("/api/days/{day}/export.csv")
async def export_day_csv(day: str):
    """按上海时区日历日合并导出该日全部批次的战败行。"""
    db = DbSessionLocal()
    try:
        groups = build_day_groups(db)
        if day not in groups["by_day"]:
            raise HTTPException(404, "该日无批次数据")
        view = load_day_view(db, day, groups["by_day"])
        batch_ids = view["batch_ids"]
        rows = (
            db.execute(select(Loss).where(Loss.batch_id.in_(batch_ids)).order_by(Loss.loss_id))
            .scalars()
            .all()
            if batch_ids
            else []
        )
        return _losses_to_csv_response(rows, f"losses_day_{day}.csv")
    finally:
        db.close()
