import csv
import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.config import PROJECT_ROOT, settings
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
async def dashboard(request: Request, batch_id: int | None = None):
    db = DbSessionLocal()
    try:
        batches = db.execute(select(Batch).order_by(Batch.id.desc()).limit(50)).scalars().all()
        if not batches:
            return templates.TemplateResponse(
                request,
                "dashboard.html",
                {
                    "batches": [],
                    "batch": None,
                    "loss_rows": [],
                    "insight": None,
                    "consultant_counts": {},
                    "consultant_counts_json": "{}",
                    "theme_counts_json": "{}",
                    "vehicle_model_distinct": 0,
                },
            )
        if batch_id is None:
            batch_id = batches[0].id
        batch = db.get(Batch, batch_id)
        if not batch:
            batch = batches[0]
            batch_id = batch.id
        losses = (
            db.execute(select(Loss).where(Loss.batch_id == batch_id).order_by(Loss.loss_id))
            .scalars()
            .all()
        )
        insight_row = db.get(BatchInsight, batch_id)
        insight = None
        if insight_row:
            try:
                insight = json.loads(insight_row.payload_json)
            except json.JSONDecodeError:
                insight = {}
        consultant_counts: dict[str, int] = {}
        for row in losses:
            consultant_counts[row.consultant] = consultant_counts.get(row.consultant, 0) + 1
        loss_rows: list[dict] = []
        for r in losses:
            tags = json.loads(r.customer_tags or "[]")
            tags_str = ",".join(tags) if isinstance(tags, list) else str(tags)
            loss_rows.append(
                {
                    "loss_id": r.loss_id,
                    "date": r.date,
                    "consultant": r.consultant,
                    "vehicle_model": r.vehicle_model,
                    "customer_tags": tags_str,
                    "loss_description": r.loss_description,
                }
            )
        vehicle_model_distinct = len({r.vehicle_model for r in losses})
        theme_counts = (insight or {}).get("theme_counts") or {}
        if not isinstance(theme_counts, dict):
            theme_counts = {}
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "batches": batches,
                "batch": batch,
                "loss_rows": loss_rows,
                "insight": insight,
                "consultant_counts": consultant_counts,
                "consultant_counts_json": json.dumps(consultant_counts, ensure_ascii=False),
                "theme_counts_json": json.dumps(theme_counts, ensure_ascii=False),
                "vehicle_model_distinct": vehicle_model_distinct,
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
            headers={
                "Content-Disposition": f'attachment; filename="losses_batch_{batch_id}.csv"',
            },
        )
    finally:
        db.close()
