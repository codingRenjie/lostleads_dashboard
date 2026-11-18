import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.asr import transcribe_file
from app.config import settings
from app.db import Batch, BatchInsight, FileError, Loss, SessionLocal
from app.llm_extract import (
    aggregate_batch_insights,
    extract_losses_batch,
    extract_losses_from_transcript,
    transcript_suggests_loss,
)

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1]
    base = re.sub(r"[^\w.\-\u4e00-\u9fff]", "_", base, flags=re.UNICODE)
    return (base[:200] or "audio").strip()


def consultant_from_logical_name(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1]
    stem = Path(base).stem
    if "_" in stem:
        part = stem.split("_", 1)[0].strip()
        return part or stem.strip() or "待确认"
    return stem.strip() or "待确认"


def cn_date_from_ms(last_modified_ms: int | None) -> str:
    if last_modified_ms is None:
        dt = datetime.now()
    else:
        try:
            dt = datetime.fromtimestamp(last_modified_ms / 1000.0)
        except (OSError, ValueError, OverflowError):
            dt = datetime.now()
    return f"{dt.year}年{dt.month}月{dt.day}日"


def _max_loss_sequence(session) -> int:
    rows = session.execute(select(Loss.loss_id)).scalars().all()
    mx = 0
    for lid in rows:
        if lid and isinstance(lid, str) and lid.upper().startswith("L"):
            try:
                mx = max(mx, int(lid[1:], 10))
            except ValueError:
                continue
    return mx


def _allocate_loss_ids(session, n: int) -> list[str]:
    start = _max_loss_sequence(session)
    return [f"L{start + i + 1:04d}" for i in range(n)]


def _asr_worker(args: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    index, task = args
    logical = task["logical_name"]
    path = task["path"]
    lm = task.get("last_modified_ms")
    consultant = consultant_from_logical_name(logical)
    date_str = cn_date_from_ms(lm if isinstance(lm, int) else None)
    try:
        transcript = transcribe_file(path)
    except Exception as e:
        logger.exception("ASR failed for %s", logical)
        return {
            "index": index,
            "logical_name": logical,
            "consultant": consultant,
            "date_str": date_str,
            "error": str(e)[:2000],
            "transcript": None,
        }
    if not (transcript or "").strip():
        return {
            "index": index,
            "logical_name": logical,
            "consultant": consultant,
            "date_str": date_str,
            "error": "转写结果为空",
            "transcript": None,
        }
    return {
        "index": index,
        "logical_name": logical,
        "consultant": consultant,
        "date_str": date_str,
        "error": None,
        "transcript": transcript.strip(),
    }


def _llm_one_row(r: dict[str, Any]) -> dict[str, Any]:
    if r.get("error"):
        return {**r, "losses": []}
    try:
        losses = extract_losses_from_transcript(r["transcript"], r["consultant"])
        return {**r, "losses": losses}
    except Exception as e:
        logger.exception("LLM extract failed for %s", r["logical_name"])
        return {
            **r,
            "losses": [],
            "error": f"战败抽取失败: {e}"[:2000],
        }


def _flush_llm_batch(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunk:
        return []
    if settings.llm_batch_size < 2 or len(chunk) == 1:
        return [_llm_one_row(chunk[0])]
    segs = [(r["logical_name"], r["consultant"], r["transcript"]) for r in chunk]
    try:
        m = extract_losses_batch(segs)
    except Exception as e:
        logger.warning("批量战败抽取失败，改为逐条: %s", e)
        return [_llm_one_row(r) for r in chunk]
    out: list[dict[str, Any]] = []
    for r in chunk:
        losses = m.get(r["logical_name"])
        if losses is None:
            losses = []
        out.append({**r, "losses": losses})
    return out


def _run_llm_phase(asr_done: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """在 ASR 结果有序列表上跑 LLM：batch_size>=2 时分组批量，否则并行逐条。"""
    if settings.llm_batch_size >= 2:
        filled: list[dict[str, Any]] = []
        i = 0
        n = len(asr_done)
        bs = max(2, settings.llm_batch_size)
        while i < n:
            row = asr_done[i]
            if row.get("error"):
                filled.append({**row, "losses": []})
                i += 1
                continue
            chunk: list[dict[str, Any]] = []
            while i < n and len(chunk) < bs and not asr_done[i].get("error"):
                chunk.append(asr_done[i])
                i += 1
            filled.extend(_flush_llm_batch(chunk))
        return filled

    w = max(1, min(settings.pipeline_max_workers, len(asr_done)))
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(_llm_one_row, asr_done))


def process_batch_files(batch_id: int, tasks: list[dict[str, Any]]) -> None:
    """
    tasks: list of { "path": Path, "logical_name": str, "last_modified_ms": int | None }
    """
    session = SessionLocal()
    try:
        batch = session.get(Batch, batch_id)
        if not batch:
            logger.error("batch %s not found", batch_id)
            return
        batch.status = "processing"
        batch.error_message = None
        session.commit()

        n = len(tasks)
        workers = max(1, min(settings.pipeline_max_workers, n))

        asr_results: list[dict[str, Any] | None] = [None] * n
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_asr_worker, (i, tasks[i])): i for i in range(n)}
            for fut in as_completed(futs):
                i = futs[fut]
                asr_results[i] = fut.result()

        asr_done = [asr_results[i] for i in range(n) if asr_results[i] is not None]
        filled = _run_llm_phase(asr_done)

        for row in filled:
            if row.get("error"):
                session.add(
                    FileError(
                        batch_id=batch_id,
                        filename=row["logical_name"],
                        message=row["error"],
                    )
                )
            elif not row.get("losses") and row.get("transcript"):
                logger.info(
                    "本文件未产出战败行：%s 转写长度=%s 含战败线索=%s 预览=%r",
                    row["logical_name"],
                    len(row["transcript"]),
                    transcript_suggests_loss(row["transcript"]),
                    row["transcript"][:160],
                )

        flat: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for row in filled:
            if row.get("error"):
                continue
            for item in row.get("losses") or []:
                flat.append((row, item))

        if flat:
            ids = _allocate_loss_ids(session, len(flat))
            for idx, (row, item) in enumerate(flat):
                session.add(
                    Loss(
                        loss_id=ids[idx],
                        batch_id=batch_id,
                        date=row["date_str"],
                        consultant=row["consultant"],
                        vehicle_model=item["vehicle_model"],
                        customer_tags=json.dumps(item["customer_tags"], ensure_ascii=False),
                        loss_description=item["loss_description"],
                        source_filename=row["logical_name"],
                    )
                )
        session.commit()

        all_losses = (
            session.execute(select(Loss).where(Loss.batch_id == batch_id)).scalars().all()
        )
        if all_losses:
            records = [
                {
                    "consultant": r.consultant,
                    "vehicle_model": r.vehicle_model,
                    "customer_tags": json.loads(r.customer_tags or "[]"),
                    "loss_description": r.loss_description,
                }
                for r in all_losses
            ]
            try:
                insight = aggregate_batch_insights(records)
            except Exception as e:
                logger.exception("batch insight failed")
                batch.status = "complete"
                batch.error_message = f"汇总分析失败（已保存战败行）: {e}"[:2000]
                session.merge(
                    BatchInsight(
                        batch_id=batch_id,
                        payload_json=json.dumps(
                            {
                                "theme_counts": {},
                                "bottlenecks": [],
                                "strategies": [],
                                "summary": "",
                                "error": str(e),
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()
                return

            session.merge(
                BatchInsight(
                    batch_id=batch_id,
                    payload_json=json.dumps(insight, ensure_ascii=False),
                )
            )

        batch.status = "complete"
        batch.error_message = None
        session.commit()
    except Exception as e:
        logger.exception("batch %s fatal", batch_id)
        try:
            batch = session.get(Batch, batch_id)
            if batch:
                batch.status = "failed"
                batch.error_message = str(e)[:2000]
                session.commit()
        except Exception:
            session.rollback()
    finally:
        session.close()
