"""按本地日历日合并批次，供仪表盘「当日 / 历史」使用。"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Batch, BatchInsight, Loss

TZ = ZoneInfo("Asia/Shanghai")


def batch_day_key(batch: Batch) -> str:
    dt = batch.created_at
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ).date().isoformat()


def format_day_label(day: str) -> str:
    """2026-04-08 -> 2026年4月8日"""
    try:
        y, m, d = day.split("-")
        return f"{int(y)}年{int(m)}月{int(d)}日"
    except ValueError:
        return day


def _merge_insights(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    theme_counts: dict[str, int] = {}
    bottlenecks: list[str] = []
    strategies: list[str] = []
    summaries: list[str] = []
    seen_b: set[str] = set()
    seen_s: set[str] = set()
    for p in payloads:
        tc = p.get("theme_counts") or {}
        if isinstance(tc, dict):
            for k, v in tc.items():
                key = str(k).strip()
                if not key:
                    continue
                try:
                    theme_counts[key] = theme_counts.get(key, 0) + int(v)
                except (TypeError, ValueError):
                    continue
        for x in p.get("bottlenecks") or []:
            s = str(x).strip()
            if s and s not in seen_b:
                seen_b.add(s)
                bottlenecks.append(s)
        for x in p.get("strategies") or []:
            s = str(x).strip()
            if s and s not in seen_s:
                seen_s.add(s)
                strategies.append(s)
        sm = str(p.get("summary") or "").strip()
        if sm:
            summaries.append(sm)
    summary = " ".join(summaries)
    if len(summary) > 600:
        summary = summary[:597] + "…"
    return {
        "theme_counts": theme_counts,
        "bottlenecks": bottlenecks,
        "strategies": strategies,
        "summary": summary,
    }


def build_day_groups(session: Session) -> dict[str, Any]:
    batches = session.execute(select(Batch).order_by(Batch.id.desc())).scalars().all()
    by_day: dict[str, list[Batch]] = defaultdict(list)
    for b in batches:
        by_day[batch_day_key(b)].append(b)

    day_keys = sorted(by_day.keys(), reverse=True)
    latest_day = day_keys[0] if day_keys else None
    history_days = day_keys[1:] if len(day_keys) > 1 else []

    day_summaries = []
    for day in day_keys:
        bs = by_day[day]
        ids = [b.id for b in bs]
        loss_n = len(
            session.execute(select(Loss).where(Loss.batch_id.in_(ids))).scalars().all()
        )
        day_summaries.append(
            {
                "day": day,
                "label": format_day_label(day),
                "batch_ids": ids,
                "batch_count": len(bs),
                "loss_count": loss_n,
                "is_latest": day == latest_day,
            }
        )

    return {
        "batches": batches,
        "by_day": dict(by_day),
        "day_keys": day_keys,
        "latest_day": latest_day,
        "history_days": history_days,
        "day_summaries": day_summaries,
    }


def load_day_view(session: Session, day: str, by_day: dict[str, list[Batch]]) -> dict[str, Any]:
    day_batches = by_day.get(day) or []
    batch_ids = [b.id for b in day_batches]
    losses: list[Loss] = []
    if batch_ids:
        losses = (
            session.execute(
                select(Loss).where(Loss.batch_id.in_(batch_ids)).order_by(Loss.loss_id)
            )
            .scalars()
            .all()
        )

    insight_payloads: list[dict[str, Any]] = []
    for bid in batch_ids:
        row = session.get(BatchInsight, bid)
        if not row:
            continue
        try:
            data = json.loads(row.payload_json)
            if isinstance(data, dict):
                insight_payloads.append(data)
        except json.JSONDecodeError:
            continue
    insight = _merge_insights(insight_payloads) if insight_payloads else None

    consultant_counts: dict[str, int] = {}
    loss_rows: list[dict[str, Any]] = []
    for r in losses:
        consultant_counts[r.consultant] = consultant_counts.get(r.consultant, 0) + 1
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

    statuses = {b.status for b in day_batches}
    if not day_batches:
        status = "empty"
    elif statuses == {"complete"}:
        status = "complete"
    elif "failed" in statuses and "processing" not in statuses and "pending" not in statuses:
        status = "failed"
    elif "processing" in statuses or "pending" in statuses:
        status = "processing"
    else:
        status = ",".join(sorted(statuses))

    theme_counts = (insight or {}).get("theme_counts") or {}
    if not isinstance(theme_counts, dict):
        theme_counts = {}

    return {
        "day": day,
        "day_label": format_day_label(day),
        "batch_ids": batch_ids,
        "batch_count": len(day_batches),
        "status": status,
        "loss_rows": loss_rows,
        "insight": insight,
        "consultant_counts": consultant_counts,
        "theme_counts": theme_counts,
        "vehicle_model_distinct": len({r.vehicle_model for r in losses}),
    }
