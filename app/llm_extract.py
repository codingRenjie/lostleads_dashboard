import json
import logging
import re
from typing import Any

import httpx
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# 长转写 + 网络波动时默认超时过短易触发 ConnectTimeout；适当放宽连接与读超时
_LLM_TIMEOUT = httpx.Timeout(connect=60.0, read=240.0, write=60.0, pool=10.0)

LOSS_KEYWORDS = [
    "战败",
    "流失",
    "弃单",
    "不考虑",
    "无法挽留",
    "跑单了",
    "客户掉了",
]

# 口语/同义表述：用于「首次抽取出空但转写里其实有战败」时的二次尝试
LOSS_HINT_FRAGMENTS = [
    "没买",
    "没订",
    "没成",
    "没拿下",
    "黄了",
    "跑了",
    "竞品",
    "别家",
    "隔壁店",
    "定了",
    "订了",
    "选特斯拉",
    "选理想",
    "选问界",
    "太贵",
    "超预算",
    "再考虑",
    "不买了",
    "不打算",
    "去看了",
    "对比了",
]

EXTRACT_SYSTEM = """你是汽车门店销售战报分析助手。根据销售顾问的语音转写文本，提取「战败客户」结构化信息。

「战败」定义（宜宽不宜窄）：客户最终未在本店成交、明确改选竞品、表示不再考虑、订单流失、接待后流失等，只要叙述里能看出「这一单没成」，就应记一条战败，包括口语（如「没拿下」「黄了」「订了别家」「太贵没谈拢」）。

规则：
1. 仅输出一个 JSON 对象，不要 Markdown 代码块以外的文字。
2. **只有**在整段转写完全不涉及任何客户、任何销售过程时，才允许 losses 为空数组。只要谈到客户且存在未成交/流失/对比竞品等，就应输出至少 1 条 losses。
3. 一条转写可含多个客户战败，每个案例一条。
4. customer_tags：短标签数组（需求特征、决策链、场景、关注点）；严禁真实姓名、手机、地址、公司名等 PII。
5. loss_description：客观叙述客户异议、顾问应对、竞品与离场原因，**建议 60～280 字**；若原文很短，也至少写满 **28 个汉字**，信息不足处可写「转写未提及」但不要编造具体数字。
6. vehicle_model：品牌+车型+配置；无法识别写「待确认」。
7. JSON 结构：{"losses":[{"vehicle_model":"...","customer_tags":["..."],"loss_description":"..."}]}"""


EXTRACT_SYSTEM_FORCE = """你是汽车门店销售战报分析助手。程序已判定这段转写**含有战败或未成交相关语义**。

你必须输出 JSON，且 **losses 数组至少包含 1 条**，从转写中概括战败信息；禁止返回空 losses。
若信息模糊，仍写 1 条并在 loss_description 中如实说明「转写较简略，已知信息为…」。
其余字段规则与常规抽取相同。仅输出 JSON。"""


AGG_SYSTEM = """你是销售管理顾问。根据本批次多条战败记录（JSON），输出汇总分析，供店长晨会使用。
要求：
1. 仅输出 JSON。
2. theme_counts：短标签到数量的映射，标签为自由归纳的战败原因主题（不要用固定 DMS 分类表），2-8 个主题为宜。
3. bottlenecks：字符串数组，2-5 条，指出可能存在的业务瓶颈（需能从记录中推断）。
4. strategies：字符串数组，3-7 条，可执行的销售策略调整建议，需与记录事实呼应。
5. summary：一段 80-150 字中文，概括本批战败整体态势。
JSON 结构：
{"theme_counts":{"主题A":2},"bottlenecks":["..."],"strategies":["..."],"summary":"..."}"""


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """解析模型输出中的 JSON；兼容前后多余说明、Markdown 围栏。"""
    s = _strip_code_fence(raw)
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(s[start : end + 1])
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 LLM 返回的 JSON：{e}") from e
    raise ValueError(f"无法解析 LLM 返回的 JSON，原文片段：{raw[:400]}")


def transcript_suggests_loss(transcript: str) -> bool:
    t = transcript.strip()
    if not t:
        return False
    for kw in LOSS_KEYWORDS:
        if kw in t:
            return True
    for frag in LOSS_HINT_FRAGMENTS:
        if frag in t:
            return True
    return False


def _chat(messages: list[dict[str, str]]) -> str:
    key = settings.llm_api_key
    if not key:
        raise RuntimeError("LLM_API_KEY not configured")
    client = OpenAI(
        api_key=key,
        base_url=settings.llm_base_url.rstrip("/"),
        timeout=_LLM_TIMEOUT,
        max_retries=2,
    )
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        # 新版 Kimi（kimi-k2.6 / kimi-k3 等）仅允许 temperature=1
        temperature=1,
    )
    choice = resp.choices[0].message
    return (choice.content or "").strip()


def _normalize_loss_items(losses: Any) -> list[dict[str, Any]]:
    if not isinstance(losses, list):
        return []
    out: list[dict[str, Any]] = []
    for item in losses:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("loss_description") or "").strip()
        # 过短多为模型敷衍，但强制抽取路径可能略短，下限略降
        if len(desc) < 18:
            continue
        vm = str(item.get("vehicle_model") or "待确认").strip() or "待确认"
        tags = item.get("customer_tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip() for t in tags if str(t).strip()][:20]
        out.append(
            {
                "vehicle_model": vm[:256],
                "customer_tags": tags,
                "loss_description": desc[:2000],
            }
        )
    return out


EXTRACT_BATCH_SYSTEM = """你是汽车门店销售战报分析助手。将一次收到多段「销售语音转写」，分别抽取每段的战败客户信息。

规则与单段抽取相同（战败定义宜宽不宜窄、customer_tags 无 PII、loss_description 60～280 字建议、无车型写待确认）。

输出**仅一个** JSON 对象，结构严格为：
{"files":[{"source":"<必须与用户消息中 ### source: 行完全一致>","losses":[{"vehicle_model":"...","customer_tags":["..."],"loss_description":"..."}]}, ...]}

- 每个输入片段对应 files 中一项；source 字符串必须与输入一致（含路径/文件夹名）。
- 某段无战败则 losses 为 []。
- 不要 Markdown 代码块。"""


def _single_extract_core(transcript: str, consultant: str) -> list[dict[str, Any]]:
    kw_line = "、".join(LOSS_KEYWORDS)
    user = (
        f"顾问姓名（来自文件名，仅供参考）：{consultant}\n"
        f"战败相关常见词（含口语同义，有则更应输出 losses）：{kw_line}\n\n"
        f"转写全文：\n{transcript}\n"
    )
    raw = _chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    data = _parse_llm_json(raw)
    return _normalize_loss_items(data.get("losses"))


def extract_losses_force(transcript: str, consultant: str) -> list[dict[str, Any]]:
    raw2 = _chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM_FORCE},
            {
                "role": "user",
                "content": f"顾问：{consultant}\n\n转写全文：\n{transcript}\n",
            },
        ]
    )
    data2 = _parse_llm_json(raw2)
    return _normalize_loss_items(data2.get("losses"))


def extract_losses_from_transcript(transcript: str, consultant: str) -> list[dict[str, Any]]:
    out = _single_extract_core(transcript, consultant)
    if not out and transcript_suggests_loss(transcript):
        out = extract_losses_force(transcript, consultant)
    return out


def extract_losses_batch(
    segments: list[tuple[str, str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """
    一次 Kimi 请求处理多段转写。segments: (source 标识, consultant, transcript)
    """
    if not segments:
        return {}
    if len(segments) == 1:
        s, c, t = segments[0]
        return {s: extract_losses_from_transcript(t, c)}

    parts: list[str] = []
    cap = 12000
    for source, cons, trans in segments:
        body = trans[:cap]
        if len(trans) > cap:
            body += f"\n…[转写已截断，全文共{len(trans)}字]"
        parts.append(f"### source:{source}\n顾问:{cons}\n转写:\n{body}\n")
    user = "\n".join(parts)
    raw = _chat(
        [
            {"role": "system", "content": EXTRACT_BATCH_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    try:
        data = _parse_llm_json(raw)
    except ValueError:
        logger.warning("批量抽取 JSON 解析失败，改为逐条调用")
        return {s: extract_losses_from_transcript(t, c) for s, c, t in segments}

    files = data.get("files")
    result: dict[str, list[dict[str, Any]]] = {}
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source") or "").strip()
            losses = _normalize_loss_items(item.get("losses"))
            if src:
                result[src] = losses

    for source, cons, trans in segments:
        if source not in result:
            result[source] = []
        if not result[source] and transcript_suggests_loss(trans):
            result[source] = extract_losses_force(trans, cons)
    return result


def aggregate_batch_insights(records: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps(records, ensure_ascii=False)
    user = f"本批次战败记录（JSON 数组，每条含 consultant, vehicle_model, customer_tags, loss_description）：\n{payload}"
    raw = _chat(
        [
            {"role": "system", "content": AGG_SYSTEM},
            {"role": "user", "content": user},
        ]
    )
    data = _parse_llm_json(raw)
    if not isinstance(data, dict):
        return {
            "theme_counts": {},
            "bottlenecks": [],
            "strategies": [],
            "summary": "",
        }
    tc = data.get("theme_counts") or {}
    if not isinstance(tc, dict):
        tc = {}
    tc = {str(k): int(v) for k, v in tc.items() if str(k).strip()}

    def _as_str_list(x: Any) -> list[str]:
        if not isinstance(x, list):
            return []
        return [str(i).strip() for i in x if str(i).strip()]

    return {
        "theme_counts": tc,
        "bottlenecks": _as_str_list(data.get("bottlenecks")),
        "strategies": _as_str_list(data.get("strategies")),
        "summary": str(data.get("summary") or "").strip(),
    }
