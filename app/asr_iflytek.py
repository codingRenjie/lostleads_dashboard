"""科大讯飞「语音听写（流式版）」WebSocket 识别（≤60s）。

MP3 等压缩格式**不能**按任意字节切块发送，否则会破坏帧边界，触发 code=10043（audioCoding decode fail）。
本模块统一转为 16kHz / 单声道 / 16bit PCM（raw），再按讯飞建议的 1280 字节/帧发送。
转换依赖本机 **ffmpeg**（需已安装并在 PATH 中）。
"""

import base64
import hashlib
import hmac
import json
import logging
import shutil
import subprocess
import time
from email.utils import formatdate
from pathlib import Path
from urllib.parse import quote, urlparse

import websocket

from app.config import settings


def _chunk_sleep() -> None:
    time.sleep(max(0.0, float(settings.iflytek_chunk_sleep_sec)))

logger = logging.getLogger(__name__)

# 16kHz、16bit、单声道：40ms 一帧 = 16000 * 2 * 0.04 = 1280 字节（与官方 PCM 建议一致）
CHUNK_BYTES = 1280


def _assemble_ws_url(host_url: str, api_key: str, api_secret: str) -> str:
    parsed = urlparse(host_url)
    host = parsed.netloc
    path = parsed.path or "/v2/iat"
    date = formatdate(usegmt=True)
    sign_str = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
    sign_sha = hmac.new(
        api_secret.encode("utf-8"),
        sign_str.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    signature = base64.b64encode(sign_sha).decode("utf-8")
    auth_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(auth_origin.encode("utf-8")).decode("utf-8")
    q = (
        f"authorization={quote(authorization, safe='')}"
        f"&date={quote(date, safe='')}"
        f"&host={quote(host, safe='')}"
    )
    return f"{host_url}?{q}"


def _read_wav_pcm16k_mono(path: Path) -> bytes:
    import wave

    with wave.open(str(path), "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        if ch != 1 or sw != 2 or rate != 16000:
            raise ValueError("not 16k/mono/16bit")
        return wf.readframes(wf.getnframes())


def _ffmpeg_to_pcm_16k_mono(path: Path) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "本机未找到 ffmpeg，无法把 MP3/M4A 等转为讯飞所需的 PCM。"
            "macOS 可执行：brew install ffmpeg；安装后重启终端再运行 uvicorn。"
        )
    r = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        capture_output=True,
        timeout=120,
    )
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg 解码失败：{err[:800] or '未知错误'}")
    if not r.stdout:
        raise RuntimeError("ffmpeg 输出为空，请检查音频是否过短或已损坏")
    return r.stdout


def _load_audio_pcm(path: Path) -> bytes:
    """得到 16kHz 单声道 16bit 小端 PCM（s16le）原始字节。"""
    ext = path.suffix.lower()
    if ext == ".wav":
        try:
            return _read_wav_pcm16k_mono(path)
        except ValueError:
            logger.info("WAV 非 16k/mono/16bit，改用 ffmpeg：%s", path.name)
    return _ffmpeg_to_pcm_16k_mono(path)


def _split_chunks(pcm: bytes) -> list[bytes]:
    if not pcm:
        return []
    if len(pcm) < 4 or len(pcm) % 2 != 0:
        raise RuntimeError("PCM 长度无效（需至少 4 字节且为偶数，请检查源文件或 ffmpeg 输出）")
    out: list[bytes] = []
    for i in range(0, len(pcm), CHUNK_BYTES):
        out.append(pcm[i : i + CHUNK_BYTES])
    if len(out) == 1:
        b = out[0]
        n = len(b)
        mid = (n // 2 // 2) * 2
        if mid < 2:
            mid = 2
        if n - mid < 2:
            mid = n - 2
        out = [b[:mid], b[mid:]]
    return out


def _frame_code_ok(code: object) -> bool:
    if code is None:
        return True
    try:
        return int(code) == 0
    except (TypeError, ValueError):
        return False


def _extract_text_from_result(data: dict) -> str:
    result = data.get("result") or {}
    parts: list[str] = []
    for ws in result.get("ws") or []:
        for cw in ws.get("cw") or []:
            w = cw.get("w")
            if w:
                parts.append(str(w))
    return "".join(parts)


def transcribe_iflytek(audio_path: Path) -> str:
    app_id = settings.iflytek_app_id.strip()
    api_key = settings.iflytek_api_key.strip()
    api_secret = settings.iflytek_api_secret.strip()
    if not (app_id and api_key and api_secret):
        raise RuntimeError("未配置 IFLYTEK_APP_ID / IFLYTEK_API_KEY / IFLYTEK_API_SECRET")

    pcm = _load_audio_pcm(audio_path)
    chunks = _split_chunks(pcm)
    if not chunks:
        raise RuntimeError("有效 PCM 长度为 0")

    encoding = "raw"
    url = _assemble_ws_url(settings.iflytek_iat_url, api_key, api_secret)
    last_err: Exception | None = None
    ws = None
    for attempt in range(3):
        try:
            ws = websocket.create_connection(url, timeout=90)
            break
        except Exception as e:
            last_err = e
            logger.warning("讯飞 WebSocket 连接失败（第 %s/3 次）：%s", attempt + 1, e)
            time.sleep(2.0 * (attempt + 1))
    if ws is None:
        raise RuntimeError(
            "无法连接科大讯飞听写服务（已重试 3 次）。"
            "若为 DNS 错误，请检查本机网络/DNS；若使用代理/VPN，可尝试关闭或放行 iat-api.xfyun.cn。"
        ) from last_err

    texts: list[str] = []
    try:
        for i, chunk in enumerate(chunks):
            if i == 0:
                status = 0
            elif i == len(chunks) - 1:
                status = 2
            else:
                status = 1
            b64 = base64.b64encode(chunk).decode("utf-8")
            if i == 0:
                payload = {
                    "common": {"app_id": app_id},
                    "business": {
                        "language": settings.iflytek_language,
                        "domain": settings.iflytek_domain,
                        "accent": settings.iflytek_accent,
                    },
                    "data": {
                        "status": status,
                        "format": "audio/L16;rate=16000",
                        "encoding": encoding,
                        "audio": b64,
                    },
                }
            else:
                payload = {
                    "data": {
                        "status": status,
                        "format": "audio/L16;rate=16000",
                        "encoding": encoding,
                        "audio": b64,
                    }
                }
            try:
                ws.send(json.dumps(payload))
            except OSError as e:
                raise RuntimeError(
                    f"向科大讯飞发送音频时连接中断（{e!r}）。"
                    "常见于服务端在上一帧解码失败后主动断开；请确认本机已安装 ffmpeg 且 MP3 已转为 PCM 发送。"
                ) from e
            _chunk_sleep()

        while True:
            try:
                raw_msg = ws.recv()
            except websocket.WebSocketConnectionClosedException:
                break
            if not raw_msg:
                continue
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue
            code = msg.get("code")
            if not _frame_code_ok(code):
                err = msg.get("message") or str(msg)
                extra = ""
                if str(code) == "10043":
                    extra = "（多为音频编码/分帧问题；当前版本已用 ffmpeg 转 PCM，若仍失败请检查 ffmpeg 版本与源文件是否损坏）"
                raise RuntimeError(f"科大讯飞识别错误（code={code}）：{err}{extra}")
            data = msg.get("data")
            if isinstance(data, dict):
                piece = _extract_text_from_result(data)
                if piece:
                    texts.append(piece)
    finally:
        try:
            ws.close()
        except Exception:
            pass

    merged = "".join(texts).strip()
    if not merged:
        raise RuntimeError(
            "科大讯飞返回空识别文本。常见原因："
            "① 控制台「剩余服务量」为 0；"
            "② 未开通「语音听写（流式版）」；"
            "③ AppId / APIKey / APISecret 不一致；"
            "④ 音频过长或静音；⑤ 未安装 ffmpeg 导致无法正确转码。"
        )
    return merged
