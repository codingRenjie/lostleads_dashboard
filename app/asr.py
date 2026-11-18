"""语音转文字：优先科大讯飞流式听写；未配置讯飞时回退 OpenAI 兼容 Whisper。"""

from pathlib import Path

from openai import OpenAI

from app.config import settings


def transcribe_file(audio_path: Path) -> str:
    if settings.iflytek_configured():
        from app.asr_iflytek import transcribe_iflytek

        return transcribe_iflytek(audio_path)

    base = settings.effective_asr_base().lower()
    if "moonshot.cn" in base or "api.moonshot" in base:
        raise RuntimeError(
            "未配置可用的科大讯飞 ASR：当前 LLM_BASE_URL 指向 Moonshot（Kimi），"
            "Kimi 不提供与 Whisper 相同的语音转写接口，不能复用 LLM_KEY 做转写。"
            "请在项目根目录 .env 填写三项（名称需完全一致）：IFLYTEK_APP_ID、IFLYTEK_API_KEY、IFLYTEK_API_SECRET，"
            "并在讯飞开放平台为该应用开通「语音听写（流式版）」且保证有剩余服务量。"
        )

    key = settings.effective_asr_key()
    if not key:
        raise RuntimeError(
            "未配置语音识别：请在 .env 填写科大讯飞 IFLYTEK_APP_ID / IFLYTEK_API_KEY / IFLYTEK_API_SECRET，"
            "或配置 ASR_API_KEY + ASR_BASE_URL 使用 Whisper 兼容接口。"
        )

    client = OpenAI(api_key=key, base_url=settings.effective_asr_base())
    with audio_path.open("rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=settings.asr_model,
            file=audio_file,
        )
    return (result.text or "").strip()
