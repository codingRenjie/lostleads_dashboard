from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- Kimi（Moonshot，OpenAI 兼容 Chat Completions）----------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.moonshot.cn/v1"
    llm_model: str = "kimi-k2.6"

    # ---------- 科大讯飞 语音听写（流式版）WebSocket ----------
    iflytek_app_id: str = ""
    iflytek_api_key: str = ""
    iflytek_api_secret: str = ""
    iflytek_iat_url: str = "wss://iat-api.xfyun.cn/v2/iat"
    iflytek_language: str = "zh_cn"
    iflytek_domain: str = "iat"
    iflytek_accent: str = "mandarin"

    # ---------- 可选：Whisper 兼容 ASR（未填讯飞时用）----------
    asr_api_key: str = ""
    asr_base_url: str = ""
    asr_model: str = "whisper-1"

    database_url: str = "sqlite:///./data/lost_leads.db"

    # ---------- 性能：并行与讯飞分帧间隔 ----------
    # 同时处理的音频路数（仅 ASR+CPU；LLM 另见 llm_batch_size）
    pipeline_max_workers: int = 4
    # 一次 Kimi 请求合并几条转写（>=2 启用批量抽取；1 则为每条音频单独一次抽取，但与 ASR 仍并行）
    llm_batch_size: int = 4
    # 讯飞 WebSocket 每帧发送后休眠秒数；离线文件可低于官方 40ms 建议以提速，若 10043 可调回 0.04
    iflytek_chunk_sleep_sec: float = 0.01

    @property
    def sqlite_path(self) -> Path:
        url = self.database_url
        if url.startswith("sqlite:////"):
            return Path(url.replace("sqlite:////", "/"))
        if url.startswith("sqlite:///./"):
            return PROJECT_ROOT / url.replace("sqlite:///./", "")
        if url.startswith("sqlite:///"):
            return Path(url.replace("sqlite:///", ""))
        return PROJECT_ROOT / "data" / "lost_leads.db"

    def iflytek_configured(self) -> bool:
        return bool(
            self.iflytek_app_id.strip()
            and self.iflytek_api_key.strip()
            and self.iflytek_api_secret.strip()
        )

    def effective_asr_key(self) -> str:
        return self.asr_api_key or self.llm_api_key

    def effective_asr_base(self) -> str:
        return (self.asr_base_url or self.llm_base_url).rstrip("/")


settings = Settings()
