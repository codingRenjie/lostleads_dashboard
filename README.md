# Lost Leads Bot（MVP）

本机 Web：选择战报文件夹 → 云 ASR 转写 → 大模型抽取战败记录（对齐 `table_sample.md`）→ SQLite 持久化 → 本批次汇总分析与仪表盘。

## 环境

- Python 3.10+
- **大模型（Kimi）**：Moonshot 开放 API，与 OpenAI SDK 兼容。在 **[项目根目录的 `.env`](.env)** 配置 `LLM_API_KEY`、`LLM_BASE_URL`（默认 `https://api.moonshot.cn/v1`）、`LLM_MODEL`（如 `kimi-k2.6`；旧版 `moonshot-v1-*` 已下线）。说明见 [Moonshot 控制台](https://platform.moonshot.cn/)。
- **语音识别（科大讯飞）**：在 **同一 `.env`** 配置 `IFLYTEK_APP_ID`、`IFLYTEK_API_KEY`、`IFLYTEK_API_SECRET`（控制台开通「语音听写（流式版）」）。**MP3/M4A 等会先经本机 `ffmpeg` 转为 16k 单声道 PCM 再发送**（避免直接切 MP3 字节导致讯飞报 `10043 audioCoding decode fail`）。macOS 可：`brew install ffmpeg`。
- 若不使用讯飞，可改配 `.env.example` 中的 **Whisper 兼容** `ASR_*` 项（需同时不配齐 `IFLYTEK_*`）。
- **性能（可选）**：`PIPELINE_MAX_WORKERS`（并行转写路数）、`LLM_BATCH_SIZE`（≥2 时合并多条转写为一次 Kimi 抽取）、`IFLYTEK_CHUNK_SLEEP_SEC`（讯飞分帧间隔，遇 `10043` 可调高）。见 `.env.example`。

## 快速开始

```bash
cd "/Users/renjiesmacbook/Workspace/For Demo/Lost Leads Bot"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 配置密钥：编辑项目根目录 `.env`（可与 `.env.example` 对照；`.env` 已被 git 忽略勿提交）
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开 <http://127.0.0.1:8000/> 上传文件夹；<http://127.0.0.1:8000/dashboard> 查看仪表盘与 CSV 导出。

## 数据与备份

- 数据库默认：`data/lost_leads.db`（可复制该文件做手动备份）。
- 上传缓存：`data/uploads/<batch_id>/`。

详细需求见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)。
