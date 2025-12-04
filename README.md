## In-Time Voice 本地语音翻译 & 语音克隆

这是一个基于本地麦克风 / 扬声器的实时语音翻译工具，支持：

- 扬声器翻译（在线模式）：监听扬声器输出 → 翻译 → 在扬声器播放
- 麦克风翻译（本地模式）：监听麦克风 → 翻译 → 输出到虚拟麦克风（QQ 等当作麦克风使用）
- 语音克隆：上传一段参考录音到 SiliconFlow，后续翻译语音统一使用该音色合成

### 主要文件说明

- `run.py`：同时启动扬声器翻译和麦克风翻译的总入口
- `main.py`：单路翻译入口（在线 / 本地 / 流式），`run.py` 会调用它
- `voice_translator.py`：ASR + 文本翻译 + TTS（TTS 已统一走 SiliconFlow CosyVoice2）
- `online_translator.py`：扬声器流式翻译逻辑
- `local_translator.py`：麦克风流式翻译逻辑
- `streaming_translator.py`：旧版流式翻译/虚拟麦克风实现（部分逻辑仍被复用）
- `audio_checker.py`：枚举音频设备、自动选择默认设备
- `voice_recoder.py`：一次性录音工具（非流式场景）
- `voice_clone.py`：封装 SiliconFlow 语音克隆与 TTS 调用

### 环境准备

1. 建议使用独立虚拟环境（已在项目中自带一个 `intime_voice/` 目录，仅供本地使用，已通过 `.gitignore` 排除）。
2. 至少需要的依赖（参考）：

```bash
pip install requests sounddevice soundfile pyaudio python-dotenv dashscope
```

3. 在项目根目录创建 `.env`（已被忽略，不会上传到 GitHub）：

```env
DASHSCOPE_API_KEY=你的_dashscope_key
SILICONFLOW_API_KEY=你的_siliconflow_key
```

### 常用命令

- 扬声器 + 麦克风双翻译（推荐）：

```bash
python run.py --mic-voice-clone-file test_data/1.m4a
```

- 仅在线扬声器翻译：

```bash
python main.py --mode online --streaming
```

- 仅本地麦克风翻译：

```bash
python main.py --mode local --streaming --virtual-mic
```

### 推送到 GitHub

在 `D:\code\DSAA2012\intime_voice` 目录下执行一次：

```bash
git init
git remote add origin https://github.com/ChenSRFurina/In_Time_Voice.git
git add .
git commit -m "Initial commit: core translation and voice clone"
git branch -M main
git push -u origin main
```


