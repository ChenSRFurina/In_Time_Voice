## 项目技术报告：In-Time Voice 本地语音翻译与语音克隆系统

### 1. 整体目标与应用场景

本项目实现了一个**本地实时语音翻译系统**，支持两条主要路径：

- **扬声器翻译（在线模式）**：从系统扬声器输出（如 QQ 语音、游戏语音）捕获英文 → 实时翻译成中文 → 播放到扬声器。
- **麦克风翻译（本地模式）**：从麦克风捕获中文 → 实时翻译成英文 → 通过虚拟麦克风输出给 QQ / 游戏等上层应用。

在此基础上，系统集成了 **SiliconFlow CosyVoice2 语音克隆**：

- 用户提供一段本地参考录音（m4a / wav 等）→ 自动转换为支持格式 → 上传到 SiliconFlow → 获取专属音色 `voice_id` → 后续所有 TTS 均使用该音色合成。

### 2. 目录结构与关键模块

- **核心入口**
  - `main.py`：统一命令行入口，支持 `--mode local/online`、`--streaming`、`--virtual-mic` 等参数，内部根据模式选择对应的翻译器。
  - `run.py`：双进程服务入口，同时拉起：
    - 扬声器翻译进程：`python main.py --mode online --streaming ...`
    - 麦克风翻译进程：`python main.py --mode local --streaming --virtual-mic ...`

- **翻译管线**
  - `voice_translator.py`：核心“ASR → 翻译 → TTS”流水线：
    - ASR：调用 Qwen ASR WebSocket（DashScope）。
    - 文本翻译：调用 Qwen 文本 Generation API。
    - TTS：统一通过 `voice_clone.synthesize_with_clone()` 调用 **SiliconFlow CosyVoice2**。
  - `online_translator.py`：扬声器流式翻译（监听扬声器输出），包括：
    - 流式 ASR 回调。
    - 基于句末检测的“句子完成”事件。
    - 去重 / 冷却机制，避免重复翻译和自我回声。
  - `local_translator.py`：麦克风流式翻译（监听本地麦克风或 VoiceMeeter），包括：
    - 设备采样率自适应与重采样逻辑（设备 44.1kHz / 48kHz → ASR 16kHz）。
    - 本地句子结束检测和翻译队列。
    - 将 TTS 结果写入虚拟麦克风输出队列。
  - `streaming_translator.py`：早期流式实现，部分逻辑仍可复用（如虚拟麦克风输出流处理），目前主入口由 `local_translator.py` / `online_translator.py` 负责。

- **语音克隆与 TTS**
  - `voice_clone.py`：封装 SiliconFlow CosyVoice2 语音克隆与 TTS：
    - `_load_config / _save_config`：读取 / 写入 `voice_clone_config.json`，记录 `sf_voice_id`、参考录音路径等。
    - `set_clone_reference(audio_file, sample_text, ...)`：
      - 确认本地录音存在。
      - 若格式非 wav/mp3/pcm/opus，自动通过 **pydub** 或 **ffmpeg** 转换为 WAV。
      - 使用 `upload_voice_sample()` 调用 `https://api.siliconflow.cn/v1/uploads/audio/voice` 上传并创建音色。
      - 从响应中提取 `voiceId` / `uri`，保存为 `sf_voice_id`。
    - `synthesize_with_clone(text, ...)`：调用 `https://api.siliconflow.cn/v1/audio/speech`，使用 `sf_voice_id` 与 CosyVoice2 进行 TTS，返回原始音频字节。
  - `voice_translator.py::tts()`：
    - 简化为直接调用 `synthesize_with_clone(text, voice_id or get_clone_voice_id())`，不再依赖 DashScope TTS。

- **音频与设备工具**
  - `audio_checker.py`：
    - 列出所有音频设备，打印输入 / 输出通道及默认采样率。
    - 提供 `get_default_input_device(avoid_virtual=True)`，用于选择真实麦克风。
    - `find_virtual_audio_input_device()`、`find_speaker_output_device()`：查找虚拟线缆（VB-CABLE）等设备。
  - `voice_recoder.py`：一次性录音到文件的工具函数，用于本地测试非流式翻译。

### 3. 双翻译服务架构（`run.py`）

#### 3.1 总体流程

- **`DualTranslatorService.start()`**：
  - 打印说明与注意事项（VB-CABLE 安装、系统输入输出配置）。
  - 启动两个子进程：
    - 扬声器翻译进程（在线模式，监听扬声器输出）。
    - 麦克风翻译进程（本地模式，监听麦克风，输出到虚拟麦克风）。
  - 主循环监控两个子进程状态，任一退出时打印退出码；双双退出时结束服务。

#### 3.2 扬声器翻译进程

- 从 `main.py` 导入：`SPEAKER_CAPTURE_INDEX`, `SPEAKER_OUTPUT_INDEX`，作为默认输入 / 输出设备索引。
- 若导入失败，通过 `audio_checker` 自动扫描虚拟音频设备。
- 构造命令：

```text
python main.py --mode online --streaming --source-lang english --local-lang chinese --input-device <capture> --output-device <speaker>
```

- 子进程中：
  - `main.py` 解析参数，构建 `StreamingOnlineTranslator` 实例。
  - 启动流式 ASR、翻译队列、播放线程、音频输入流。

#### 3.3 麦克风翻译进程

- 从 `main.py` 导入：`VOICEMEETER_INPUT_INDEX`, `CABLE_OUTPUT_INDEX` 作为默认麦克风输入与虚拟麦克风输出索引。
- 调用 `set_clone_reference(self.mic_voice_clone_file, sample_text="这句话用于语音克隆", apply_scope="mic", force=True)`：
  - 对指定参考录音执行语音克隆；
  - 成功后在子进程 env 中设置 `VOICE_CLONE_ENABLED=1`，并打印 `sf_voice_id`。
- 构造命令：

```text
python main.py --mode local --streaming --local-lang chinese --target-lang english --virtual-mic --input-device <mic> --virtual-mic-device <cable>
```

- 子进程中：
  - `main.py` 解析参数，构建 `StreamingLocalTranslator`，并在每次翻译完成后调用 `voice_translator.tts()` 输出克隆音色英文语音到虚拟麦克风。

### 4. 单翻译模式架构（`main.py`）

#### 4.1 模式选择

- `--mode local`：本地语音翻译（从本地录音 / 麦克风到对方语言）。
- `--mode online`：在线语音翻译（从对方语音到本地播放）。
- `--streaming`：启用流式模式（持续监听）。
- `--virtual-mic`：在本地模式下，将译文音频输出到虚拟麦克风。

#### 4.2 本地模式（非流式）

- 流程：
  1. 如果传入 `--voice` 文件路径，则调用 `translate_local_voice()` 进行一次性处理。
  2. `translate_local_voice` 内部调用 `voice_translator.voice_translator()`：
     - `asr_transcribe()` 调用 Qwen ASR WebSocket 识别原文。
     - `translate_text()` 调用 Qwen Generation 进行文本翻译。
     - `tts()` 调用 SiliconFlow CosyVoice2 合成译文语音。
  3. 若启用 `--virtual-mic`，则通过 `main.py::_output_to_virtual_microphone()` 将音频写入虚拟麦克风。

#### 4.3 在线模式（非流式）

- 接受远端语音文件：
  - `translate_online_voice()`：对方发来的语音文件 / 字节 → ASR → 翻译 → TTS → 返回音频字节。
  - 可结合 `online_translator.play_audio_bytes()` 在本地直接播放。

### 5. 流式翻译内部细节

#### 5.1 音频流与采样率管理

- 输入设备采样率（例如 44.1kHz / 48kHz）通过 `sounddevice.InputStream` 采集。
- 为了匹配 ASR 模型的 16kHz 要求，在回调中使用 `scipy.signal.resample` 或手工线性插值重采样：
  - 计算目标样本数：`num_samples = len(indata) * 16000 / sample_rate`。
  - 对每个通道做重采样。
  - 转换为 int16 PCM 字节流发送给流式 ASR。

#### 5.2 句子结束检测与翻译队列

- 基于流式 ASR 输出增量文本，检测停顿 / 标点来判断句子结束。
- 检测到句子完成时，将文本放入翻译队列，由后台线程：
  - 调用 `translate_text()` 做文本翻译；
  - 调用 `tts()` 生成目标语言音频；
  - 将音频放入播放队列或虚拟麦克风输出队列。

#### 5.3 去重与冷却逻辑

- 使用哈希表记录最近翻译过的文本及时间戳，限定时间窗口内重复文本会被跳过：
  - 防止虚拟麦克风回环导致的“自我翻译”循环。
  - 保证同一句话不会被误翻译多次。

### 6. 语音克隆（SiliconFlow CosyVoice2）流程

#### 6.1 参考录音上传

1. 用户通过 `--mic-voice-clone-file` 指定参考录音（可为 `m4a`、`wav` 等）。
2. `voice_clone._ensure_supported_audio()`：
   - 若后缀不在 `{.wav, .mp3, .pcm, .opus}` 之中：
     - 优先用 `pydub.AudioSegment.from_file()` 转 `wav`；
     - 若不存在 pydub，则 fallback 到系统 `ffmpeg` 命令进行转换；
     - 转换失败会抛出包含 stderr 的错误日志。
3. `upload_voice_sample()` 调用 SiliconFlow 上传音频并附带 `sample_text`：
   - 接口：`POST https://api.siliconflow.cn/v1/uploads/audio/voice`
   - 参数：`model`, `text`（参考文本），`customName`，`file`（音频）。
4. 从返回 JSON 中提取音色标识：
   - 优先使用 `voiceId` / `voice_id`；
   - 若无，则尝试使用 `data.voiceId` / `data.voice_id`；
   - 若仍无，则使用 `uri` 作为备选 voice 标识。

#### 6.2 音色配置与持久化

- `voice_clone_config.json` 中记录：
  - `mode`: `"siliconflow"`
  - `reference_audio`: 原始录音路径
  - `sf_voice_id`: 语音克隆返回的 id / uri
  - `sf_api_key`: 可选，本地覆盖 `SILICONFLOW_API_KEY`
  - `sf_model`: 使用的 CosyVoice2 模型名称
  - `apply_scope`: 如 `"mic"` 表示仅麦克风路径使用该音色

#### 6.3 TTS 调用

1. `voice_translator.tts()` 在检测到 `VOICE_CLONE_ENABLED=1` 时调用 `synthesize_with_clone()`。
2. `synthesize_with_clone`：
   - 读取 `sf_voice_id` 与模型名。
   - 构造 payload：`{"model": ..., "input": text, "voice": voice_id, "format": "mp3"}`。
   - 请求 `POST https://api.siliconflow.cn/v1/audio/speech`。
   - 返回 raw 音频字节用于播放 / 输出。

### 7. 配置与环境变量

- `.env`（不会提交到 Git）：
  - `DASHSCOPE_API_KEY`：Qwen ASR / Generation 使用。
  - `SILICONFLOW_API_KEY`：CosyVoice2 语音克隆与 TTS 使用。
  - 可选：`SILICONFLOW_TTS_MODEL`，自定义 CosyVoice2 模型。

- `voice_clone_config.json`：
  - 由程序运行时创建或更新，记录当前使用的 voice_id 与参考录音信息。

### 8. 未来可扩展点

- **更多语言支持**：
  - 当前主要针对 `chinese ↔ english`，可以在 `voice_translator._LANGUAGE_CODE_MAP` 与翻译提示词中扩展日语、韩语等。

- **更智能的句子结束检测**：
  - 目前逻辑依赖标点和静默时间阈值，可考虑使用能量阈值 + VAD 或引入模型级 endpointing。

- **更精细的音色管理**：
  - 在 `voice_clone_config.json` 中支持多个 voice 配置（如不同人物声音），并通过命令行参数选择当前使用的 voice id。

- **可视化与 UI**：
  - 基于当前技术报告，可以绘制数据流图（ASR → 翻译 → TTS）、设备拓扑图（真实麦克风 / 扬声器 / 虚拟线缆）、状态机图（句子检测 / 冷却 / 去重）。

此报告覆盖了当前项目的核心文件、数据流、语音克隆与 TTS 整合方式，后续你在写文章或画图时，可以直接以模块为单位拆解：
- 顶层架构（`run.py` 双进程 + `main.py` 单进程）。
- 语音翻译流水线（`voice_translator.py`）。
- 设备与流控制（`audio_checker.py`、`local_translator.py`、`online_translator.py`）。
- 语音克隆与 TTS（`voice_clone.py` + SiliconFlow API）。


