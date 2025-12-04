"""Voice translation pipeline wired to Qwen3 ASR & CosyVoice APIs."""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
import threading
import queue
import warnings
import wave
from pathlib import Path
from typing import Callable, Optional, Union

from voice_clone import get_clone_voice_id, synthesize_with_clone

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv()
except ImportError:
    pass



# 抑制 Windows asyncio 的警告
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overlapped.*')

try:
    import dashscope  # type: ignore[import-not-found]
    from dashscope import Transcription, Generation  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover - optional dependency
    dashscope = None  # type: ignore[assignment]
    Transcription = None  # type: ignore[assignment, misc]
    Generation = None  # type: ignore[assignment, misc]
    _dashscope_import_error = str(e)

try:
    import websocket  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    websocket = None  # type: ignore[assignment]

try:
    import websocket  # type: ignore[import-not-found]
    import threading
except ImportError:  # pragma: no cover - optional dependency
    websocket = None  # type: ignore[assignment]
    threading = None  # type: ignore[assignment]


AudioInput = Union[str, Path, bytes]

# 语言代码映射：将语言名称转换为 API 需要的语言代码
_LANGUAGE_CODE_MAP = {
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
}

# 语言名称映射：用于翻译提示
_LANGUAGE_NAME_MAP = {
    "chinese": "中文",
    "english": "英文",
    "japanese": "日文",
    "korean": "韩文",
}


def _normalize_language_code(language: str) -> str:
    """将语言名称转换为 API 需要的语言代码"""
    language_lower = language.lower().strip()
    return _LANGUAGE_CODE_MAP.get(language_lower, language_lower)


QWEN_ASR_MODEL = os.getenv("QWEN_ASR_MODEL", "qwen3-asr-flash-realtime")
QWEN_ASR_ENDPOINT = os.getenv(
    "QWEN_ASR_ENDPOINT",
    "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
)
QWEN_API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
# 减少默认超时时间，避免长时间阻塞（15秒足够处理大部分语音）
QWEN_TIMEOUT = float(os.getenv("QWEN_ASR_TIMEOUT", "15"))

def _ensure_audio_path(audio_input: AudioInput) -> Path:
    if isinstance(audio_input, Path):
        return audio_input
    if isinstance(audio_input, str):
        return Path(audio_input)
    if isinstance(audio_input, bytes):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_input)
        tmp.flush()
        return Path(tmp.name)
    raise TypeError(f"Unsupported audio input type: {type(audio_input)!r}")


def _load_audio_bytes(audio_file_path: Path) -> bytes:
    """读取音频文件（WAV/PCM）原始字节用于上传。"""
    if audio_file_path.suffix.lower() == ".wav":
        with wave.open(str(audio_file_path), "rb") as wav_file:
            nchannels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            if nchannels != 1 or sampwidth != 2 or framerate != 16000:
                print(
                    f"[警告] WAV 参数为 {nchannels}声道/{sampwidth * 8}bit/{framerate}Hz，"
                    "与 ASR 预期 1ch/16bit/16000Hz 不符，可能导致识别失败。"
                )
    return audio_file_path.read_bytes()


def _ensure_dashscope_installed() -> None:
    if dashscope is None:
        error_msg = "需要 `dashscope` 依赖以调用 DashScope API，请先 `pip install dashscope`。"
        if '_dashscope_import_error' in globals():
            error_msg += f"\n导入错误详情: {_dashscope_import_error}"
        raise RuntimeError(error_msg)
    if Transcription is None:
        raise RuntimeError("无法从 dashscope 导入 Transcription，请确保 dashscope 版本正确。")


def _encode_audio(audio_path: Path) -> tuple[str, str]:
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    fmt = audio_path.suffix.lstrip(".") or "wav"
    return audio_b64, fmt


def _ensure_api_key(name: str, value: Optional[str]) -> str:
    if value:
        return value
    raise RuntimeError(f"{name} 未配置，请设置环境变量提供 API Key。")


def asr_transcribe(
    audio_input: AudioInput,
    source_language: str,
    *,
    model_name: str = QWEN_ASR_MODEL,
    endpoint: str = QWEN_ASR_ENDPOINT,
    api_key: Optional[str] = QWEN_API_KEY,
    timeout: float = QWEN_TIMEOUT,
) -> str:
    """
    使用 DashScope WebSocket 实时 ASR API 进行语音识别。
    """
    if websocket is None:
        raise RuntimeError("需要 `websocket-client` 依赖以调用 DashScope WebSocket API，请先 `pip install websocket-client`。")
    
    audio_path = _ensure_audio_path(audio_input)
    
    # 转换语言代码
    language_code = _normalize_language_code(source_language)
    
    # 设置 API Key
    api_key_value = _ensure_api_key('QWEN_API_KEY', api_key)
    
    # 构建 WebSocket URL
    ws_url = f"{endpoint}?model={model_name}"
    ws_headers = [
        f"Authorization: Bearer {api_key_value}",
        "OpenAI-Beta: realtime=v1"
    ]
    
    # 用于存储识别结果的队列
    result_queue: queue.Queue = queue.Queue()
    is_connected = threading.Event()
    is_complete = threading.Event()
    error_occurred = threading.Event()
    error_message = [None]
    
    def on_open(ws):
        print(f"[ASR] WebSocket 连接已建立")
        is_connected.set()
        # 发送会话更新事件
        event = {
            "event_id": "event_session_update",
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "wav",
                "sample_rate": 16000,
                "input_audio_transcription": {
                    "language": language_code
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.2,
                    "silence_duration_ms": 800
                }
            }
        }
        print(f"[ASR] 发送会话更新事件，语言: {language_code}")
        ws.send(json.dumps(event))
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            event_type = data.get("type")
            print(f"[ASR] 收到服务器消息: {event_type}")
            
            # 对于 text 事件，打印完整数据以便调试
            if event_type == "conversation.item.input_audio_transcription.text":
                print(f"[ASR] 完整消息数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # 处理流式识别结果（text 事件）
            if event_type == "conversation.item.input_audio_transcription.text":
                # 流式返回的文本片段
                # 尝试多种方式提取文本
                delta = data.get("delta", "")
                item = data.get("item", {})
                transcript = item.get("transcript", "") if item else ""
                
                # 如果都没有，尝试从 data 直接获取
                if not transcript and not delta:
                    transcript = data.get("transcript", "")
                
                # stash 字段包含流式识别的临时结果
                stash = data.get("stash", "")
                
                # 优先使用 transcript，然后是 stash，最后是 delta
                text_content = transcript if transcript else (stash if stash else delta)
                
                if text_content:
                    print(f"[ASR] 识别文本片段: {text_content}")
                    # 清空队列，只保留最新的结果
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break
                    result_queue.put(text_content)
                    # 标记收到了文本
                    is_complete.clear()  # 清除完成标记，因为可能还有更多文本
                else:
                    print(f"[ASR] 警告: text 事件中没有找到文本内容，原始数据: {data}")
            
            elif event_type == "conversation.item.input_audio_transcription.completed":
                # 完成事件，获取最终结果
                transcript = data.get("transcript", "")
                # 如果没有 transcript，尝试从 item 中获取
                if not transcript:
                    item = data.get("item", {})
                    transcript = item.get("transcript", "")
                
                print(f"[ASR] 识别完成，最终结果: {transcript}")
                if transcript:
                    # 清空队列，放入最终结果
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break
                    result_queue.put(transcript)
                else:
                    print("[ASR] 警告: 识别结果为空")
                is_complete.set()
                try:
                    ws.close()
                except Exception:
                    pass
            elif event_type == "error":
                # 处理错误事件
                error_info = data.get("error", {})
                if isinstance(error_info, dict):
                    error_msg = error_info.get("message", error_info.get("code", "Unknown error"))
                else:
                    error_msg = str(error_info)
                error_message[0] = error_msg
                error_occurred.set()
                try:
                    ws.close()
                except Exception:
                    pass
            elif event_type and "error" in event_type.lower():
                # 捕获其他可能的错误事件类型
                error_msg = data.get("message", data.get("error", "Unknown error"))
                error_message[0] = str(error_msg)
                error_occurred.set()
                try:
                    ws.close()
                except Exception:
                    pass
        except json.JSONDecodeError:
            # 如果不是 JSON，可能是纯文本错误消息
            if "error" in message.lower() or "Error" in message:
                error_message[0] = message
                error_occurred.set()
        except Exception as e:
            # 捕获其他异常
            error_message[0] = f"处理消息时出错: {e}"
            error_occurred.set()
    
    def on_error(ws, error):
        print(f"[ASR] WebSocket 错误: {error}")
        error_message[0] = str(error)
        error_occurred.set()
    
    def on_close(ws, close_status_code, close_msg):
        is_complete.set()
    
    def send_audio(ws, audio_file_path):
        # 等待会话更新完成，确保服务器已准备好接收音频
        print(f"[ASR] 等待 WebSocket 连接...")
        if not is_connected.wait(timeout=5.0):
            error_message[0] = "WebSocket 连接超时，无法发送音频。"
            print(f"[ASR] 错误: {error_message[0]}")
            error_occurred.set()
            return
        
        print(f"[ASR] WebSocket 已连接，等待服务器处理会话更新...")
        # 额外等待一小段时间，确保 session.update 已被服务器处理
        time.sleep(0.5)
        
        audio_path_obj = Path(audio_file_path)
        audio_bytes = _load_audio_bytes(audio_path_obj)
        if not audio_bytes:
            error_message[0] = "音频文件为空，无法发送到 ASR。"
            print(f"[ASR] 错误: {error_message[0]}")
            error_occurred.set()
            return
        
        print(f"[ASR] 开始发送音频数据，大小: {len(audio_bytes)} 字节 ({len(audio_bytes) / 1024:.2f} KB)")
        
        # 按块发送音频数据
        offset = 0
        chunk_size = 3200  # ~0.1s PCM16/16kHz
        audio_sent = False
        
        try:
            while offset < len(audio_bytes) and not is_complete.is_set() and not error_occurred.is_set():
                # 检查连接状态
                if not hasattr(ws, 'sock') or not ws.sock:
                    break
                try:
                    if not ws.sock.connected:
                        break
                except AttributeError:
                    # 某些版本的 websocket-client 可能没有 connected 属性
                    pass
                
                audio_chunk = audio_bytes[offset:offset + chunk_size]
                if not audio_chunk:
                    break
                
                encoded_data = base64.b64encode(audio_chunk).decode('utf-8')
                event = {
                    "event_id": f"event_{int(time.time() * 1000)}",
                    "type": "input_audio_buffer.append",
                    "audio": encoded_data,
                }
                try:
                    ws.send(json.dumps(event))
                    audio_sent = True
                    offset += chunk_size
                    time.sleep(0.1)  # 模拟实时采集
                except Exception as e:
                    error_message[0] = f"发送音频数据失败: {e}"
                    print(f"[ASR] 错误: {error_message[0]}")
                    error_occurred.set()
                    break
        
            if audio_sent:
                print(f"[ASR] 音频数据发送完成，共发送 {offset} 字节")
                print(f"[ASR] 等待服务器检测语音活动并返回识别结果...")
            else:
                print(f"[ASR] 警告: 未发送任何音频数据")
        except Exception as e:
            error_message[0] = f"发送音频时出错: {e}"
            print(f"[ASR] 错误: {error_message[0]}")
            error_occurred.set()
        
        # 注意：在使用 server_vad 模式时，服务器会自动检测语音活动并提交音频
        # 客户端不应该手动调用 commit，否则会导致 "Error committing input audio buffer" 错误
        # 只有在非 server_vad 模式下才需要手动 commit
        # 由于当前配置使用 server_vad，这里不发送 commit 事件
    
    # 创建 WebSocket 连接
    ws = websocket.WebSocketApp(
        ws_url,
        header=ws_headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # 启动音频发送线程
    audio_thread = threading.Thread(target=send_audio, args=(ws, str(audio_path)))
    audio_thread.daemon = True
    audio_thread.start()
    
    # 运行 WebSocket（阻塞直到连接关闭）
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()
    
    # 等待音频发送线程完成
    print(f"[ASR] 等待音频发送线程完成（超时: {timeout}秒）...")
    audio_thread.join(timeout=timeout)
    
    # 等待完成或超时
    # 在 server_vad 模式下，服务器需要检测到静音后才会返回结果
    # 所以需要等待足够的时间（至少包含 silence_duration_ms + 处理时间）
    wait_timeout = timeout
    print(f"[ASR] 等待识别结果（超时: {wait_timeout}秒）...")
    
    # 流式处理：当检测到句子完成时立即返回
    start_wait = time.time()
    last_text = None
    last_text_time = None
    silence_duration = 0.8  # 如果 0.8 秒内文本没有变化，认为句子完成（更短的等待时间）
    
    try:
        while (time.time() - start_wait) < wait_timeout:
            # 检查是否有错误
            if error_occurred.is_set():
                error_msg = error_message[0] or "未知错误"
                raise RuntimeError(f"WebSocket ASR 错误: {error_msg}")
            
            # 检查是否完成（收到 completed 事件）
            if is_complete.is_set():
                print("[ASR] 收到 completed 事件")
                break
            
            # 检查队列中是否有新的结果（流式返回的文本）
            current_text = None
            if not result_queue.empty():
                # 获取最新的文本
                while not result_queue.empty():
                    try:
                        current_text = result_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # 检查文本是否真的更新了
                if current_text and current_text != last_text:
                    last_text = current_text
                    last_text_time = time.time()
                    print(f"[ASR] 文本更新: {current_text}")
                elif current_text:
                    # 文本没有变化，但仍在更新，重置计时器
                    last_text_time = time.time()
            elif last_text_time is not None:
                # 如果之前收到过文本，但现在队列为空，检查是否已经静默足够长时间
                if (time.time() - last_text_time) > silence_duration:
                    print(f"[ASR] 检测到文本稳定（{silence_duration}秒无变化），认为句子完成")
                    break
            
            # 使用更短的 sleep，并允许 KeyboardInterrupt
            time.sleep(0.1)
    except KeyboardInterrupt:
        # 如果收到中断信号，立即关闭 WebSocket 并重新抛出
        print("[ASR] 收到中断信号，正在关闭 WebSocket...")
        try:
            ws.close()
        except Exception:
            pass
        raise
    
    # 尝试获取结果
    result = None
    if not result_queue.empty():
        # 获取最新的结果
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break
    
    # 确保 WebSocket 已关闭（无论是否有结果）
    try:
        ws.close()
    except Exception:
        pass
    
    # 不等待 WebSocket 线程结束，因为它是 daemon 线程，主程序退出时会自动终止
    # 这样可以避免阻塞，让程序能更快响应 Ctrl+C
    
    if result:
        print(f"[ASR] 最终识别结果: {result}")
        return str(result).strip()
    else:
        # 如果超时但没有结果，返回空字符串而不是抛出异常
        # 这样可以让调用者继续处理，而不是阻塞整个流程
        if error_occurred.is_set():
            error_msg = error_message[0] or "未知错误"
            print(f"[ASR] 警告: WebSocket ASR 错误: {error_msg}")
            return ""  # 返回空字符串，让调用者决定如何处理
        print(f"[ASR] 警告: ASR 请求超时（{timeout}秒），未收到识别结果，跳过本次翻译")
        return ""  # 返回空字符串，不抛出异常，避免阻塞后续语音输入

def translate_text(text: str, source_language: str, target_language: str) -> str:
    """
    使用 DashScope Generation API 进行文本翻译。
    将源语言文本翻译为目标语言。
    """
    print(f"DEBUG: 翻译函数调用")
    print(f"DEBUG: 源语言: {source_language}")
    print(f"DEBUG: 目标语言: {target_language}")
    print(f"DEBUG: 原文: {text}")
    
    if dashscope is None or Generation is None:
        # 如果没有安装 dashscope，使用简单的占位翻译
        print("DEBUG: dashscope 未安装，返回占位翻译")
        return f"{text} -> translated to {target_language}"
    
    # 获取语言名称
    source_lang_name = _LANGUAGE_NAME_MAP.get(source_language.lower(), source_language)
    target_lang_name = _LANGUAGE_NAME_MAP.get(target_language.lower(), target_language)
    
    # 如果源语言和目标语言相同，直接返回原文
    if source_language.lower() == target_language.lower():
        print("DEBUG: 源语言和目标语言相同，跳过翻译")
        return text
    
    # 设置 API Key（如果还没有设置）
    if not dashscope.api_key:
        api_key = QWEN_API_KEY or os.getenv("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = api_key
    
    # 构建翻译提示词 - 使用更明确的指令
    if target_lang_name == "英文":
        prompt = f"Translate the following {source_lang_name} text to English. Only return the English translation, do not add any explanations or comments:\n\n{text}"
    else:
        prompt = f"请将以下{source_lang_name}文本翻译成{target_lang_name}，只返回翻译结果，不要添加任何解释或注释：\n\n{text}"
    
    print(f"DEBUG: 翻译提示词: {prompt[:100]}...")
    
    try:
        # 使用 Generation API 进行翻译
        response = Generation.call(
            model='qwen-turbo',  # 使用 Qwen 模型进行翻译
            prompt=prompt,
            max_tokens=1000,
        )
        
        print(f"DEBUG: 翻译 API 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            translated_text = response.output.text.strip()
            # 清理可能的格式问题
            translated_text = translated_text.replace('"', '').replace("'", '').strip()
            # 移除可能的引号包裹
            if translated_text.startswith('"') and translated_text.endswith('"'):
                translated_text = translated_text[1:-1]
            if translated_text.startswith("'") and translated_text.endswith("'"):
                translated_text = translated_text[1:-1]
            print(f"DEBUG: 翻译结果: {translated_text}")
            return translated_text
        else:
            # 如果 API 调用失败，返回占位翻译
            print(f"DEBUG: 翻译 API 调用失败: {response.message}")
            return f"{text} -> translated to {target_language}"
            
    except Exception as e:
        # 如果出现异常，返回占位翻译
        print(f"DEBUG: 翻译异常: {e}")
        import traceback
        traceback.print_exc()
        return f"{text} -> translated to {target_language}"

def tts(
    text: str,
    target_language: str,
    *,
    model_name: str = "",
    voice: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 0.0,
) -> bytes:
    """
    使用 SiliconFlow CosyVoice2 API 进行语音合成。

    兼容旧参数签名，但内部始终调用 `voice_clone.synthesize_with_clone`。
    """
    if " -> translated to " in text:
        text = text.split(" -> translated to ")[0].strip()

    try:
        return synthesize_with_clone(text, voice_id=voice or get_clone_voice_id())
    except Exception as exc:  # pragma: no cover - 直接向上抛出
        raise RuntimeError(f"SiliconFlow TTS 调用失败: {exc}") from exc


def voice_translator(
    audio_input: AudioInput,
    *,
    target_language: str,
    source_language: str,
    asr: Callable[[AudioInput, str], str] = asr_transcribe,
    translator: Callable[[str, str, str], str] = translate_text,
    tts_engine: Callable[[str, str], bytes] = tts,
) -> bytes:
    """
    Orchestrate ASR -> translation -> TTS for the provided audio sample.
    """
    print("=" * 60)
    print("开始语音翻译流程")
    print(f"源语言: {source_language}")
    print(f"目标语言: {target_language}")
    print("=" * 60)

    print("\n[步骤 1/3] 语音识别 (ASR)...")
    transcribed_text = asr(audio_input, source_language)
    print(f"识别结果: {transcribed_text}")
    
    # 如果 ASR 返回空字符串（超时或失败），跳过本次翻译
    if not transcribed_text or not transcribed_text.strip():
        print("⚠️  警告: ASR 未返回有效结果，跳过本次翻译")
        return b""  # 返回空字节，让调用者知道翻译失败但不抛出异常
    
    print("\n[步骤 2/3] 文本翻译...")
    translated_text = translator(transcribed_text, source_language, target_language)
    print(f"翻译结果: {translated_text}")
    
    # 检查翻译是否真的改变了
    if transcribed_text == translated_text and source_language.lower() != target_language.lower():
        print("⚠️  警告: 翻译结果与原文相同，可能翻译未生效！")
    
    # 如果翻译结果为空，也跳过 TTS
    if not translated_text or not translated_text.strip():
        print("⚠️  警告: 翻译结果为空，跳过 TTS")
        return b""
    
    print("\n[步骤 3/3] 语音合成 (TTS)...")
    audio_result = tts_engine(translated_text, target_language)
    print(f"✓ 语音合成完成，音频大小: {len(audio_result)} 字节")
    print("=" * 60)
    
    return audio_result