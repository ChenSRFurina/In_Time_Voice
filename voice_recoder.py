"""麦克风录音功能，支持按键触发录音。"""

from __future__ import annotations

import audioop  # type: ignore[import-not-found]
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import pyaudio  # type: ignore[import-not-found]
except ImportError:
    pyaudio = None  # type: ignore[assignment]

try:
    import keyboard  # type: ignore[import-not-found]
except ImportError:
    keyboard = None  # type: ignore[assignment]


# 音频参数
AUDIO_FORMAT = pyaudio.paInt16 if pyaudio else None  # 16位深度
CHANNELS = 1  # 单声道
SAMPLE_RATE = 16000  # 采样率（与 ASR API 匹配）
CHUNK = 3200  # 每次读取的帧数（约 0.2 秒）


def _ensure_recording_dependencies(require_keyboard: bool = True) -> None:
    """检查录音依赖是否安装"""
    if pyaudio is None:
        raise RuntimeError("需要 `pyaudio` 库进行录音，请先 `pip install pyaudio`")
    if require_keyboard and keyboard is None:
        raise RuntimeError("需要 `keyboard` 库进行按键监听，请先 `pip install keyboard`")


def record_audio(
    output_path: Optional[Path] = None,
    max_duration: float = 60.0,
    output_dir: Optional[Path] = None,
    *,
    auto_mode: bool = False,
    silence_threshold: int = 500,
    silence_duration: float = 1.0,
) -> Path:
    """
    录音功能：
        - 默认模式：按下 'k' 键开始，再次按下 'k' 键停止
        - auto_mode=True：持续录制直到检测到静音或达到 max_duration
    
    Args:
        output_path: 输出文件路径，如果为 None 则自动生成
        max_duration: 最大录音时长（秒），超过后自动停止
        output_dir: 输出目录，仅在 output_path 为 None 时使用
        auto_mode: True 时无需按键，检测静音自动停止
        silence_threshold: auto_mode 下识别静音的音量阈值（audioop RMS）
        silence_duration: auto_mode 下持续静音多久视为完成（秒）
    
    Returns:
        保存的音频文件路径
    """
    _ensure_recording_dependencies(require_keyboard=not auto_mode)

    if output_path is None:
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            output_path = output_dir / f"recording_{timestamp}.wav"
        else:
            output_path = Path(f"recording_{timestamp}.wav")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if auto_mode:
        return _record_audio_auto(
            output_path,
            max_duration=max_duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
        )
    return _record_audio_interactive(output_path, max_duration=max_duration)


def _record_audio_interactive(output_path: Path, *, max_duration: float) -> Path:
    # 录音状态
    is_recording = False
    frames = []
    recording_lock = threading.Lock()
    
    # 初始化 PyAudio
    audio = pyaudio.PyAudio()
    
    def toggle_recording():
        """切换录音状态"""
        nonlocal is_recording
        with recording_lock:
            if is_recording:
                is_recording = False
                print("\n[录音停止] 正在保存文件...")
            else:
                is_recording = True
                frames.clear()
                print("\n[开始录音] 请说话... (再次按 'k' 键停止)")
    
    # 注册 'k' 键热键
    keyboard.add_hotkey('k', toggle_recording)
    
    print("=" * 60)
    print("语音翻译录音工具")
    print("=" * 60)
    print("按 'k' 键开始录音，再次按 'k' 键停止录音")
    print("按 'Esc' 键退出程序")
    print("=" * 60)
    print("\n等待按键...")
    
    try:
        # 打开音频流
        stream = audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        start_time = None

        while True:
            if keyboard.is_pressed('esc'):
                print("\n[退出] 程序已退出")
                break
            
            with recording_lock:
                recording = is_recording
            
            if recording:
                if start_time is None:
                    start_time = time.time()
                
                # 检查最大录音时长
                if time.time() - start_time > max_duration:
                    print(f"\n[自动停止] 已达到最大录音时长 {max_duration} 秒")
                    is_recording = False
                    break
                
                # 读取音频数据
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    print(f"\n[错误] 读取音频数据失败: {e}")
                    break
            else:
                if start_time is not None and frames:
                    # 录音已停止，保存文件
                    break
                start_time = None
                time.sleep(0.1)  # 避免 CPU 占用过高
        
        sample_width = audio.get_sample_size(AUDIO_FORMAT)
        if not frames:
            raise RuntimeError("没有录制到音频数据")
        _persist_recording(output_path, frames, sample_width)
        return output_path
            
    except KeyboardInterrupt:
        print("\n[中断] 用户中断")
        raise
    except Exception as e:
        print(f"\n[错误] 录音失败: {e}")
        raise
    finally:
        # 清理资源
        _safe_close_stream(stream)
        _safe_terminate_audio(audio)
    return output_path


def _record_audio_auto(
    output_path: Path,
    *,
    max_duration: float,
    silence_threshold: int,
    silence_duration: float,
) -> Path:
    """无需热键的自动录音模式。"""
    import signal
    
    frames = []
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    sample_width = audio.get_sample_size(AUDIO_FORMAT)

    silence_chunk_limit = max(1, int((silence_duration * SAMPLE_RATE) / CHUNK))
    silent_chunks = 0
    start_time = time.time()
    
    # 标记是否应该停止录音
    should_stop = False
    
    def signal_handler(sig, frame):
        nonlocal should_stop
        should_stop = True
        print("\n[自动录音] 收到中断信号，准备停止录音...")
    
    # 设置信号处理
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    print("[自动录音] 开始捕获音频...")

    try:
        while not should_stop:
            try:
                # 使用非阻塞读取，避免长时间阻塞
                # 但 pyaudio 的 read 方法本身是阻塞的，所以我们只能尽快处理
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                rms = audioop.rms(data, 2)  # 16bit -> width=2
                if rms < silence_threshold:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                elapsed = time.time() - start_time
                if silent_chunks >= silence_chunk_limit and elapsed > 0.5:
                    print("[自动录音] 检测到持续静音，停止录音。")
                    break
                if elapsed >= max_duration:
                    print(f"[自动录音] 已达到最大录音时长 {max_duration} 秒。")
                    break
            except KeyboardInterrupt:
                print("\n[自动录音] 收到中断信号，停止录音。")
                should_stop = True
                break
    except KeyboardInterrupt:
        should_stop = True
        raise
    except Exception as exc:
        raise RuntimeError(f"自动录音失败: {exc}") from exc
    finally:
        # 恢复原始信号处理
        signal.signal(signal.SIGINT, original_handler)
        _safe_close_stream(stream)
        _safe_terminate_audio(audio)

    if not frames:
        raise RuntimeError("自动模式未捕获到音频数据")

    _persist_recording(output_path, frames, sample_width)

    return output_path


def _persist_recording(output_path: Path, frames, sample_width: int) -> None:
    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    file_size = output_path.stat().st_size
    duration = len(frames) * CHUNK / SAMPLE_RATE
    print(f"✓ 录音已保存: {output_path.resolve()}")
    print(f"  文件大小: {file_size} 字节 ({file_size / 1024:.2f} KB)")
    print(f"  录音时长: {duration:.2f} 秒")


def _safe_close_stream(stream) -> None:
    try:
        if stream:
            stream.stop_stream()
            stream.close()
    except Exception:
        pass


def _safe_terminate_audio(audio) -> None:
    try:
        if audio:
            audio.terminate()
    except Exception:
        pass

