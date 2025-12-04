"""Command line entrypoint for demo voice translation workflows."""

from __future__ import annotations

import argparse
import io
import shutil
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Voicemeeter 设备配置（Windows）
try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import sounddevice as sd  # type: ignore[import-not-found]
except ImportError:
    sd = None  # type: ignore[assignment]


try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv()  # 加载 .env 文件
except ImportError:
    pass

# Voicemeeter 设备配置（Windows）
# 可以根据实际设备索引修改这些值
VOICEMEETER_INPUT_INDEX = 3   # 输入：从 VoiceMeeter Output 读取原始音频（作为输入设备）
CABLE_OUTPUT_INDEX = 13        # 输出：向 VB-Audio Point Input 写入处理后的音频（作为输出设备）

# 扬声器翻译设备配置（在线模式）
# 可以根据实际设备索引修改这些值
SPEAKER_CAPTURE_INDEX = 31     # 输入：用于捕获扬声器输出的输入设备（在线模式）
SPEAKER_OUTPUT_INDEX = 44      # 输出：用于播放翻译后音频的输出设备（在线模式）


# 支持直接运行和作为模块运行
try:
    from .audio_checker import (
        detect_microphone_activity, 
        list_input_devices,
        list_all_devices,
        get_default_input_device,
        find_virtual_audio_input_device,
        find_blackhole_input_device,  # 兼容性导入
        find_speaker_output_device,
    )
    from .voice_recoder import record_audio
    from .local_translator import StreamingLocalTranslator, translate_local_voice
    from .online_translator import StreamingOnlineTranslator, translate_online_voice, play_audio_bytes
except ImportError:
    # 直接运行时使用绝对导入
    from audio_checker import (
        detect_microphone_activity, 
        list_input_devices, 
        list_all_devices,
        get_default_input_device,
        find_virtual_audio_input_device,
        find_blackhole_input_device,  # 兼容性导入
        find_speaker_output_device,
    )
    from voice_recoder import record_audio
    from local_translator import StreamingLocalTranslator, translate_local_voice
    from online_translator import StreamingOnlineTranslator, translate_online_voice, play_audio_bytes

AudioInput = Union[str, Path, bytes]

ORIGIN_AUDIO_DIR = Path("origin_audio")
TRANSLATED_AUDIO_DIR = Path("translated_audio")
ORIGIN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _materialise_output(data: AudioInput, destination: Optional[str], input_path: Optional[Path] = None) -> Optional[Path]:
    translated_dir = TRANSLATED_AUDIO_DIR
    translated_dir.mkdir(parents=True, exist_ok=True)

    if destination:
        # 如果提供了目标路径，检查是否为绝对路径
        dest_path = Path(destination)
        if dest_path.is_absolute():
            # 绝对路径，直接使用
            target = dest_path
        else:
            # 相对路径，保存到 translated_audio 文件夹
            target = translated_dir / dest_path.name
    elif isinstance(data, bytes):
        if input_path:
            stem = input_path.stem
            target = translated_dir / f"{stem}_translated.wav"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = translated_dir / f"translated_{timestamp}.wav"
    else:
        source_path = Path(data) if isinstance(data, (str, Path)) else None
        if source_path:
            # 确保保存到 translated_audio 文件夹，而不是使用源文件名直接保存
            stem = source_path.stem
            target = translated_dir / f"{stem}_translated.wav"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = translated_dir / f"translated_{timestamp}.wav"

    # 确保目标目录存在
    target.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, bytes):
        # 检查音频格式（通过文件头判断）
        # WAV 文件以 "RIFF" 开头，MP3 以 "ID3" 或 0xFF 0xFB 开头
        is_wav = data[:4] == b'RIFF'
        is_mp3 = data[:3] == b'ID3' or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)
        
        # 根据实际格式调整文件扩展名
        if not is_wav and (is_mp3 or not target.suffix):
            # 如果是 MP3 或其他格式，更改扩展名
            target = target.with_suffix('.mp3')
        
        target.write_bytes(data)
        print(f"✓ 翻译后的音频已保存: {target.resolve()}")
        print(f"  文件大小: {len(data)} 字节 ({len(data) / 1024:.2f} KB)")
        print(f"  格式: {'WAV' if is_wav else 'MP3' if is_mp3 else '未知'}")
        return target

    if isinstance(data, (Path, str)):
        source_path = Path(data)
        if source_path.exists() and source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target

    return None


def _persist_origin_audio(voice_input: AudioInput) -> Optional[Path]:
    origin_dir = ORIGIN_AUDIO_DIR
    origin_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(voice_input, bytes):
        target = origin_dir / f"origin_{timestamp}.wav"
        target.write_bytes(voice_input)
        print(f"✓ 原始音频已保存: {target.resolve()}")
        return target

    if isinstance(voice_input, (Path, str)):
        source_path = Path(voice_input)
        if not source_path.exists():
            return None
        
        # 确保保存到 origin_audio 文件夹
        # 如果源文件已经在 origin_audio 文件夹中，直接返回
        if source_path.parent.resolve() == origin_dir.resolve():
            print(f"✓ 原始音频已在目标文件夹: {source_path.resolve()}")
            return source_path
        
        # 否则，复制到 origin_audio 文件夹
        if source_path.suffix:
            # 使用带时间戳的文件名，避免覆盖
            stem = source_path.stem
            target = origin_dir / f"{stem}_{timestamp}.wav"
        else:
            target = origin_dir / f"origin_{timestamp}.wav"

        shutil.copy2(source_path, target)
        print(f"✓ 原始音频已导出: {target.resolve()}")
        return target

    return None


def _parse_playback_device(device_spec: Optional[str]):
    if not device_spec:
        return None
    device_spec = device_spec.strip()
    if not device_spec:
        return None
    try:
        return int(device_spec)
    except ValueError:
        return device_spec


def _find_virtual_audio_device() -> Optional[int]:
    """查找虚拟音频设备（Windows: VB-CABLE Input 等，macOS: BlackHole）
    只返回有输出通道的设备（用于输出音频）
    """
    if sd is None:
        return None
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            name_lower = dev['name'].lower()
            max_output_channels = dev.get('max_output_channels', 0)
            
            # 只选择有输出通道的设备
            if max_output_channels == 0:
                continue
            
            # Windows 虚拟音频设备 - 优先选择 VB-Audio Point Input 或 CABLE Input（有输出通道）
            if ("cable" in name_lower and "input" in name_lower) or \
               ("vb-cable" in name_lower and "input" in name_lower):
                return i
            # 也支持其他 VB-CABLE 变体（如果有输出通道）
            if ("vb-cable" in name_lower or "vb cable" in name_lower or "virtual cable" in name_lower):
                if max_output_channels > 0:
                    return i
            # macOS 虚拟音频设备
            if "blackhole" in name_lower and max_output_channels > 0:
                return i
    except Exception:
        pass
    return None


def _find_blackhole_device() -> Optional[int]:
    """查找 BlackHole 设备（兼容性函数，推荐使用 _find_virtual_audio_device）"""
    return _find_virtual_audio_device()


def _test_input_device(device_index: int, duration: float = 5.0) -> None:
    """测试输入设备，显示音量信息（用于验证扬声器捕获配置）"""
    if sd is None or np is None:
        print("⚠️  需要 sounddevice 和 numpy 库")
        return
    
    try:
        device_info = sd.query_devices(device_index)
        print("=" * 70)
        print(f"测试输入设备: [{device_index}] {device_info['name']}")
        print(f"输入通道: {device_info.get('max_input_channels', 0)}")
        print(f"输出通道: {device_info.get('max_output_channels', 0)}")
        print(f"采样率: {device_info.get('default_samplerate', 'N/A')} Hz")
        print("=" * 70)
        print(f"\n正在监听 {duration} 秒...")
        print("请播放一些音频（例如：QQ 语音、音乐等）来测试设备是否正常工作")
        print("如果看到音量变化，说明设备配置正确！")
        print("-" * 70)
        print()
        
        max_volume = 0.0
        sample_count = 0
        
        def audio_callback(indata, frames, time_info, status):
            nonlocal max_volume, sample_count
            if status:
                if status.input_overflow:
                    print(f"\n⚠️  输入溢出")
                elif status.input_underflow:
                    print(f"\n⚠️  输入欠载")
            
            if np is not None:
                volume = np.linalg.norm(indata) * 10
                max_volume = max(max_volume, volume)
                sample_count += 1
                
                # 显示音量条
                volume_bar_length = 40
                volume_bar = "█" * int(min(volume * 2, volume_bar_length))
                volume_percent = min(volume * 100, 100)
                
                # 每 10 个样本更新一次显示（减少闪烁）
                if sample_count % 10 == 0:
                    print(f"音量: {volume_percent:5.1f}% |{volume_bar:<{volume_bar_length}}| (最大: {max_volume*100:.1f}%)", end='\r')
        
        try:
            stream = sd.InputStream(
                device=device_index,
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=3200,
            )
            stream.start()
            
            import time
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(0.1)
            
            stream.stop()
            stream.close()
            
            print("\n" + "=" * 70)
            if max_volume > 0.1:
                print(f"✅ 测试成功！检测到音频信号")
                print(f"   最大音量: {max_volume*100:.1f}%")
                print(f"   设备配置正确，可以捕获扬声器输出")
            else:
                print(f"⚠️  测试完成，但未检测到明显的音频信号")
                print(f"   最大音量: {max_volume*100:.1f}%")
                print(f"   请检查：")
                print(f"   1. 系统输出是否设置为包含 BlackHole 的 Multi-Output Device")
                print(f"   2. 是否有音频正在播放")
                print(f"   3. 音量是否足够大")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ 无法查询设备 {device_index}: {e}")
        print(f"   请使用 --list-all-devices 查看可用设备列表")


def _play_audio_bytes(audio_bytes: bytes, *, device_spec: Optional[str]) -> None:
    if sd is None or np is None:
        print("[播放跳过] sounddevice 或 numpy 未安装。")
        return
    if not audio_bytes:
        return

    target_device = _parse_playback_device(device_spec)
    buffer = io.BytesIO(audio_bytes)

    try:
        with wave.open(buffer, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()

            dtype_map = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}
            dtype = dtype_map.get(sample_width)
            if dtype is None:
                raise ValueError(f"不支持的采样位宽: {sample_width}")

            audio_array = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                audio_array = audio_array.reshape((-1, channels))

            try:
                sd.play(audio_array, sample_rate, device=target_device)
                # 不等待播放完成，避免阻塞
                # sd.wait()  # 注释掉 wait，让音频在后台播放
            except KeyboardInterrupt:
                print("[播放] 播放被中断")
                return
            return
    except (wave.Error, ValueError) as exc:
        print(f"[播放提示] WAV 解析失败，尝试按 PCM16 播放: {exc}")

    try:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        try:
            sd.play(audio_array, 16000, device=target_device)
            # 不等待播放完成，避免阻塞
            # sd.wait()  # 注释掉 wait，让音频在后台播放
        except KeyboardInterrupt:
            print("[播放] 播放被中断")
    except Exception as exc:
        print(f"[播放失败] 无法播放音频: {exc}")


def _output_to_virtual_microphone(audio_bytes: bytes, output_device: Optional[int | str] = None) -> bool:
    """
    将翻译后的音频输出到虚拟麦克风（Windows: VB-Audio Point Input 等，macOS: BlackHole）
    这样上层应用（如 QQ 语音）就能从对应的虚拟音频输出设备接收到翻译后的音频
    
    注意：程序向 VB-Audio Point Input（输出设备，索引68）写入音频，QQ/游戏从对应的虚拟音频输出设备（输入设备）读取音频
    """
    if sd is None or np is None:
        print("[虚拟麦克风输出跳过] sounddevice 或 numpy 未安装。")
        return False
    
    if not audio_bytes:
        return False
    
    # 如果没有指定设备，使用配置的 CABLE_OUTPUT_INDEX（VB-Audio Point Input）
    if output_device is None:
        try:
            if sd is not None:
                device_info = sd.query_devices(CABLE_OUTPUT_INDEX)
                if device_info.get('max_output_channels', 0) > 0:
                    output_device = CABLE_OUTPUT_INDEX
                    print(f"[虚拟麦克风] 使用配置的 VB-Audio Point Input 设备: {device_info['name']} (索引: {CABLE_OUTPUT_INDEX})")
                else:
                    print(f"[虚拟麦克风] 警告: 设备 {CABLE_OUTPUT_INDEX} 没有输出通道，尝试自动查找...")
                    output_device = _find_virtual_audio_device()
        except Exception as e:
            print(f"[虚拟麦克风] 配置的设备 {CABLE_OUTPUT_INDEX} 不可用: {e}，尝试自动查找...")
            output_device = _find_virtual_audio_device()
        
        if output_device is None:
            import sys
            if sys.platform == 'win32':
                print("[虚拟麦克风输出跳过] 未找到虚拟音频设备（如 VB-Audio Point Input），请安装 VB-Audio 或使用 --virtual-mic-device 指定设备")
            else:
                print("[虚拟麦克风输出跳过] 未找到 BlackHole 设备，请安装 BlackHole 或使用 --virtual-mic-device 指定设备")
            return False
        
        if output_device != CABLE_OUTPUT_INDEX:
            device_name = sd.query_devices(output_device)['name']
            print(f"[虚拟麦克风] 使用自动查找的设备: {device_name} (索引: {output_device})")
    
    try:
        # 尝试使用 soundfile 读取音频（支持多种格式）
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
            
            if audio_data.ndim == 1:
                audio_float = audio_data.reshape(-1, 1).astype(np.float32)
            else:
                audio_float = audio_data.astype(np.float32)
        except ImportError:
            # 如果没有 soundfile，使用 wave（仅支持 WAV）
            buffer = io.BytesIO(audio_bytes)
            try:
                with wave.open(buffer, "rb") as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    frames = wav_file.readframes(wav_file.getnframes())
                    
                    audio_array = np.frombuffer(frames, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32767.0
                    
                    if channels > 1:
                        audio_float = audio_float.reshape(-1, channels)
                    else:
                        audio_float = audio_float.reshape(-1, 1)
            except Exception as e:
                print(f"[虚拟麦克风输出失败] 无法读取音频: {e}")
                return False
        
        # 确保音频值在有效范围内
        audio_float = np.clip(audio_float, -1.0, 1.0)
        
        # 查询设备信息，确定正确的声道数
        try:
            device_info = sd.query_devices(output_device)
            max_output_channels = device_info.get('max_output_channels', 0)
            
            # 确定目标声道数：优先使用设备支持的最大声道数（通常是2）
            if max_output_channels >= 2:
                target_channels = 2  # 强制使用立体声
            elif max_output_channels == 1:
                target_channels = 1  # 单声道
            else:
                target_channels = 2  # 默认使用立体声
                print(f"[虚拟麦克风] 警告: 无法确定设备声道数，默认使用2声道")
            
            # 调整声道数
            if channels != target_channels:
                if target_channels == 2 and channels == 1:
                    # 单声道转立体声：复制到两个声道
                    audio_float = np.repeat(audio_float, 2, axis=1)
                    print(f"[虚拟麦克风] 将音频从单声道转换为立体声")
                elif target_channels == 1 and channels == 2:
                    # 立体声转单声道：取平均值
                    audio_float = np.mean(audio_float, axis=1, keepdims=True)
                    print(f"[虚拟麦克风] 将音频从立体声转换为单声道")
                channels = target_channels
        except Exception as e:
            print(f"[虚拟麦克风] 查询设备信息失败: {e}，使用原始声道数 {channels}")
        
        # 输出到虚拟麦克风（向 VB-Audio Point Input 写入）
        device_name = sd.query_devices(output_device).get('name', f'设备 {output_device}')
        print(f"[虚拟麦克风] 开始输出音频到设备 {output_device} ({device_name})...")
        print(f"[虚拟麦克风] 音频参数: {sample_rate} Hz, {channels} 声道")
        try:
            with sd.OutputStream(
                device=output_device,
                samplerate=sample_rate,
                channels=channels,
                dtype='float32',
            ) as stream:
                stream.write(audio_float)
                # 不等待，立即返回，避免阻塞
                # stream.wait()  # 注释掉 wait，让音频在后台播放
        except KeyboardInterrupt:
            # 如果收到中断信号，立即返回
            print("[虚拟麦克风] 输出被中断")
            return False
        except Exception as e:
            print(f"[虚拟麦克风] 输出失败: {e}")
            print(f"[虚拟麦克风] 尝试参数: 设备={output_device}, 采样率={sample_rate}, 声道数={channels}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"[虚拟麦克风] ✓ 音频已输出到虚拟麦克风")
        return True
        
    except Exception as e:
        print(f"[虚拟麦克风输出失败] {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice translation workflow demo.")
    # 注意: --mode 参数已废弃，现在统一使用 voice_translator 函数
    parser.add_argument("--mode", choices={"local", "online"}, default="local", 
                       help="翻译模式: local=本地语音翻译后发送给对方, online=对方语音翻译后在本地说/播放")
    parser.add_argument("--voice", help="可选音频文件路径，用于一次性处理（而非持续监听模式）")
    parser.add_argument("--local-lang", default="chinese", help="本地语言（你使用的语言）")
    parser.add_argument("--target-lang", default="english", help="目标语言（本地模式：翻译成此语言发送给对方）")
    parser.add_argument("--source-lang", help="源语言（在线模式：对方语音的语言，默认自动检测）")
    parser.add_argument("--output", help="翻译后音频的保存路径（默认保存到 translated_audio/）")
    parser.add_argument("--language-hint", help="可选的语言提示")
    parser.add_argument("--max-duration", type=float, default=30.0, help="每次录音的最大时长（秒）")
    parser.add_argument("--activity-threshold", type=float, default=1.2, help="检测麦克风活动的音量阈值")
    parser.add_argument("--activity-window", type=float, default=0.5, help="每次活动检测的时间窗口（秒）")
    parser.add_argument("--silence-threshold", type=int, default=800, help="自动停止的静音阈值（RMS）")
    parser.add_argument("--silence-duration", type=float, default=1.2, help="停止录音前的静音持续时间（秒）")
    parser.add_argument("--idle-sleep", type=float, default=0.2, help="检测间隔的休眠时间（秒）")
    parser.add_argument("--virtual-mic", action="store_true", 
                       help="（本地模式）将翻译后的音频输出到虚拟麦克风（Windows: VB-CABLE，macOS: BlackHole），供 QQ 语音等应用使用")
    parser.add_argument("--virtual-mic-device", 
                       help="虚拟麦克风设备索引或名称（默认自动查找虚拟音频设备）")
    parser.add_argument("--input-device", help="输入设备索引或名称（用于麦克风）")
    parser.add_argument("--output-device", help="输出设备索引或名称（用于扬声器播放，仅在线模式）")
    parser.add_argument("--streaming", action="store_true",
                       help="使用流式翻译模式：持续监听麦克风，实时翻译（不阻塞）")
    parser.add_argument("--list-devices", action="store_true", help="列出所有可用的音频输入设备并退出")
    parser.add_argument("--list-all-devices", action="store_true", help="列出所有音频设备（输入和输出）并退出")
    parser.add_argument("--test-input-device", type=int, help="测试指定的输入设备，监听 5 秒音频（用于验证扬声器捕获）")

    args = parser.parse_args()

    # 如果用户要求列出设备，则列出并退出
    if args.list_all_devices:
        list_all_devices()
        return
    
    if args.list_devices:
        list_input_devices()
        return
    
    # 如果用户要求测试输入设备
    if args.test_input_device is not None:
        _test_input_device(args.test_input_device)
        return

    # 如果提供了音频文件，进行一次性处理
    if args.voice:
        voice_input: AudioInput = Path(args.voice)
        if args.mode == "online":
            # 在线模式：翻译对方语音并在本地说/播放（一次性处理）
            audio_bytes = translate_online_voice(
                voice_input,
                local_lang=args.local_lang,
                source_language=getattr(args, "source_lang", None),
            )
            if audio_bytes:
                play_audio_bytes(audio_bytes, output_device=getattr(args, "output_device", None))
        else:
            # 本地模式：翻译本地语音并保存/输出到虚拟麦克风
            _run_local_translation_pipeline(voice_input, args)
        return

    # 解析输入设备
    input_device = None
    if args.input_device is not None:
        try:
            input_device = int(args.input_device)
        except (ValueError, TypeError):
            input_device = args.input_device  # 保持为字符串（设备名称）
    
    # 如果启用了流式翻译模式
    if getattr(args, "streaming", False):
        if args.mode == "online":
            # 在线模式的流式翻译：监听音频输入，翻译后播放
            # 优先使用虚拟音频设备捕获扬声器输出
            speaker_capture_device = input_device
            use_virtual = False
            
            # 如果用户没有指定输入设备，优先使用配置的 SPEAKER_CAPTURE_INDEX
            if speaker_capture_device is None:
                try:
                    if sd is not None:
                        device_info = sd.query_devices(SPEAKER_CAPTURE_INDEX)
                        if device_info.get('max_input_channels', 0) > 0:
                            speaker_capture_device = SPEAKER_CAPTURE_INDEX
                            print(f"[音频捕获] 使用配置的扬声器捕获设备索引: {SPEAKER_CAPTURE_INDEX} ({device_info['name']})")
                except Exception as e:
                    print(f"[音频捕获] 配置的扬声器捕获设备索引 {SPEAKER_CAPTURE_INDEX} 不可用: {e}")
            
            if speaker_capture_device is None:
                # 自动查找虚拟音频设备用于捕获扬声器输出
                virtual_device = find_virtual_audio_input_device()
                if virtual_device is not None:
                    speaker_capture_device = virtual_device
                    use_virtual = True
                    import sys
                    if sd is not None:
                        device_name = sd.query_devices(virtual_device)['name']
                    else:
                        device_name = f"设备 {virtual_device}"
                    if sys.platform == 'win32':
                        print(f"[音频捕获] ✓ 自动检测到虚拟音频设备（{device_name}），将用于捕获扬声器输出")
                    else:
                        print(f"[音频捕获] ✓ 自动检测到 BlackHole 设备，将用于捕获扬声器输出")
                else:
                    # 如果没有虚拟音频设备，使用默认输入设备（通常是麦克风）
                    speaker_capture_device = get_default_input_device(avoid_virtual=True)
                    if speaker_capture_device is not None:
                        try:
                            device_info = sd.query_devices(speaker_capture_device)
                            device_name = device_info['name']
                            use_virtual = False
                            print(f"[音频捕获] 使用默认输入设备: {device_name} (索引: {speaker_capture_device})")
                            print(f"  💡 当前使用 {device_name}（通常是麦克风）")
                            import sys
                            if sys.platform == 'win32':
                                print(f"  💡 注意：要捕获扬声器输出，请使用 VB-CABLE 虚拟音频设备")
                                print(f"  💡 要捕获扬声器输出：")
                                print(f"     1. 安装 VB-Audio Virtual Cable: https://vb-audio.com/Cable/")
                                print(f"     2. 在 Windows 声音设置中将 VB-CABLE 设置为默认播放设备")
                                print(f"     3. 使用 --input-device <VB-CABLE索引> 指定虚拟音频设备")
                            else:
                                print(f"  💡 注意：macOS 无法直接捕获扬声器输出，必须使用 BlackHole + Multi-Output Device")
                                print(f"  💡 要捕获扬声器输出：")
                                print(f"     1. 安装 BlackHole 并配置 Multi-Output Device")
                                print(f"     2. 使用 --input-device <BlackHole索引> 指定 BlackHole 设备")
                        except Exception:
                            use_virtual = False
                    else:
                        print("⚠️  警告: 未找到可用的输入设备")
                        print("  请确保：")
                        import sys
                        if sys.platform == 'win32':
                            print("  1. 已连接麦克风，或")
                            print("  2. 已安装 VB-CABLE 虚拟音频设备（用于捕获扬声器输出）")
                        else:
                            print("  1. 已连接麦克风，或")
                            print("  2. 已安装 BlackHole 并配置 Multi-Output Device（用于捕获扬声器输出）")
                        print("  3. 运行 --list-all-devices 查看所有设备")
                        print("  4. 使用 --input-device <索引> 手动指定设备")
                        speaker_capture_device = None
                        use_virtual = False
            
            # 解析输出设备（用于播放翻译后的音频）
            output_device = getattr(args, "output_device", None)
            if output_device is not None:
                try:
                    output_device = int(output_device)
                except (ValueError, TypeError):
                    output_device = output_device  # 保持为字符串（设备名称）
            else:
                # 如果用户没有指定输出设备，优先使用配置的 SPEAKER_OUTPUT_INDEX
                try:
                    if sd is not None:
                        device_info = sd.query_devices(SPEAKER_OUTPUT_INDEX)
                        if device_info.get('max_output_channels', 0) > 0:
                            output_device = SPEAKER_OUTPUT_INDEX
                            print(f"[音频播放] 使用配置的扬声器输出设备索引: {SPEAKER_OUTPUT_INDEX} ({device_info['name']})")
                except Exception as e:
                    print(f"[音频播放] 配置的扬声器输出设备索引 {SPEAKER_OUTPUT_INDEX} 不可用: {e}")
                
                # 如果配置的设备不可用，自动查找扬声器设备作为输出
                if output_device is None:
                    speaker_output = find_speaker_output_device()
                    if speaker_output is not None:
                        output_device = speaker_output
                        try:
                            device_info = sd.query_devices(output_device)
                            print(f"[音频播放] ✓ 自动检测到扬声器设备: {device_info['name']} (索引: {output_device})")
                        except Exception:
                            pass
            
            # 显示配置提示
            if speaker_capture_device is None:
                import sys
                print("\n" + "=" * 70)
                print("⚠️  未找到可用的音频输入设备")
                print("=" * 70)
                print("\n请确保：")
                print("  1. 已连接麦克风，或")
                if sys.platform == 'win32':
                    print("  2. 已安装 VB-CABLE 虚拟音频设备（用于捕获扬声器输出）")
                    print()
                    print("要捕获扬声器输出，请按以下步骤配置：")
                    print()
                    print("📋 步骤 1: 安装 VB-Audio Virtual Cable")
                    print("   下载地址: https://vb-audio.com/Cable/")
                    print()
                    print("📋 步骤 2: 设置 Windows 声音输出")
                    print("   1. 打开 Windows 设置 → 系统 → 声音")
                    print("   2. 将 VB-CABLE 设置为默认播放设备")
                    print()
                    print("📋 步骤 3: 运行程序")
                    print("   运行: --list-all-devices 查看设备列表")
                    print("   使用: --input-device <VB-CABLE索引>")
                else:
                    print("  2. 已安装 BlackHole 并配置 Multi-Output Device（用于捕获扬声器输出）")
                    print()
                    print("要捕获扬声器输出，请按以下步骤配置：")
                    print()
                    print("📋 步骤 1: 安装 BlackHole")
                    print("   下载地址: https://github.com/ExistentialAudio/BlackHole")
                    print()
                    print("📋 步骤 2: 配置 Multi-Output Device")
                    print("   1. 打开「音频 MIDI 设置」: open -a 'Audio MIDI Setup'")
                    print("   2. 创建 Multi-Output Device：")
                    print("      - 点击左下角「+」按钮，选择「创建多路输出设备」")
                    print("      - 勾选您的扬声器 + BlackHole 2ch")
                    print("      - 将主时钟源设置为您的扬声器")
                    print()
                    print("📋 步骤 3: 设置系统输出")
                    print("   系统设置 → 声音 → 输出 → 选择 Multi-Output Device")
                    print()
                    print("📋 步骤 4: 运行程序")
                    print("   运行: --list-all-devices 查看设备列表")
                    print("   使用: --input-device <BlackHole索引>")
                print("=" * 70)
                print()
                return  # 如果找不到设备，直接返回
            else:
                import sys
                print("\n" + "=" * 70)
                print("✅ 扬声器音频翻译服务")
                print("=" * 70)
                if use_virtual:
                    try:
                        device_info = sd.query_devices(speaker_capture_device)
                        print(f"✓ 输入设备（捕获）: {device_info['name']} (索引: {speaker_capture_device})")
                    except Exception:
                        print(f"✓ 输入设备（捕获）: BlackHole (索引: {speaker_capture_device})")
                    
                    if output_device is not None:
                        try:
                            device_info = sd.query_devices(output_device)
                            print(f"✓ 输出设备（播放）: {device_info['name']} (索引: {output_device})")
                        except Exception:
                            print(f"✓ 输出设备（播放）: 扬声器 (索引: {output_device})")
                    
                    print()
                    print("📌 工作流程：")
                    if sd is not None:
                        device_name = sd.query_devices(speaker_capture_device)['name']
                    else:
                        device_name = f"设备 {speaker_capture_device}"
                    if sys.platform == 'win32':
                        print(f"   1. 从虚拟音频设备（{device_name}）捕获扬声器播放的音频")
                    else:
                        print(f"   1. 从 BlackHole 捕获扬声器播放的音频")
                    print("   2. 实时识别并翻译音频内容")
                    print("   3. 将翻译后的音频播放到扬声器")
                    print()
                    print("⚠️  重要提示：")
                    if sys.platform == 'win32':
                        print("   请确保已将 VB-CABLE 设置为 Windows 默认播放设备")
                        print()
                        print("   如果未配置，请按以下步骤：")
                        print("   1. 安装 VB-Audio Virtual Cable: https://vb-audio.com/Cable/")
                        print("   2. Windows 设置 → 系统 → 声音 → 将 VB-CABLE 设置为默认播放设备")
                    else:
                        print("   请确保已配置 Multi-Output Device（包含扬声器 + BlackHole）")
                        print("   并将系统输出设置为该 Multi-Output Device")
                        print()
                        print("   如果未配置，请按以下步骤：")
                        print("   1. 打开「音频 MIDI 设置」: open -a 'Audio MIDI Setup'")
                        print("   2. 创建 Multi-Output Device（包含扬声器 + BlackHole 2ch）")
                        print("   3. 系统设置 → 声音 → 输出 → 选择 Multi-Output Device")
                    print()
                else:
                    print("✓ 使用默认输入设备")
                    print()
                    print("⚠️  重要说明：")
                    import sys
                    if sys.platform == 'win32':
                        print("   要捕获扬声器输出（QQ 语音等播放的音频），请使用 VB-CABLE 虚拟音频设备：")
                        print()
                        print("   1. 安装 VB-Audio Virtual Cable: https://vb-audio.com/Cable/")
                        print("   2. Windows 设置 → 系统 → 声音 → 将 VB-CABLE 设置为默认播放设备")
                        print("   3. 程序会自动检测并使用 VB-CABLE 设备")
                        print()
                        print("   这样配置后：")
                        print("   - 音频会同时发送到 VB-CABLE（程序能捕获）")
                    else:
                        print("   macOS 无法直接捕获扬声器输出（系统播放的音频）")
                        print("   当前使用的是输入设备（通常是麦克风），不是扬声器输出")
                        print()
                        print("💡 如果要捕获扬声器输出（QQ 语音等播放的音频）：")
                        print("   必须使用 BlackHole + Multi-Output Device 配置：")
                        print()
                        print("   1. 安装 BlackHole: https://github.com/ExistentialAudio/BlackHole")
                        print("   2. 创建 Multi-Output Device（包含扬声器 + BlackHole）")
                        print("   3. 将系统输出设置为该 Multi-Output Device")
                        print("   4. 程序会自动检测并使用 BlackHole 设备")
                        print()
                        print("   这样配置后：")
                        print("   - 音频会同时发送到扬声器（您能听到）")
                        print("   - 音频会同时发送到 BlackHole（程序能捕获）")
                    print()
                    print("💡 如果只想使用麦克风：")
                    print("   - 当前配置已可以使用，程序会捕获麦克风输入")
                    print()
                print("=" * 70)
                print()
            
            try:
                # 如果 source_lang 未指定或为 None，使用默认值 "english"
                source_lang = getattr(args, "source_lang", None)
                if source_lang is None:
                    source_lang = "english"
                
                translator = StreamingOnlineTranslator(
                    source_language=source_lang,
                    local_lang=args.local_lang,
                    input_device=speaker_capture_device,  # 扬声器输出捕获设备（虚拟音频设备）
                    output_device=output_device,  # 播放设备（扬声器）
                )
                translator.start()
            except KeyboardInterrupt:
                print("\n[退出] 收到中断信号...")
            except Exception as e:
                print(f"\n[错误] {e}")
                import traceback
                traceback.print_exc()
            return
        
        # 本地模式的流式翻译
        # 解析虚拟麦克风设备
        virtual_mic_device = None
        if getattr(args, "virtual_mic", False):
            virtual_mic_device = getattr(args, "virtual_mic_device", None)
            if virtual_mic_device is None:
                # 优先使用配置的虚拟音频设备索引
                try:
                    if sd is not None:
                        device_info = sd.query_devices(CABLE_OUTPUT_INDEX)
                        if device_info.get('max_output_channels', 0) > 0:
                            virtual_mic_device = CABLE_OUTPUT_INDEX
                            print(f"[虚拟麦克风] 使用配置的虚拟音频设备索引: {CABLE_OUTPUT_INDEX} ({device_info['name']})")
                except Exception as e:
                    print(f"[虚拟麦克风] 配置的虚拟音频设备索引 {CABLE_OUTPUT_INDEX} 不可用: {e}")
                # 如果配置的设备不可用，尝试自动查找（只查找有输出通道的设备）
                if virtual_mic_device is None:
                    print(f"[虚拟麦克风] 配置的设备 {CABLE_OUTPUT_INDEX} 不可用，尝试自动查找输出设备...")
                    try:
                        if sd is not None:
                            devices = sd.query_devices()
                            # 优先查找 VB-Audio Point Input
                            for i, dev in enumerate(devices):
                                name_lower = dev['name'].lower()
                                if "vb-audio point" in name_lower and "input" in name_lower:
                                    if dev.get('max_output_channels', 0) > 0:
                                        virtual_mic_device = i
                                        print(f"[虚拟麦克风] 找到输出设备: {dev['name']} (索引: {i})")
                                        break
                            # 如果没找到，查找其他 CABLE Input 设备
                            if virtual_mic_device is None:
                                for i, dev in enumerate(devices):
                                    name_lower = dev['name'].lower()
                                    if ("cable" in name_lower and "input" in name_lower) or \
                                       ("vb-cable" in name_lower and "input" in name_lower):
                                        if dev.get('max_output_channels', 0) > 0:
                                            virtual_mic_device = i
                                            print(f"[虚拟麦克风] 找到输出设备: {dev['name']} (索引: {i})")
                                            break
                        else:
                            virtual_mic_device = _find_virtual_audio_device()
                    except Exception as e:
                        print(f"[虚拟麦克风] 自动查找失败: {e}")
                        if sd is not None:
                            virtual_mic_device = _find_virtual_audio_device()
                        else:
                            virtual_mic_device = None
            
            # 验证设备是否有输出通道
            if virtual_mic_device is not None:
                # virtual_mic_device 已经是 int 类型（通过 type=int）或自动查找的结果
                if not isinstance(virtual_mic_device, int):
                    try:
                        virtual_mic_device = int(virtual_mic_device)
                    except (ValueError, TypeError):
                        pass
                # 验证设备
                try:
                    device_info = sd.query_devices(virtual_mic_device)
                    max_output_channels = device_info.get('max_output_channels', 0)
                    max_input_channels = device_info.get('max_input_channels', 0)
                    
                    if max_output_channels == 0:
                        print(f"[虚拟麦克风] 错误: 设备 {virtual_mic_device} ({device_info['name']}) 没有输出通道！")
                        print(f"[虚拟麦克风] 这是一个输入设备（max_input_channels={max_input_channels}），不能用于输出音频")
                        print(f"[虚拟麦克风] 请使用 VB-Audio Point Input（索引 68）作为输出设备")
                        virtual_mic_device = None  # 清除无效设备
                    else:
                        print(f"[虚拟麦克风] 使用设备: {device_info['name']} (索引: {virtual_mic_device})")
                        print(f"  输出通道数: {max_output_channels}")
                        print(f"  输入通道数: {max_input_channels}")
                        print(f"\n📌 重要配置提示：")
                        print(f"  1. 程序向 {device_info['name']} 的输出通道写入音频")
                        print(f"  2. QQ/游戏需要选择对应的虚拟音频输出设备作为麦克风输入")
                        print(f"  3. 确保系统设置 → 声音 → 输入设备设置为对应的虚拟音频输出设备")
                        print()
                except Exception as e:
                    print(f"[虚拟麦克风] 无法查询设备 {virtual_mic_device}: {e}")
                    virtual_mic_device = None  # 清除无效设备
        
        # 如果用户没有通过命令行指定输入设备，默认强制使用配置的 VOICEMEETER_INPUT_INDEX
        if input_device is None:
            input_device = VOICEMEETER_INPUT_INDEX
            print(f"[输入设备] 未指定输入设备，强制使用配置的索引: {VOICEMEETER_INPUT_INDEX}")
        
        # 验证输入设备
        if input_device is not None:
            try:
                device_info = sd.query_devices(input_device)
                print(f"[输入设备] 使用设备: {device_info['name']} (索引: {input_device})")
                if device_info['max_input_channels'] == 0:
                    if input_device == VOICEMEETER_INPUT_INDEX:
                        print("⚠️  提示: 该设备报告没有输入通道，但根据配置强制使用 VOICEMEETER_INPUT_INDEX")
                    else:
                        print(f"⚠️  警告: 设备 {input_device} 没有输入通道，尝试使用系统默认设备")
                        input_device = None
            except Exception as e:
                if input_device == VOICEMEETER_INPUT_INDEX:
                    print(f"⚠️  提示: 无法查询配置的 VOICEMEETER_INPUT_INDEX ({e})，仍尝试使用该索引")
                else:
                    print(f"⚠️  警告: 无法查询输入设备 {input_device}: {e}")
                    input_device = None
        
        # 如果没有指定设备，优先使用配置的 Voicemeeter 设备索引
        if input_device is None:
            try:
                if sd is not None:
                    device_info = sd.query_devices(VOICEMEETER_INPUT_INDEX)
                    if device_info.get('max_input_channels', 0) > 0:
                        input_device = VOICEMEETER_INPUT_INDEX
                        print(f"[输入设备] 使用配置的 Voicemeeter 设备索引: {VOICEMEETER_INPUT_INDEX} ({device_info['name']})")
            except Exception as e:
                print(f"[输入设备] 配置的 Voicemeeter 设备索引 {VOICEMEETER_INPUT_INDEX} 不可用: {e}")
            
            # 如果 Voicemeeter 设备不可用，尝试获取默认设备
            if input_device is None:
                input_device = get_default_input_device(avoid_virtual=True)
                if input_device is not None:
                    try:
                        device_info = sd.query_devices(input_device)
                        print(f"[输入设备] 使用默认设备: {device_info['name']} (索引: {input_device})")
                    except Exception:
                        pass
        
        # 创建并启动流式本地翻译器
        try:
            translator = StreamingLocalTranslator(
                source_language=args.local_lang,
                target_language=args.target_lang,
                input_device=input_device,
                virtual_mic_device=virtual_mic_device,
            )
            translator.start()
        except KeyboardInterrupt:
            print("\n[退出] 收到中断信号...")
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()
        return
    
    # 如果没有指定设备，尝试获取默认设备（避免虚拟音频设备，优先使用麦克风）
    if input_device is None:
        input_device = get_default_input_device(avoid_virtual=True)
    
    print("=" * 60)
    mode_desc = "本地模式（翻译本地语音并发送给对方）" if args.mode == "local" else "在线模式（翻译对方语音并在本地播放）"
    print(f"语音翻译服务已启动 - {mode_desc}")
    print("按 Ctrl+C 结束")
    print()
    
    if input_device is not None:
        try:
            device_info = sd.query_devices(input_device)
            print(f"使用输入设备: {device_info['name']} (索引: {input_device})")
        except Exception:
            print(f"使用输入设备: {input_device}")
    else:
        print("⚠️  警告: 未找到输入设备，将尝试使用系统默认设备")
        print("提示: 使用 --list-devices 查看可用设备，或使用 --input-device 指定设备")
    
    # 如果启用了虚拟麦克风输出（仅本地模式），显示提示信息
    if args.mode == "local" and getattr(args, "virtual_mic", False):
        import sys
        virtual_device = _find_virtual_audio_device()
        if virtual_device is not None:
            try:
                device_info = sd.query_devices(virtual_device)
                print(f"\n✓ 虚拟麦克风输出已启用")
                print(f"  输出设备: {device_info['name']} (索引: {virtual_device})")
                if sys.platform == 'win32':
                    print(f"\n📌 重要提示（让 QQ 语音使用虚拟音频设备）：")
                    print(f"  方法 1（推荐）：在 Windows 系统设置中设置")
                    print(f"    1. 打开「Windows 设置」→「系统」→「声音」→「输入」")
                    print(f"    2. 选择「{device_info['name']}」作为输入设备")
                    print(f"    3. QQ 语音会自动使用系统默认输入设备（{device_info['name']}）")
                    print(f"  方法 2：在 QQ 语音中设置（如果支持）")
                    print(f"    1. 打开 QQ 语音设置")
                    print(f"    2. 在音频设置中，将输入设备设置为「{device_info['name']}」")
                else:
                    print(f"\n📌 重要提示（让 QQ 语音使用 BlackHole）：")
                    print(f"  方法 1（推荐）：在 macOS 系统设置中设置")
                    print(f"    1. 打开「系统设置」→「声音」→「输入」")
                    print(f"    2. 选择「{device_info['name']}」作为输入设备")
                    print(f"    3. QQ 语音会自动使用系统默认输入设备（BlackHole）")
                    print(f"  方法 2：在 QQ 语音中设置（如果支持）")
                    print(f"    1. 打开 QQ 语音设置")
                    print(f"    2. 在音频设置中，将输入设备设置为「{device_info['name']}」")
                print(f"\n  翻译后的音频将自动输出到虚拟麦克风")
                print(f"  对方将听到翻译后的音频")
            except Exception:
                print(f"\n✓ 虚拟麦克风输出已启用 (设备索引: {virtual_device})")
                if sys.platform == 'win32':
                    print(f"\n📌 请在 Windows 系统设置中将虚拟音频设备设置为默认输入设备")
                else:
                    print(f"\n📌 请在 macOS 系统设置中将 BlackHole 设置为默认输入设备")
        else:
            import sys
            if sys.platform == 'win32':
                print(f"\n⚠️  警告: 未找到虚拟音频设备（如 VB-CABLE）")
                print(f"  请安装 VB-Audio Virtual Cable: https://vb-audio.com/Cable/")
            else:
                print(f"\n⚠️  警告: 未找到 BlackHole 设备")
                print(f"  请安装 BlackHole: https://github.com/ExistentialAudio/BlackHole")
            print(f"  或使用 --virtual-mic-device 指定虚拟麦克风设备")
    
    print("\n等待麦克风唤醒词或语音活动...")
    print("=" * 60)

    # 设置信号处理，确保能响应 Ctrl+C
    import signal
    import sys
    import os
    
    # 使用全局变量标记是否应该退出
    should_exit = False
    
    def signal_handler(sig, frame):
        global should_exit
        should_exit = True
        print("\n\n[退出] 收到中断信号，正在关闭服务...")
        # 强制退出，不等待清理
        os._exit(0)  # 使用 os._exit 强制退出，不等待线程
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while not should_exit:
            try:
                active = detect_microphone_activity(
                    listen_seconds=args.activity_window,
                    threshold=args.activity_threshold,
                    device=input_device,
                )
            except KeyboardInterrupt:
                # 重新抛出 KeyboardInterrupt，让外层处理
                raise
            except Exception as exc:
                if should_exit:
                    break
                print(f"[检测失败] {exc}")
                print("继续尝试...")
                try:
                    time.sleep(args.idle_sleep)
                except KeyboardInterrupt:
                    raise
                continue
            
            if should_exit:
                break
            
            if not active:
                try:
                    time.sleep(args.idle_sleep)
                except KeyboardInterrupt:
                    raise
                continue

            if should_exit:
                break

            print("\n[检测到活跃音频] 准备录制...")
            try:
                recorded_file = record_audio(
                    max_duration=args.max_duration,
                    auto_mode=True,
                    silence_threshold=args.silence_threshold,
                    silence_duration=args.silence_duration,
                    output_dir=ORIGIN_AUDIO_DIR,  # 指定录音文件保存到 origin_audio 文件夹
                )
            except KeyboardInterrupt:
                # 重新抛出 KeyboardInterrupt
                raise
            except Exception as exc:
                if should_exit:
                    break
                print(f"[录音失败] {exc}")
                continue

            if should_exit:
                break

            try:
                if args.mode == "online":
                    # 在线模式：翻译对方语音并在本地说/播放
                    audio_bytes = translate_online_voice(
                        recorded_file,
                        local_lang=args.local_lang,
                        source_language=getattr(args, "source_lang", None),
                    )
                    if audio_bytes:
                        play_audio_bytes(audio_bytes, output_device=getattr(args, "output_device", None))
                else:
                    # 本地模式：翻译本地语音并保存/输出到虚拟麦克风
                    _run_local_translation_pipeline(recorded_file, args)
            except KeyboardInterrupt:
                # 重新抛出 KeyboardInterrupt
                raise
            except Exception as exc:
                if should_exit:
                    break
                print(f"[翻译失败] {exc}")
                continue
    except KeyboardInterrupt:
        print("\n[退出] 收到中断信号，正在关闭服务...")
    finally:
        # 强制退出，不等待清理
        import os
        os._exit(0)


def _run_local_translation_pipeline(voice_input: AudioInput, args) -> None:
    """运行本地翻译流程：ASR -> 翻译 -> TTS -> 输出到虚拟麦克风"""
    _persist_origin_audio(voice_input)

    # 使用 local_translator 模块
    try:
        result = translate_local_voice(
            voice_input,
            local_lang=args.local_lang,
            target_lang=args.target_lang,
            language_hint=getattr(args, "language_hint", None),
        )
    except KeyboardInterrupt:
        # 如果是用户中断，重新抛出
        raise
    except Exception as e:
        print(f"\n✗ 翻译流程失败: {e}")
        print("⚠️  继续监听下一次语音输入...")
        import traceback
        traceback.print_exc()
        return
    
    # 如果翻译结果为空（ASR 超时或失败），跳过后续处理
    if not result or len(result) == 0:
        print("⚠️  翻译结果为空，跳过输出和保存")
        return

    # 如果启用了虚拟麦克风输出，将翻译后的音频输出到虚拟麦克风（BlackHole）
    if getattr(args, "virtual_mic", False) and isinstance(result, bytes):
        virtual_mic_device = getattr(args, "virtual_mic_device", None)
        if virtual_mic_device is None:
            # 使用配置的虚拟音频设备索引作为默认值
            try:
                if sd is not None:
                    device_info = sd.query_devices(CABLE_OUTPUT_INDEX)
                    if device_info.get('max_output_channels', 0) > 0:
                        virtual_mic_device = CABLE_OUTPUT_INDEX
            except Exception:
                pass
        if virtual_mic_device is not None and not isinstance(virtual_mic_device, int):
            try:
                virtual_mic_device = int(virtual_mic_device)
            except (ValueError, TypeError):
                pass
        _output_to_virtual_microphone(result, output_device=virtual_mic_device)

    # 保存翻译后的音频文件
    output_path = _materialise_output(result, args.output, input_path=Path(voice_input) if isinstance(voice_input, (str, Path)) else None)
    if output_path:
        print("\n✓ 翻译完成！")
        print(f"  输入文件: {voice_input}")
        print(f"  输出文件: {output_path.resolve()}")
    else:
        print("⚠️  警告: 翻译音频文件未生成")


if __name__ == "__main__":
    # 支持直接运行和作为模块运行
    try:
        main()
    except KeyboardInterrupt:
        print("\n[退出] 程序已停止")
        import sys
        sys.exit(0)