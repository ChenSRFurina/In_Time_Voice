import threading
import time
from typing import Callable, Optional, Union

import numpy as np  # type: ignore[import-not-found]
import sounddevice as sd  # type: ignore[import-not-found]

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1


def _volume_level(indata) -> float:
    """计算当前音量等级。"""
    return float(np.linalg.norm(indata) * 10)


def get_default_input_device(avoid_virtual: bool = True) -> Optional[Union[int, str]]:
    """
    获取默认的输入设备。
    
    Args:
        avoid_virtual: 如果为 True，会优先选择非虚拟音频设备（如 VB-CABLE）
    """
    try:
        # 获取默认输入设备索引
        default_input_idx = sd.default.device[0]
        if default_input_idx is not None:
            try:
                device_info = sd.query_devices(default_input_idx)
                if device_info['max_input_channels'] > 0:
                    # 如果 avoid_virtual 为 True，且默认设备是虚拟音频设备，尝试找其他设备
                    if avoid_virtual and _is_virtual_audio_device(device_info['name']):
                        # 继续查找其他非虚拟设备
                        pass
                    else:
                        return default_input_idx
            except Exception:
                pass
    except Exception:
        pass
    
    # 尝试查找任何可用的输入设备
    try:
        devices = sd.query_devices()
        # 如果 avoid_virtual 为 True，优先选择非虚拟音频设备
        non_virtual_devices = []
        virtual_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                if _is_virtual_audio_device(device['name']):
                    virtual_devices.append(i)
                else:
                    non_virtual_devices.append(i)
        
        # 优先返回非虚拟设备
        if avoid_virtual and non_virtual_devices:
            return non_virtual_devices[0]
        elif non_virtual_devices:
            return non_virtual_devices[0]
        elif virtual_devices:
            return virtual_devices[0]
    except Exception:
        pass
    
    return None


def _is_virtual_audio_device(device_name: str) -> bool:
    """检查设备是否是虚拟音频设备（Windows: VB-CABLE 等，macOS: BlackHole）"""
    name_lower = device_name.lower()
    # Windows 虚拟音频设备
    if "vb-cable" in name_lower or "vb cable" in name_lower or "virtual cable" in name_lower:
        return True
    # macOS 虚拟音频设备
    if "blackhole" in name_lower:
        return True
    return False


def list_input_devices() -> None:
    """列出所有可用的输入设备。"""
    try:
        devices = sd.query_devices()
        print("=== 可用的输入设备 ===")
        has_input = False
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                has_input = True
                default_marker = " (默认)" if i == sd.default.device[0] else ""
                virtual_marker = ""
                if _is_virtual_audio_device(device['name']):
                    if "vb-cable" in device['name'].lower() or "vb cable" in device['name'].lower():
                        virtual_marker = " [虚拟音频设备 - 可用于捕获扬声器输出]"
                    elif "blackhole" in device['name'].lower():
                        virtual_marker = " [BlackHole - 可用于捕获扬声器输出]"
                print(f"设备 {i}: {device['name']}{default_marker}{virtual_marker}")
                print(f"  输入通道: {device['max_input_channels']}")
                print(f"  输出通道: {device['max_output_channels']}")
                print(f"  默认采样率: {device['default_samplerate']}")
                print("-" * 50)
        
        if not has_input:
            print("⚠️  未找到任何输入设备！")
            print("请检查：")
            print("  1. 麦克风是否已连接")
            print("  2. 系统权限是否允许访问麦克风")
            print("  3. 音频驱动是否正常工作")
    except Exception as exc:
        print(f"⚠️  无法列出音频设备: {exc}")


def list_all_devices() -> None:
    """列出所有音频设备（包括输入和输出）。"""
    try:
        devices = sd.query_devices()
        print("=" * 70)
        print("=== 所有音频设备（输入和输出）===")
        print("=" * 70)
        print()
        
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        
        virtual_devices = []
        has_input = False
        has_output = False
        
        for i, device in enumerate(devices):
            max_in = device.get('max_input_channels', 0)
            max_out = device.get('max_output_channels', 0)
            
            # 检查是否是虚拟音频设备
            is_virtual = _is_virtual_audio_device(device['name'])
            if is_virtual:
                virtual_devices.append((i, device, max_in, max_out))
            
            # 标记默认设备
            markers = []
            if i == default_input:
                markers.append("默认输入")
            if i == default_output:
                markers.append("默认输出")
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            
            # 打印设备信息
            print(f"[{i}] {device['name']}{marker_str}")
            if max_in > 0:
                has_input = True
                print(f"    📥 输入通道: {max_in}")
            if max_out > 0:
                has_output = True
                print(f"    📤 输出通道: {max_out}")
            print(f"    采样率: {device.get('default_samplerate', 'N/A')} Hz")
            
            if is_virtual:
                device_type = "虚拟音频设备"
                if "vb-cable" in device['name'].lower() or "vb cable" in device['name'].lower():
                    device_type = "VB-CABLE 虚拟音频设备"
                elif "blackhole" in device['name'].lower():
                    device_type = "BlackHole 虚拟音频设备"
                print(f"    ⭐ {device_type} - 可用于捕获扬声器输出")
            print("-" * 70)
        
        # 显示虚拟音频设备配置提示
        if virtual_devices:
            print()
            print("=" * 70)
            print("📌 虚拟音频设备配置提示（用于捕获扬声器输出）")
            print("=" * 70)
            for idx, device, max_in, max_out in virtual_devices:
                print(f"\n设备 [{idx}] {device['name']}:")
                if max_in > 0:
                    print(f"  ✓ 可用作输入设备捕获扬声器输出（{max_in} 通道）")
                    print(f"    使用: --input-device {idx}")
                else:
                    print(f"  ⚠️  此设备没有输入通道，无法用于捕获音频")
                if max_out > 0:
                    print(f"  ✓ 可用作输出设备（{max_out} 通道）")
            
            # Windows 配置提示
            import sys
            if sys.platform == 'win32':
                print("\nWindows 配置步骤：")
                print("1. 安装 VB-Audio Virtual Cable (VB-CABLE)")
                print("   下载地址: https://vb-audio.com/Cable/")
                print("2. 在 Windows 声音设置中，将 VB-CABLE 设置为默认播放设备")
                print("3. 运行程序时使用 --input-device <VB-CABLE索引> 来捕获扬声器输出")
            else:
                print("\nmacOS 配置步骤：")
                print("1. 在「音频 MIDI 设置」中创建 Multi-Output Device")
                print("2. 勾选您的扬声器和 BlackHole 2ch")
                print("3. 在系统设置中选择该 Multi-Output Device 作为输出")
                print("4. 运行程序时使用 --input-device <BlackHole索引> 来捕获扬声器输出")
            print("=" * 70)
        elif not has_input:
            print()
            print("⚠️  未找到虚拟音频设备")
            import sys
            if sys.platform == 'win32':
                print("要捕获扬声器输出，请安装 VB-Audio Virtual Cable:")
                print("  https://vb-audio.com/Cable/")
            else:
                print("要捕获扬声器输出，请安装 BlackHole:")
                print("  https://github.com/ExistentialAudio/BlackHole")
        
    except Exception as exc:
        print(f"⚠️  无法列出音频设备: {exc}")


def find_virtual_audio_input_device() -> Optional[int]:
    """查找虚拟音频设备中可用作输入（用于捕获扬声器输出）的设备索引。"""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if _is_virtual_audio_device(device['name']):
                # 检查是否有输入通道
                if device.get('max_input_channels', 0) > 0:
                    return i
        return None
    except Exception:
        return None


def find_blackhole_input_device() -> Optional[int]:
    """查找 BlackHole 设备（兼容性函数，推荐使用 find_virtual_audio_input_device）"""
    return find_virtual_audio_input_device()


def find_speaker_output_device() -> Optional[int]:
    """
    查找输出设备（可用作播放设备）。
    优先返回系统默认输出设备（可能是扬声器、耳机等）。
    """
    try:
        devices = sd.query_devices()
        default_output = sd.default.device[1]
        
        # 优先返回默认输出设备（通常是用户选择的扬声器或耳机）
        if default_output is not None:
            try:
                device_info = sd.query_devices(default_output)
                if device_info.get('max_output_channels', 0) > 0:
                    # 排除虚拟音频设备和 Multi-Output Device
                    device_name = device_info['name'].lower()
                    if not _is_virtual_audio_device(device_info['name']) and "multi-output" not in device_name:
                        return default_output
            except Exception:
                pass
        
        # 查找包含关键词的输出设备（包括耳机）
        output_keywords = ["speaker", "built-in", "macbook", "imac", "airpods", "headphone", "headset", "earphone", "earbud", "bluetooth"]
        for keyword in output_keywords:
            for i, device in enumerate(devices):
                device_name = device['name'].lower()
                if keyword in device_name and device.get('max_output_channels', 0) > 0:
                    # 排除虚拟音频设备
                    if not _is_virtual_audio_device(device['name']):
                        return i
        
        # 如果找不到，返回任何有输出通道的非虚拟音频设备
        for i, device in enumerate(devices):
            if device.get('max_output_channels', 0) > 0 and not _is_virtual_audio_device(device['name']):
                return i
        
        return None
    except Exception:
        return None


def detect_microphone_activity(
    listen_seconds: float = 1.0,
    threshold: float = 1.0,
    *,
    samplerate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    on_activity: Optional[Callable[[float], None]] = None,
    device: Optional[Union[int, str]] = None,
) -> bool:
    """
    监听麦克风指定时长并返回是否检测到活跃音量。

    Args:
        listen_seconds: 每次检测的时间窗（秒）
        threshold: 认定为活跃的音量阈值
        samplerate: 采样率
        channels: 声道数
        on_activity: 可选回调，收到音量值时触发
        device: 输入设备索引或名称，如果为None则使用默认设备
    """
    detected = False
    error_occurred = None
    stream_active = threading.Event()
    stream_active.set()

    def audio_callback(indata, frames, callback_time, status):
        nonlocal detected, error_occurred
        if not stream_active.is_set():
            return
        
        if status:
            print(f"[音频状态] {status}")
            if status.input_overflow:
                error_occurred = "输入溢出"
            if status.input_underflow:
                error_occurred = "输入欠载"

        try:
            volume = _volume_level(indata)
            if on_activity:
                on_activity(volume)
            if volume > threshold:
                detected = True
        except Exception as e:
            error_occurred = f"处理音频数据时出错: {e}"

    # 如果没有指定设备，尝试获取默认输入设备
    if device is None:
        device = get_default_input_device()
        if device is None:
            print("⚠️  警告: 未找到可用的输入设备，尝试使用系统默认设备")
            # 让sounddevice使用系统默认设备

    try:
        # 验证设备是否可用
        if device is not None:
            try:
                device_info = sd.query_devices(device)
                if device_info['max_input_channels'] == 0:
                    print(f"⚠️  警告: 设备 {device} ({device_info['name']}) 没有输入通道")
                    device = None  # 回退到默认设备
            except Exception:
                print(f"⚠️  警告: 无法查询设备 {device}，使用默认设备")
                device = None

        stream = None
        try:
            stream = sd.InputStream(
                callback=audio_callback,
                samplerate=samplerate,
                channels=channels,
                device=device,
            )
            stream.start()
            
            # 使用 time.sleep 替代 sd.sleep，这样能更好地响应 KeyboardInterrupt
            # 分段睡眠以便更快响应中断
            elapsed = 0.0
            sleep_chunk = 0.1  # 每100ms检查一次
            try:
                while elapsed < listen_seconds:
                    time.sleep(min(sleep_chunk, listen_seconds - elapsed))
                    elapsed += sleep_chunk
                    if not stream_active.is_set():
                        break
            except KeyboardInterrupt:
                # 确保 KeyboardInterrupt 能够立即中断
                raise
        finally:
            stream_active.clear()
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        raise
    except sd.PortAudioError as exc:
        error_msg = f"麦克风检测失败 (PortAudio错误): {exc}"
        if "No input channels" in str(exc) or "Invalid device" in str(exc):
            error_msg += "\n提示: 请检查麦克风是否已连接并具有输入权限"
        raise RuntimeError(error_msg) from exc
    except OSError as exc:
        error_msg = f"麦克风检测失败 (系统错误): {exc}"
        if "Permission denied" in str(exc) or "权限" in str(exc):
            error_msg += "\n提示: 请授予应用程序麦克风访问权限"
        raise RuntimeError(error_msg) from exc
    except Exception as exc:
        error_msg = f"麦克风检测失败: {exc}"
        if error_occurred:
            error_msg += f"\n额外信息: {error_occurred}"
        raise RuntimeError(error_msg) from exc

    if error_occurred:
        print(f"⚠️  警告: {error_occurred}")

    return detected


def monitor_microphone(duration: int = 10, threshold: float = 1.0) -> None:
    """CLI 监控：持续打印音量并提示是否活跃。"""
    print("开始监控麦克风...")

    end_time = time.time() + duration

    while time.time() < end_time:
        active = detect_microphone_activity(
            listen_seconds=0.5,
            threshold=threshold,
            on_activity=lambda vol: print(f"当前音量级别: {vol:.2f}"),
        )
        if active:
            print("🎤 麦克风活跃!")


if __name__ == "__main__":
    monitor_microphone(duration=30)