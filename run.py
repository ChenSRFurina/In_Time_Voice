#!/usr/bin/env python3
"""同时运行扬声器音频翻译和麦克风音频翻译服务"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from voice_clone import set_clone_reference

# 默认设备索引（保持与 main.py 中的配置一致）
DEFAULT_VOICEMEETER_INPUT_INDEX = 3
DEFAULT_CABLE_OUTPUT_INDEX = 13
DEFAULT_SPEAKER_CAPTURE_INDEX = 31
DEFAULT_SPEAKER_OUTPUT_INDEX = 44


class DualTranslatorService:
    """双翻译服务：同时运行扬声器翻译和麦克风翻译"""
    
    def __init__(
        self,
        # 扬声器翻译配置
        speaker_source_lang: str = "english",
        speaker_target_lang: str = "chinese",
        # 麦克风翻译配置
        mic_source_lang: str = "chinese",
        mic_target_lang: str = "english",
        # 麦克风翻译使用的音色克隆参考录音（可选）
        mic_voice_clone_file: Optional[str] = None,
    ):
        self.speaker_source_lang = speaker_source_lang
        self.speaker_target_lang = speaker_target_lang
        self.mic_source_lang = mic_source_lang
        self.mic_target_lang = mic_target_lang
        self.mic_voice_clone_file = mic_voice_clone_file
        
        self.speaker_process: Optional[subprocess.Popen] = None
        self.mic_process: Optional[subprocess.Popen] = None
        self.is_running = False
    
    def start(self):
        """启动双翻译服务"""
        if self.is_running:
            print("⚠️  服务已在运行中")
            return
        
        print("=" * 70)
        print("🚀 启动双翻译服务")
        print("=" * 70)
        print()
        print("📌 功能说明：")
        print("   1. 扬声器翻译：从扬声器捕获音频 → 翻译 → 播放到扬声器")
        print("   2. 麦克风翻译：从麦克风捕获音频 → 翻译 → 输出到虚拟麦克风")
        print()
        print("📋 配置信息：")
        print(f"   扬声器翻译: {self.speaker_source_lang} → {self.speaker_target_lang}")
        print(f"   麦克风翻译: {self.mic_source_lang} → {self.mic_target_lang}")
        print()
        print("⚠️  重要提示：")
        import sys
        if sys.platform == 'win32':
            print("   1. 确保已安装 VB-CABLE 虚拟音频设备")
            print("   2. Windows 声音输出应设置为 VB-CABLE（用于捕获扬声器输出）")
            print("   3. Windows 声音输入应设置为 VB-CABLE（如果使用虚拟麦克风）")
        else:
            print("   1. 确保已配置 Multi-Output Device（包含扬声器 + BlackHole）")
            print("   2. 系统输出应设置为 Multi-Output Device")
            print("   3. 系统输入应设置为 BlackHole（如果使用虚拟麦克风）")
        print()
        print("按 Ctrl+C 停止服务")
        print("=" * 70)
        print()
        
        self.is_running = True
        
        # 启动扬声器翻译进程（在线模式）
        # 使用 main.py 中配置的设备索引
        try:
            from intime_voice.main import SPEAKER_CAPTURE_INDEX, SPEAKER_OUTPUT_INDEX
            speaker_input_device = SPEAKER_CAPTURE_INDEX
            speaker_output_device = SPEAKER_OUTPUT_INDEX
        except ImportError:
            # 如果无法导入，尝试从 main.py 直接读取
            import importlib.util
            main_path = Path(__file__).parent / "main.py"
            if main_path.exists():
                spec = importlib.util.spec_from_file_location("main", main_path)
                main_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(main_module)
                speaker_input_device = getattr(main_module, "SPEAKER_CAPTURE_INDEX", DEFAULT_SPEAKER_CAPTURE_INDEX)
                speaker_output_device = getattr(main_module, "SPEAKER_OUTPUT_INDEX", DEFAULT_SPEAKER_OUTPUT_INDEX)
            else:
                # 回退到默认索引，再尝试自动查找
                speaker_input_device = DEFAULT_SPEAKER_CAPTURE_INDEX
                speaker_output_device = DEFAULT_SPEAKER_OUTPUT_INDEX
                from intime_voice.audio_checker import find_speaker_output_device, find_virtual_audio_input_device
                speaker_input_device = find_virtual_audio_input_device() or speaker_input_device
                speaker_output_device = find_speaker_output_device() or speaker_output_device
        
        # 获取 main.py 的路径
        main_py_path = Path(__file__).parent / "main.py"
        
        speaker_cmd = [
            sys.executable, str(main_py_path),
            "--mode", "online",
            "--streaming",
            "--source-lang", self.speaker_source_lang,
            "--local-lang", self.speaker_target_lang,
        ]
        
        # 如果配置了输入设备，指定输入设备
        if speaker_input_device is not None:
            speaker_cmd.extend(["--input-device", str(speaker_input_device)])
            try:
                import sounddevice as sd
                device_info = sd.query_devices(speaker_input_device)
                print(f"[扬声器翻译] 使用输入设备: {device_info['name']} (索引: {speaker_input_device})")
            except Exception:
                pass
        
        # 如果配置了输出设备，指定输出设备
        if speaker_output_device is not None:
            speaker_cmd.extend(["--output-device", str(speaker_output_device)])
            try:
                import sounddevice as sd
                device_info = sd.query_devices(speaker_output_device)
                print(f"[扬声器翻译] 使用输出设备: {device_info['name']} (索引: {speaker_output_device})")
            except Exception:
                pass
        else:
            print("[扬声器翻译] ⚠️  警告: 未找到输出设备，将使用系统默认设备")
        
        print(f"[扬声器翻译] 启动命令: {' '.join(speaker_cmd)}")
        try:
            self.speaker_process = subprocess.Popen(
                speaker_cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            print(f"[扬声器翻译] ✓ 进程已启动 (PID: {self.speaker_process.pid})")
        except Exception as e:
            print(f"[扬声器翻译] 启动失败: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
            return
        
        # 稍微延迟，避免同时启动造成冲突
        time.sleep(1.0)
        
        # 启动麦克风翻译进程（本地模式）
        # 使用 main.py 中配置的设备索引
        try:
            from intime_voice.main import VOICEMEETER_INPUT_INDEX, CABLE_OUTPUT_INDEX
            mic_input_device = VOICEMEETER_INPUT_INDEX
            mic_virtual_device = CABLE_OUTPUT_INDEX
        except ImportError:
            # 如果无法导入，尝试从 main.py 直接读取
            import importlib.util
            main_path = Path(__file__).parent / "main.py"
            if main_path.exists():
                spec = importlib.util.spec_from_file_location("main", main_path)
                main_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(main_module)
                mic_input_device = getattr(main_module, "VOICEMEETER_INPUT_INDEX", DEFAULT_VOICEMEETER_INPUT_INDEX)
                mic_virtual_device = getattr(main_module, "CABLE_OUTPUT_INDEX", DEFAULT_CABLE_OUTPUT_INDEX)
            else:
                # 回退到默认索引，再尝试自动查找
                mic_input_device = DEFAULT_VOICEMEETER_INPUT_INDEX
                mic_virtual_device = DEFAULT_CABLE_OUTPUT_INDEX
                from intime_voice.audio_checker import get_default_input_device
                mic_input_device = get_default_input_device(avoid_virtual=True) or mic_input_device
        
        # 获取 main.py 的路径
        main_py_path = Path(__file__).parent / "main.py"
        
        mic_cmd = [
            sys.executable, str(main_py_path),
            "--mode", "local",
            "--streaming",
            "--local-lang", self.mic_source_lang,
            "--target-lang", self.mic_target_lang,
            "--virtual-mic",  # 启用虚拟麦克风输出
        ]

        mic_env = None
        # 如果指定了音色克隆参考录音，先写入配置文件，供子进程使用
        if self.mic_voice_clone_file:
            try:
                clone_cfg = set_clone_reference(
                    self.mic_voice_clone_file,
                    sample_text="这句话用于语音克隆",
                    apply_scope="mic",
                    force=True,
                )
                ref_path = clone_cfg.get("reference_audio")
                voice_id = clone_cfg.get("sf_voice_id")
                print(f"[麦克风翻译] 使用音色克隆参考录音: {ref_path}")
                if voice_id:
                    mic_env = os.environ.copy()
                    mic_env["VOICE_CLONE_ENABLED"] = "1"
                    print(f"[麦克风翻译] SiliconFlow 语音克隆 voice_id: {voice_id}")
            except Exception as e:
                print(f"[麦克风翻译] ⚠️ 设置音色克隆参考录音失败: {e}")
        
        # 如果配置了输入设备，指定输入设备
        if mic_input_device is not None:
            mic_cmd.extend(["--input-device", str(mic_input_device)])
            try:
                import sounddevice as sd
                device_info = sd.query_devices(mic_input_device)
                print(f"[麦克风翻译] 使用输入设备: {device_info['name']} (索引: {mic_input_device})")
            except Exception:
                pass
        else:
            print("[麦克风翻译] ⚠️  警告: 未找到输入设备，将使用系统默认设备")
        
        # 如果配置了虚拟麦克风设备，指定虚拟麦克风设备
        if mic_virtual_device is not None:
            mic_cmd.extend(["--virtual-mic-device", str(mic_virtual_device)])
            try:
                import sounddevice as sd
                device_info = sd.query_devices(mic_virtual_device)
                print(f"[麦克风翻译] 使用虚拟麦克风设备: {device_info['name']} (索引: {mic_virtual_device})")
            except Exception:
                pass
        
        print(f"[麦克风翻译] 启动命令: {' '.join(mic_cmd)}")
        try:
            self.mic_process = subprocess.Popen(
                mic_cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=mic_env,
            )
            print(f"[麦克风翻译] ✓ 进程已启动 (PID: {self.mic_process.pid})")
        except Exception as e:
            print(f"[麦克风翻译] 启动失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果麦克风翻译启动失败，停止扬声器翻译
            if self.speaker_process:
                self.speaker_process.terminate()
            self.is_running = False
            return
        
        print()
        print("=" * 70)
        print("✅ 双翻译服务已启动")
        print("=" * 70)
        print()
        
        # 等待进程运行
        try:
            while self.is_running:
                # 检查进程状态
                if self.speaker_process and self.speaker_process.poll() is not None:
                    print(f"[扬声器翻译] 进程已退出 (退出码: {self.speaker_process.returncode})")
                    self.speaker_process = None
                
                if self.mic_process and self.mic_process.poll() is not None:
                    print(f"[麦克风翻译] 进程已退出 (退出码: {self.mic_process.returncode})")
                    self.mic_process = None
                
                # 如果两个进程都退出了，停止服务
                if self.speaker_process is None and self.mic_process is None:
                    print("[服务] 所有进程已退出")
                    break
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[停止] 收到中断信号...")
        finally:
            self.stop()
    
    def stop(self):
        """停止双翻译服务"""
        if not self.is_running:
            return
        
        print("\n[停止] 正在停止服务...")
        self.is_running = False
        
        # 停止扬声器翻译进程
        if self.speaker_process:
            try:
                print("[停止] 正在停止扬声器翻译进程...")
                self.speaker_process.terminate()
                # 等待进程结束，最多等待 3 秒
                try:
                    self.speaker_process.wait(timeout=3.0)
                    print("[停止] ✓ 扬声器翻译进程已停止")
                except subprocess.TimeoutExpired:
                    print("[停止] ⚠️  强制终止扬声器翻译进程...")
                    self.speaker_process.kill()
                    self.speaker_process.wait()
                    print("[停止] ✓ 扬声器翻译进程已强制终止")
            except Exception as e:
                print(f"[停止] 停止扬声器翻译进程时出错: {e}")
            self.speaker_process = None
        
        # 停止麦克风翻译进程
        if self.mic_process:
            try:
                print("[停止] 正在停止麦克风翻译进程...")
                self.mic_process.terminate()
                # 等待进程结束，最多等待 3 秒
                try:
                    self.mic_process.wait(timeout=3.0)
                    print("[停止] ✓ 麦克风翻译进程已停止")
                except subprocess.TimeoutExpired:
                    print("[停止] ⚠️  强制终止麦克风翻译进程...")
                    self.mic_process.kill()
                    self.mic_process.wait()
                    print("[停止] ✓ 麦克风翻译进程已强制终止")
            except Exception as e:
                print(f"[停止] 停止麦克风翻译进程时出错: {e}")
            self.mic_process = None
        
        print("[停止] ✓ 服务已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="同时运行扬声器音频翻译和麦克风音频翻译服务"
    )
    
    # 扬声器翻译配置
    parser.add_argument(
        "--speaker-source-lang",
        default="english",
        help="扬声器音频的源语言（默认: english）"
    )
    parser.add_argument(
        "--speaker-target-lang",
        default="chinese",
        help="扬声器音频的目标语言（默认: chinese）"
    )
    
    # 麦克风翻译配置
    parser.add_argument(
        "--mic-source-lang",
        default="chinese",
        help="麦克风音频的源语言（默认: chinese）"
    )
    parser.add_argument(
        "--mic-target-lang",
        default="english",
        help="麦克风音频的目标语言（默认: english）"
    )
    # 麦克风音色克隆参考录音
    parser.add_argument(
        "--mic-voice-clone-file",
        default=None,
        help="用于麦克风翻译的音色克隆参考录音文件路径（可选）",
    )
    
    args = parser.parse_args()
    
    # 创建服务
    service = DualTranslatorService(
        speaker_source_lang=args.speaker_source_lang,
        speaker_target_lang=args.speaker_target_lang,
        mic_source_lang=args.mic_source_lang,
        mic_target_lang=args.mic_target_lang,
        mic_voice_clone_file=args.mic_voice_clone_file,
    )
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n[信号] 收到退出信号，正在停止...")
        service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动服务
    try:
        service.start()
    except Exception as e:
        print(f"\n[错误] 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
