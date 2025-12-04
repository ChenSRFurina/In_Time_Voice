"""
流式翻译器：持续监听麦克风，实时翻译
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None

try:
    import sounddevice as sd  # type: ignore[import-not-found]
except ImportError:
    sd = None

# 导入流式 ASR
try:
    import sys
    from pathlib import Path
    # 当前文件: intime_voice/intime_voice/streaming_translator.py
    # 目标文件: intime_voice/test_auto/streaming_asr.py
    # 从 intime_voice/intime_voice/ 向上到 intime_voice/，然后进入 test_auto/
    current_file = Path(__file__).resolve()
    streaming_asr_path = current_file.parent.parent / "test_auto" / "streaming_asr.py"
    
    if streaming_asr_path.exists():
        # 添加 streaming_asr.py 的父目录到路径
        test_auto_dir = str(streaming_asr_path.parent)
        if test_auto_dir not in sys.path:
            sys.path.insert(0, test_auto_dir)
        # 还需要添加 intime_voice 目录，因为 streaming_asr.py 需要导入 voice_translator
        intime_voice_dir = str(current_file.parent)
        if intime_voice_dir not in sys.path:
            sys.path.insert(0, intime_voice_dir)
        from streaming_asr import StreamingASR
    else:
        StreamingASR = None
        print(f"[流式翻译] 警告: 未找到 streaming_asr.py")
        print(f"  当前文件: {current_file}")
        print(f"  尝试的路径: {streaming_asr_path}")
        print(f"  路径是否存在: {streaming_asr_path.exists()}")
except ImportError as e:
    StreamingASR = None
    print(f"[流式翻译] 导入 StreamingASR 失败: {e}")
    import traceback
    traceback.print_exc()

from voice_translator import translate_text, tts


class StreamingTranslator:
    """流式翻译器：持续监听麦克风，实时翻译"""
    
    DEFAULT_SAMPLE_RATE = 16_000
    DEFAULT_CHUNK_SIZE = 3200  # ~0.2秒 @ 16kHz
    
    def __init__(
        self,
        source_language: str = "chinese",
        target_language: str = "english",
        input_device: Optional[int | str] = None,
        virtual_mic_device: Optional[int | str] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        if sd is None or np is None:
            raise RuntimeError("需要 sounddevice 和 numpy")
        if StreamingASR is None:
            raise RuntimeError(
                "需要 StreamingASR，请确保 test_auto/streaming_asr.py 存在。\n"
                f"当前文件位置: {Path(__file__).resolve()}\n"
                f"尝试查找: {Path(__file__).resolve().parent.parent / 'test_auto' / 'streaming_asr.py'}"
            )
        
        self.source_language = source_language
        self.target_language = target_language
        self.input_device = input_device
        self.virtual_mic_device = virtual_mic_device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.asr_sample_rate = 16_000  # ASR 需要的采样率（固定为 16000 Hz）
        
        # 流式 ASR
        self.streaming_asr = None
        
        # 翻译状态
        self.is_translating = False
        self.translation_queue = queue.Queue()  # 待翻译的文本队列
        
        # 音频流
        self.stream = None
        self.is_running = False
        
        # 虚拟麦克风输出队列（存储完整的音频数据）
        self.virtual_mic_queue = queue.Queue(maxsize=10)
        
        # 持续的虚拟麦克风输出流（用于实时播放）
        self.virtual_mic_stream = None
        self.virtual_mic_output_queue = queue.Queue(maxsize=100)  # 待播放的音频块队列
        self.virtual_mic_stream_lock = threading.Lock()
        self.virtual_mic_sample_rate = None
        self.virtual_mic_channels = None
    
    def _on_sentence_complete(self, text: str):
        """当检测到句子完成时调用"""
        if not text or not text.strip() or self.is_translating:
            return
        
        print(f"\n[流式翻译] 检测到句子完成: {text}")
        
        # 将文本加入翻译队列
        try:
            self.translation_queue.put_nowait(text)
        except queue.Full:
            print("[流式翻译] 翻译队列已满，跳过本次翻译")
    
    def _translate_and_output(self, text: str):
        """翻译文本并输出到虚拟麦克风（在后台线程中执行）"""
        try:
            self.is_translating = True
            
            # 步骤1: 翻译
            print(f"[流式翻译] 开始翻译: {text}")
            translated_text = translate_text(text, self.source_language, self.target_language)
            print(f"[流式翻译] 翻译结果: {translated_text}")
            
            if not translated_text or not translated_text.strip():
                print("[流式翻译] 翻译结果为空，跳过")
                return
            
            # 步骤2: TTS
            print(f"[流式翻译] 开始 TTS...")
            audio_bytes = tts(translated_text, self.target_language)
            print(f"[流式翻译] TTS 完成，音频大小: {len(audio_bytes)} 字节")
            
            # 步骤3: 输出到虚拟麦克风
            if self.virtual_mic_device is not None:
                try:
                    print(f"[流式翻译] 将音频加入虚拟麦克风队列，大小: {len(audio_bytes)} 字节")
                    self.virtual_mic_queue.put_nowait(audio_bytes)
                    print(f"[流式翻译] ✓ 音频已加入虚拟麦克风队列")
                except queue.Full:
                    print("[流式翻译] 虚拟麦克风队列已满，跳过输出")
            else:
                print("[流式翻译] 虚拟麦克风设备未设置，跳过输出")
            
        except Exception as e:
            print(f"[流式翻译失败] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_translating = False
    
    def _process_translation_queue(self):
        """处理翻译队列（在后台线程中）"""
        while self.is_running:
            try:
                text = self.translation_queue.get(timeout=0.1)
                # 在后台线程中处理翻译
                threading.Thread(
                    target=self._translate_and_output,
                    args=(text,),
                    daemon=True
                ).start()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[翻译队列处理错误] {e}")
    
    def _virtual_mic_stream_callback(self, outdata, frames, time_info, status):
        """虚拟麦克风输出流的回调函数"""
        if status:
            print(f"[虚拟麦克风流状态] {status}")
        
        try:
            # 从队列获取音频块
            chunk = self.virtual_mic_output_queue.get_nowait()
            
            # 确保块大小和声道数匹配
            target_channels = outdata.shape[1] if len(outdata.shape) > 1 else 1
            chunk_channels = chunk.shape[1] if len(chunk.shape) > 1 else 1
            
            # 调整声道数
            if chunk_channels != target_channels:
                if target_channels == 2 and chunk_channels == 1:
                    chunk = np.repeat(chunk, 2, axis=1)
                elif target_channels == 1 and chunk_channels == 2:
                    chunk = np.mean(chunk, axis=1, keepdims=True)
            
            # 确保块大小匹配
            if len(chunk) < frames:
                # 如果块太小，用零填充
                padding = np.zeros((frames - len(chunk), target_channels), dtype=np.float32)
                outdata[:] = np.vstack([chunk, padding])
            elif len(chunk) > frames:
                # 如果块太大，截取
                outdata[:] = chunk[:frames]
            else:
                outdata[:] = chunk
        except queue.Empty:
            # 如果没有数据，输出静音
            outdata.fill(0)
    
    def _output_virtual_mic_worker(self):
        """虚拟麦克风输出工作线程：处理音频并加入播放队列"""
        if self.virtual_mic_device is None:
            print("[虚拟麦克风] 虚拟麦克风设备未设置，输出线程退出")
            return
        
        import io
        import wave
        
        # 尝试导入 soundfile，如果没有则使用 wave
        try:
            import soundfile as sf
            use_soundfile = True
            print("[虚拟麦克风] 使用 soundfile 库")
        except ImportError:
            use_soundfile = False
            print("[虚拟麦克风] 使用 wave 库")
        
        print(f"[虚拟麦克风] 输出工作线程已启动，等待音频数据...")
        
        while self.is_running:
            try:
                audio_bytes = self.virtual_mic_queue.get(timeout=0.1)
                print(f"[虚拟麦克风] 收到音频数据，大小: {len(audio_bytes)} 字节")
                
                # 读取音频数据
                if use_soundfile:
                    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                    channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
                    
                    if audio_data.ndim == 1:
                        audio_float = audio_data.reshape(-1, 1).astype(np.float32)
                    else:
                        audio_float = audio_data.astype(np.float32)
                else:
                    # 使用 wave 读取 WAV 数据
                    buffer = io.BytesIO(audio_bytes)
                    try:
                        with wave.open(buffer, "rb") as wf:
                            sample_rate = wf.getframerate()
                            channels = wf.getnchannels()
                            frames = wf.readframes(wf.getnframes())
                            
                            audio_array = np.frombuffer(frames, dtype=np.int16)
                            audio_float = audio_array.astype(np.float32) / 32767.0
                            
                            if channels > 1:
                                audio_float = audio_float.reshape(-1, channels)
                            else:
                                audio_float = audio_float.reshape(-1, 1)
                    except Exception as e:
                        print(f"[虚拟麦克风] 无法读取音频: {e}")
                        continue
                
                audio_float = np.clip(audio_float, -1.0, 1.0)
                
                # 验证设备并调整声道数
                try:
                    device_info = sd.query_devices(self.virtual_mic_device)
                    target_channels = min(channels, 2) if device_info.get('max_output_channels', 0) >= 2 else 1
                    
                    # 调整声道数
                    output_audio = audio_float
                    if channels > target_channels:
                        if target_channels == 1:
                            output_audio = np.mean(audio_float, axis=1, keepdims=True)
                        else:
                            output_audio = audio_float[:, :target_channels]
                    elif channels < target_channels and target_channels == 2:
                        output_audio = np.repeat(audio_float, 2, axis=1)
                    
                    # 如果采样率变化，需要重新创建输出流
                    if self.virtual_mic_stream and (
                        self.virtual_mic_sample_rate != sample_rate or 
                        self.virtual_mic_channels != target_channels
                    ):
                        print(f"[虚拟麦克风] 采样率或声道数变化，重新创建输出流...")
                        try:
                            self.virtual_mic_stream.stop()
                            self.virtual_mic_stream.close()
                        except Exception:
                            pass
                        self.virtual_mic_stream = None
                    
                    # 保存采样率和声道数（用于输出流）
                    self.virtual_mic_sample_rate = sample_rate
                    self.virtual_mic_channels = target_channels
                    
                    # 如果流不存在，创建新的流
                    if self.virtual_mic_stream is None:
                        print(f"[虚拟麦克风] 创建输出流: {sample_rate} Hz, {target_channels} 声道")
                        try:
                            self.virtual_mic_stream = sd.OutputStream(
                                device=self.virtual_mic_device,
                                samplerate=sample_rate,
                                channels=target_channels,
                                dtype='float32',
                                callback=self._virtual_mic_stream_callback,
                                blocksize=int(sample_rate * 0.05),  # 50ms 块
                            )
                            self.virtual_mic_stream.start()
                            print(f"[虚拟麦克风] ✓ 输出流已创建并启动")
                        except Exception as e:
                            print(f"[虚拟麦克风] 创建输出流失败: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    print(f"[虚拟麦克风] 处理音频: {len(output_audio)} 样本, {target_channels} 声道, {sample_rate} Hz")
                    print(f"  音频时长: {len(output_audio)/sample_rate:.2f} 秒")
                    
                    # 将音频分块加入输出队列（用于流式播放）
                    chunk_size = int(sample_rate * 0.05)  # 50ms 块，更流畅
                    chunks_put = 0
                    for i in range(0, len(output_audio), chunk_size):
                        chunk = output_audio[i:i+chunk_size]
                        try:
                            self.virtual_mic_output_queue.put(chunk.astype(np.float32), timeout=0.5)
                            chunks_put += 1
                        except queue.Full:
                            print(f"[虚拟麦克风] 输出队列已满，已放入 {chunks_put} 块，跳过剩余音频")
                            break
                    
                    print(f"[虚拟麦克风] ✓ 音频已加入播放队列 ({chunks_put} 块, {len(output_audio)/sample_rate:.2f} 秒)")
                except Exception as e:
                    print(f"[虚拟麦克风处理失败] {e}")
                    import traceback
                    traceback.print_exc()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[虚拟麦克风工作线程错误] {e}")
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            print(f"[音频状态] {status}")
        
        # 计算音量，用于调试
        if np is not None:
            volume = np.linalg.norm(indata) * 10
            # 只在有声音时打印（避免日志过多）
            if volume > 0.1:
                print(f"[音频输入] 音量: {volume:.2f}", end='\r')
        
        # 实时发送音频到流式 ASR
        if self.streaming_asr and self.streaming_asr.is_running:
            # 如果设备采样率与ASR需要的采样率不同，需要重采样
            if self.sample_rate != self.asr_sample_rate:
                # 重采样到ASR需要的采样率（16000 Hz）
                try:
                    from scipy import signal
                    # 计算重采样因子
                    num_samples = int(len(indata) * self.asr_sample_rate / self.sample_rate)
                    # 重采样
                    if indata.ndim == 1:
                        resampled = signal.resample(indata, num_samples, axis=0)
                    else:
                        resampled = signal.resample(indata, num_samples, axis=0)
                    audio_int16 = (resampled * 32767).astype(np.int16)
                except ImportError:
                    # 如果没有 scipy，使用简单的线性插值
                    if np is not None:
                        num_samples = int(len(indata) * self.asr_sample_rate / self.sample_rate)
                        if indata.ndim == 1:
                            indices = np.linspace(0, len(indata) - 1, num_samples)
                            resampled = np.interp(indices, np.arange(len(indata)), indata)
                        else:
                            resampled = np.zeros((num_samples, indata.shape[1]), dtype=indata.dtype)
                            for ch in range(indata.shape[1]):
                                indices = np.linspace(0, len(indata) - 1, num_samples)
                                resampled[:, ch] = np.interp(indices, np.arange(len(indata)), indata[:, ch])
                        audio_int16 = (resampled * 32767).astype(np.int16)
                    else:
                        # 如果numpy也没有，直接使用原始数据（可能质量下降）
                        print(f"[音频输入] 警告: 无法重采样，ASR可能无法正常工作")
                        audio_int16 = (indata * 32767).astype(np.int16)
            else:
                # 采样率匹配，直接使用
                audio_int16 = (indata * 32767).astype(np.int16)
            
            # 确保是单声道（ASR通常需要单声道）
            if audio_int16.ndim > 1 and audio_int16.shape[1] > 1:
                audio_int16 = np.mean(audio_int16, axis=1).astype(np.int16)
            
            audio_bytes = audio_int16.tobytes()
            # 发送到流式 ASR
            self.streaming_asr.send_audio_chunk(audio_bytes)
    
    def start(self):
        """启动流式翻译"""
        if self.is_running:
            return
        
        print("=" * 60)
        print("启动流式翻译服务")
        print(f"源语言: {self.source_language}")
        print(f"目标语言: {self.target_language}")
        print(f"输入设备: {self.input_device or '默认'}")
        if self.virtual_mic_device is not None:
            print(f"虚拟麦克风设备: {self.virtual_mic_device}")
        print("=" * 60)
        print("持续监听麦克风，检测到句子完成时自动翻译")
        print("按 Ctrl+C 停止")
        print()
        
        # 启动流式 ASR
        try:
            print("[流式ASR] 正在启动流式 ASR 连接...")
            self.streaming_asr = StreamingASR(
                source_language=self.source_language,
                on_sentence_complete=self._on_sentence_complete,
            )
            self.streaming_asr.start()
            print("[流式ASR] 流式 ASR 连接已启动")
        except Exception as e:
            print(f"[流式ASR] 启动失败: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 启动翻译队列处理线程
        self.is_running = True
        translation_thread = threading.Thread(target=self._process_translation_queue, daemon=True)
        translation_thread.start()
        
        # 启动虚拟麦克风输出线程
        if self.virtual_mic_device is not None:
            print(f"[虚拟麦克风] 启动虚拟麦克风输出线程，设备: {self.virtual_mic_device}")
            
            # 启动音频处理线程
            virtual_mic_thread = threading.Thread(target=self._output_virtual_mic_worker, daemon=True)
            virtual_mic_thread.start()
            
            # 注意：输出流会在第一个音频到达时创建（因为我们需要知道实际的采样率）
            # 这样可以根据 TTS 输出的采样率动态调整
        else:
            print("[虚拟麦克风] 虚拟麦克风设备未设置，跳过输出线程")
        
        # 启动音频流
        try:
            # 验证输入设备
            if self.input_device is not None:
                try:
                    device_info = sd.query_devices(self.input_device)
                    if device_info['max_input_channels'] == 0:
                        print(f"⚠️  警告: 设备 {self.input_device} ({device_info['name']}) 没有输入通道")
                        print("尝试使用系统默认输入设备...")
                        self.input_device = None
                    else:
                        print(f"[音频输入] 使用设备: {device_info['name']} (索引: {self.input_device})")
                except Exception as e:
                    print(f"⚠️  警告: 无法查询设备 {self.input_device}: {e}")
                    print("尝试使用系统默认输入设备...")
                    self.input_device = None
            
            print(f"[音频输入] 启动音频流...")
            
            # 尝试使用 44100 Hz 作为输入采样率（如果设备支持）
            desired_sample_rate = 44100
            actual_sample_rate = desired_sample_rate
            try:
                sd.check_input_settings(
                    device=self.input_device,
                    samplerate=desired_sample_rate,
                    channels=1,
                    dtype='float32',
                )
                print(f"[音频输入] 设备支持 {desired_sample_rate} Hz，使用该采样率进行输入")
                actual_sample_rate = desired_sample_rate
            except Exception as check_err:
                # 如果设备不支持 44100，尝试使用配置的采样率（通常是 16000）
                try:
                    sd.check_input_settings(
                        device=self.input_device,
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype='float32',
                    )
                    actual_sample_rate = self.sample_rate
                    print(f"[音频输入] 设备不支持 {desired_sample_rate} Hz，使用配置的采样率 {self.sample_rate} Hz")
                except Exception:
                    # 如果配置的采样率也不支持，使用设备默认采样率
                    if self.input_device is not None:
                        try:
                            device_info = sd.query_devices(self.input_device)
                            fallback_rate = device_info.get('default_samplerate')
                            if fallback_rate:
                                actual_sample_rate = int(fallback_rate)
                                print(f"[音频输入] ⚠️  设备不支持 {desired_sample_rate} Hz 和 {self.sample_rate} Hz，使用设备默认采样率 {actual_sample_rate} Hz")
                        except Exception:
                            pass
            
            # 更新采样率
            if actual_sample_rate != self.sample_rate:
                self.sample_rate = actual_sample_rate
                # 更新块大小以适应新的采样率
                self.chunk_size = int(self.sample_rate * 0.2)
            
            print(f"  输入采样率: {self.sample_rate} Hz")
            print(f"  ASR处理采样率: {self.asr_sample_rate} Hz")
            print(f"  块大小: {self.chunk_size} 样本")
            print(f"  设备: {self.input_device or '系统默认'}")
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.input_device,
                callback=self.audio_callback,
                blocksize=self.chunk_size,
            )
            
            self.stream.start()
            print("[音频输入] ✓ 音频流已启动，正在监听麦克风...")
            print("提示: 请对着麦克风说话，程序会自动检测句子完成并翻译")
            print()
            
            # 保持运行
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[停止] 收到中断信号...")
        except Exception as e:
            print(f"\n[错误] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """停止流式翻译"""
        self.is_running = False
        
        # 停止虚拟麦克风输出流
        if self.virtual_mic_stream:
            try:
                self.virtual_mic_stream.stop()
                self.virtual_mic_stream.close()
            except Exception:
                pass
            self.virtual_mic_stream = None
        
        # 停止流式 ASR
        if self.streaming_asr:
            self.streaming_asr.stop()
            self.streaming_asr = None
        
        # 停止音频流
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        print("[已停止]")

