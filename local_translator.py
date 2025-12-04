"""Workflow helpers for translating locally recorded audio and outputting to virtual microphone."""

from __future__ import annotations

import io
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Union

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:
    np = None

try:
    import sounddevice as sd  # type: ignore[import-not-found]
except ImportError:
    sd = None

try:
    import soundfile as sf  # type: ignore[import-not-found]
except ImportError:
    sf = None

import wave

# 导入流式 ASR
try:
    import sys
    current_file = Path(__file__).resolve()
    streaming_asr_path = current_file.parent.parent / "test_auto" / "streaming_asr.py"
    
    if streaming_asr_path.exists():
        test_auto_dir = str(streaming_asr_path.parent)
        if test_auto_dir not in sys.path:
            sys.path.insert(0, test_auto_dir)
        intime_voice_dir = str(current_file.parent)
        if intime_voice_dir not in sys.path:
            sys.path.insert(0, intime_voice_dir)
        from streaming_asr import StreamingASR
    else:
        StreamingASR = None
except ImportError:
    StreamingASR = None

# 支持直接运行和作为模块运行
try:
    from .voice_translator import voice_translator, translate_text, tts
except ImportError:
    from voice_translator import voice_translator, translate_text, tts

AudioInput = Union[str, Path, bytes]


class StreamingLocalTranslator:
    """流式本地翻译器：持续监听麦克风，实时翻译并输出到虚拟麦克风"""
    
    DEFAULT_SAMPLE_RATE = 16_000
    DEFAULT_CHUNK_SIZE = 3200  # ~0.2秒 @ 16kHz
    
    def __init__(
        self,
        source_language: str = "chinese",
        target_language: str = "english",
        input_device: Optional[Union[int, str]] = None,
        virtual_mic_device: Optional[Union[int, str]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
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
        self.asr_sample_rate = self.DEFAULT_SAMPLE_RATE  # ASR 需要的采样率（16000 Hz）
        
        # 流式 ASR
        self.streaming_asr = None
        
        # 翻译状态
        self.is_translating = False
        self.translation_queue = queue.Queue()  # 待翻译的文本队列
        
        # 音频流
        self.stream = None
        self.is_running = False
        
        # 虚拟麦克风输出队列
        self.virtual_mic_queue = queue.Queue(maxsize=10)
        self.virtual_mic_stream = None
        self.virtual_mic_output_queue = queue.Queue(maxsize=100)
        self.virtual_mic_stream_lock = threading.Lock()
        self.virtual_mic_sample_rate = None
        self.virtual_mic_channels = None
        
        # 去重机制：记录最近翻译的文本和时间戳，避免重复翻译
        self.recent_translations = {}  # {text_hash: timestamp}
        self.deduplication_window = 10.0  # 10秒内的相同内容不重复翻译
    
    def _file_lock(self, file, exclusive=False):
        """跨平台文件锁"""
        import sys
        if sys.platform == 'win32':
            import msvcrt
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
            except IOError:
                pass
        else:
            import fcntl
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(file.fileno(), lock_type)
    
    def _file_unlock(self, file):
        """跨平台文件解锁"""
        import sys
        if sys.platform == 'win32':
            import msvcrt
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            except IOError:
                pass
        else:
            import fcntl
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
    
    def _detect_language(self, text: str) -> Optional[str]:
        """检测文本语言，返回语言代码（如 'zh', 'en'）"""
        if not text or not text.strip():
            return None
        
        try:
            from googletrans import Translator
            translator = Translator()
            detected = translator.detect(text)
            lang_code = detected.lang.lower()
            print(f"[语言检测] 检测到语言: {lang_code} (置信度: {detected.confidence:.2f})")
            return lang_code
        except Exception as e:
            print(f"[语言检测] 检测失败: {e}，使用启发式方法")
            # 如果 googletrans 失败，使用简单的启发式方法
            # 检查是否包含中文字符
            import re
            if re.search(r'[\u4e00-\u9fff]', text):
                return 'zh'
            # 检查是否主要是英文字符
            if re.search(r'^[a-zA-Z\s\.,!?;:\'"]+$', text):
                return 'en'
            return None
    
    def _is_duplicate(self, text: str) -> bool:
        """检查文本是否在去重窗口内重复"""
        import hashlib
        current_time = time.time()
        
        # 创建文本的哈希值（用于快速比较）
        text_normalized = text.strip().lower()
        text_hash = hashlib.md5(text_normalized.encode('utf-8')).hexdigest()
        
        # 清理过期的记录（超过去重窗口的记录）
        expired_keys = [
            key for key, timestamp in self.recent_translations.items()
            if current_time - timestamp > self.deduplication_window
        ]
        for key in expired_keys:
            del self.recent_translations[key]
        
        # 检查是否在去重窗口内
        if text_hash in self.recent_translations:
            last_time = self.recent_translations[text_hash]
            elapsed = current_time - last_time
            if elapsed < self.deduplication_window:
                print(f"[去重] 检测到重复内容（{elapsed:.1f}秒前翻译过），跳过")
                return True
        
        # 更新记录
        self.recent_translations[text_hash] = current_time
        return False
    
    def _on_sentence_complete(self, text: str):
        """当检测到句子完成时调用"""
        if not text or not text.strip() or self.is_translating:
            return
        
        print(f"\n[本地翻译] 检测到句子完成: {text}")
        
        # 优先检查：是否是扬声器翻译器播放过的内容（防止将播放的翻译音频发送回去）
        if self._check_if_played_by_speaker(text):
            return
        
        # 将文本加入翻译队列（直接翻译，不进行去重和语言检测）
        try:
            self.translation_queue.put_nowait(text)
        except queue.Full:
            print("[本地翻译] 翻译队列已满，跳过本次翻译")
    
    def _translate_and_output(self, text: str):
        """翻译文本并输出到虚拟麦克风（在后台线程中执行）"""
        try:
            self.is_translating = True
            
            # 步骤1: 翻译
            print(f"[本地翻译] 开始翻译: {text}")
            translated_text = translate_text(text, self.source_language, self.target_language)
            print(f"[本地翻译] 翻译结果: {translated_text}")
            
            if not translated_text or not translated_text.strip():
                print("[本地翻译] 翻译结果为空，跳过")
                return
            
            # 步骤2: TTS
            print(f"[本地翻译] 开始 TTS...")
            audio_bytes = tts(translated_text, self.target_language)
            print(f"[本地翻译] TTS 完成，音频大小: {len(audio_bytes)} 字节")
            
            # 步骤3: 在输出到虚拟麦克风之前，先记录翻译结果到共享文件
            # 这样在线翻译器在检测到相同文本时能够立即识别并跳过
            self._record_translation_for_filtering(text, translated_text)
            
            # 步骤4: 输出到虚拟麦克风
            if self.virtual_mic_device is not None:
                try:
                    self.virtual_mic_queue.put_nowait(audio_bytes)
                    print(f"[本地翻译] ✓ 音频已加入虚拟麦克风队列")
                except queue.Full:
                    print("[本地翻译] 虚拟麦克风队列已满，跳过输出")
        except Exception as e:
            print(f"[本地翻译失败] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_translating = False
    
    def _check_if_played_by_speaker(self, text: str) -> bool:
        """检查文本是否是扬声器翻译器播放过的内容（防止将播放的翻译音频发送回去）"""
        try:
            import json
            import hashlib
            from pathlib import Path
            import re
            import sys
            
            shared_file = Path.home() / ".intime_voice_translations.json"
            if not shared_file.exists():
                return False
            
            # 使用文件锁避免并发读写问题
            with open(shared_file, 'r', encoding='utf-8') as f:
                try:
                    self._file_lock(f, exclusive=False)  # 共享锁
                    translations = json.load(f)
                finally:
                    self._file_unlock(f)
            
            current_time = time.time()
            text_normalized = text.strip().lower()
            text_hash = hashlib.md5(text_normalized.encode('utf-8')).hexdigest()
            
            # 检查文本是否在共享记录中（作为翻译结果）
            # 遍历所有记录，查找匹配的文本
            for key, data in translations.items():
                # 只检查来源是扬声器的记录
                if data.get('source') != 'speaker':
                    continue
                
                original = data.get('original', '').strip().lower()
                translated = data.get('translated', '').strip().lower()
                record_time = data.get('timestamp', 0)
                elapsed = current_time - record_time
                
                if elapsed >= 30.0:
                    continue  # 跳过过期记录
                
                # 精确匹配
                if translated == text_normalized or original == text_normalized:
                    print(f"[过滤] 检测到文本是扬声器翻译器播放过的内容（{elapsed:.1f}秒前），跳过以避免发送回去")
                    return True
                
                # 模糊匹配：移除标点符号和空格后比较
                text_normalized_no_punct = re.sub(r'[^\w\s]', '', text_normalized)
                translated_no_punct = re.sub(r'[^\w\s]', '', translated)
                original_no_punct = re.sub(r'[^\w\s]', '', original)
                
                if translated_no_punct == text_normalized_no_punct or original_no_punct == text_normalized_no_punct:
                    print(f"[过滤] 检测到文本是扬声器翻译器播放过的内容（模糊匹配，{elapsed:.1f}秒前），跳过以避免发送回去")
                    return True
                
                # 部分匹配：检查文本是否包含在翻译结果中，或翻译结果是否包含在文本中
                # 这样可以处理 "吃一大碗。" vs "吃一大碗米。" 的情况
                # 使用简单的启发式方法检查是否是中文（包含中文字符）
                import re
                is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
                if is_chinese:
                    # 如果文本是中文，检查是否是翻译结果的一部分
                    if translated_no_punct and text_normalized_no_punct:
                        # 检查文本是否包含翻译结果的主要部分（至少80%匹配）
                        if len(translated_no_punct) > 0 and len(text_normalized_no_punct) > 0:
                            # 计算相似度（简单的字符匹配）
                            common_chars = set(translated_no_punct) & set(text_normalized_no_punct)
                            similarity = len(common_chars) / max(len(set(translated_no_punct)), len(set(text_normalized_no_punct)), 1)
                            if similarity > 0.8:
                                print(f"[过滤] 检测到文本与扬声器播放内容高度相似（相似度: {similarity:.2f}，{elapsed:.1f}秒前），跳过以避免发送回去")
                                return True
                        
                        # 检查是否是子串匹配（一个包含另一个）
                        if translated_no_punct in text_normalized_no_punct or text_normalized_no_punct in translated_no_punct:
                            if abs(len(translated_no_punct) - len(text_normalized_no_punct)) <= 3:  # 允许3个字符的差异
                                print(f"[过滤] 检测到文本与扬声器播放内容部分匹配（{elapsed:.1f}秒前），跳过以避免发送回去")
                                return True
                
                # 也检查哈希值匹配
                translated_hash = hashlib.md5(translated.encode('utf-8')).hexdigest()
                original_hash = hashlib.md5(original.encode('utf-8')).hexdigest()
                if text_hash == translated_hash or text_hash == original_hash:
                    print(f"[过滤] 检测到文本是扬声器翻译器播放过的内容（哈希匹配，{elapsed:.1f}秒前），跳过以避免发送回去")
                    return True
        except Exception as e:
            # 如果检查失败，继续正常流程
            print(f"[过滤] 检查失败: {e}")
        
        return False
    
    def _record_translation_for_filtering(self, original_text: str, translated_text: str):
        """记录翻译结果到共享文件，用于在线翻译器识别并跳过"""
        try:
            import json
            import hashlib
            from pathlib import Path
            
            # 创建共享文件路径
            shared_file = Path.home() / ".intime_voice_translations.json"
            
            # 使用文件锁避免并发读写问题
            translations = {}
            if shared_file.exists():
                try:
                    with open(shared_file, 'r', encoding='utf-8') as f:
                        try:
                            self._file_lock(f, exclusive=True)  # 排他锁
                            translations = json.load(f)
                        finally:
                            self._file_unlock(f)
                except Exception:
                    translations = {}
            
            # 记录翻译结果
            current_time = time.time()
            original_normalized = original_text.strip().lower()
            translated_normalized = translated_text.strip().lower()
            original_hash = hashlib.md5(original_normalized.encode('utf-8')).hexdigest()
            translated_hash = hashlib.md5(translated_normalized.encode('utf-8')).hexdigest()
            
            translations[original_hash] = {
                'original': original_normalized,
                'translated': translated_normalized,
                'translated_hash': translated_hash,
                'timestamp': current_time
            }
            translations[translated_hash] = {
                'original': original_normalized,
                'translated': translated_normalized,
                'original_hash': original_hash,
                'timestamp': current_time
            }
            
            # 清理过期记录（超过30秒）
            expired_keys = [
                key for key, data in translations.items()
                if current_time - data.get('timestamp', 0) > 30.0
            ]
            for key in expired_keys:
                del translations[key]
            
            # 保存到文件
            with open(shared_file, 'w', encoding='utf-8') as f:
                try:
                    self._file_lock(f, exclusive=True)  # 排他锁
                    json.dump(translations, f, ensure_ascii=False, indent=2)
                    f.flush()  # 确保立即写入
                finally:
                    self._file_unlock(f)
            
            print(f"[本地翻译] ✓ 已记录翻译结果到共享文件: {original_normalized} → {translated_normalized}")
        except Exception as e:
            # 如果记录失败，不影响主流程
            print(f"[本地翻译] 记录翻译结果失败: {e}")
        except Exception as e:
            print(f"[本地翻译失败] {e}")
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
                padding = np.zeros((frames - len(chunk), target_channels), dtype=np.float32)
                outdata[:] = np.vstack([chunk, padding])
            elif len(chunk) > frames:
                outdata[:] = chunk[:frames]
            else:
                outdata[:] = chunk
        except queue.Empty:
            outdata.fill(0)
    
    def _output_virtual_mic_worker(self):
        """虚拟麦克风输出工作线程：处理音频并加入播放队列"""
        if self.virtual_mic_device is None:
            return
        
        print(f"[虚拟麦克风] 输出工作线程已启动，等待音频数据...")
        
        while self.is_running:
            try:
                audio_bytes = self.virtual_mic_queue.get(timeout=0.1)
                print(f"[虚拟麦克风] 收到音频数据，大小: {len(audio_bytes)} 字节")
                
                # 读取音频数据
                use_soundfile = sf is not None
                if use_soundfile:
                    try:
                        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                        channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
                        if audio_data.ndim == 1:
                            audio_float = audio_data.reshape(-1, 1).astype(np.float32)
                        else:
                            audio_float = audio_data.astype(np.float32)
                    except Exception:
                        use_soundfile = False  # 失败后使用 wave
                
                if not use_soundfile:
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
                    max_output_channels = device_info.get('max_output_channels', 0)
                    max_input_channels = device_info.get('max_input_channels', 0)
                    
                    # 检查设备是否有输出通道（必须是输出设备）
                    if max_output_channels == 0:
                        print(f"[虚拟麦克风] 错误: 设备 {self.virtual_mic_device} ({device_info.get('name', 'Unknown')}) 没有输出通道！")
                        print(f"[虚拟麦克风] 这是一个输入设备（max_input_channels={max_input_channels}），不能用于输出音频")
                        print(f"[虚拟麦克风] 请使用 VB-Audio Point Input（索引 68）作为输出设备，而不是输入设备")
                        continue  # 跳过这个音频，等待下一个
                    
                    # 确定目标声道数：优先使用设备支持的最大声道数（通常是2）
                    if max_output_channels >= 2:
                        target_channels = 2  # 强制使用立体声
                    elif max_output_channels == 1:
                        target_channels = 1  # 单声道
                    else:
                        # 如果设备信息不明确，默认使用2声道（大多数虚拟音频设备需要立体声）
                        target_channels = 2
                        print(f"[虚拟麦克风] 警告: 无法确定设备声道数，默认使用2声道")
                    
                    # 调整声道数
                    output_audio = audio_float
                    if channels > target_channels:
                        if target_channels == 1:
                            # 多声道转单声道：取平均值
                            output_audio = np.mean(audio_float, axis=1, keepdims=True)
                        else:
                            # 多声道转立体声：取前两个声道
                            output_audio = audio_float[:, :target_channels]
                    elif channels < target_channels:
                        if target_channels == 2 and channels == 1:
                            # 单声道转立体声：复制到两个声道
                            output_audio = np.repeat(audio_float, 2, axis=1)
                        else:
                            # 其他情况：保持原样
                            output_audio = audio_float
                    
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
                    
                    # 保存采样率和声道数
                    self.virtual_mic_sample_rate = sample_rate
                    self.virtual_mic_channels = target_channels
                    
                    # 如果流不存在，创建新的流
                    if self.virtual_mic_stream is None:
                        print(f"[虚拟麦克风] 创建输出流: {sample_rate} Hz, {target_channels} 声道")
                        print(f"[虚拟麦克风] 设备信息: {device_info.get('name', 'Unknown')}, 最大输出声道: {max_output_channels}")
                        try:
                            # 确保声道数有效
                            if target_channels < 1:
                                target_channels = 1
                            if target_channels > max_output_channels and max_output_channels > 0:
                                target_channels = max_output_channels
                                print(f"[虚拟麦克风] 调整声道数为设备支持的最大值: {target_channels}")
                            
                            self.virtual_mic_stream = sd.OutputStream(
                                device=self.virtual_mic_device,
                                samplerate=sample_rate,
                                channels=target_channels,
                                dtype='float32',
                                callback=self._virtual_mic_stream_callback,
                                blocksize=int(sample_rate * 0.05),
                            )
                            self.virtual_mic_stream.start()
                            print(f"[虚拟麦克风] ✓ 输出流已创建并启动 (设备: {device_info.get('name', 'Unknown')}, {target_channels} 声道)")
                        except Exception as e:
                            print(f"[虚拟麦克风] 创建输出流失败: {e}")
                            print(f"[虚拟麦克风] 尝试信息: 采样率={sample_rate}, 声道数={target_channels}, 设备={self.virtual_mic_device}")
                            print(f"[虚拟麦克风] 设备详细信息: {device_info}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    print(f"[虚拟麦克风] 处理音频: {len(output_audio)} 样本, {target_channels} 声道, {sample_rate} Hz")
                    
                    # 将音频分块加入输出队列
                    chunk_size = int(sample_rate * 0.05)
                    chunks_put = 0
                    for i in range(0, len(output_audio), chunk_size):
                        chunk = output_audio[i:i+chunk_size]
                        try:
                            self.virtual_mic_output_queue.put(chunk.astype(np.float32), timeout=0.5)
                            chunks_put += 1
                        except queue.Full:
                            print(f"[虚拟麦克风] 输出队列已满，已放入 {chunks_put} 块")
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
        
        if np is not None:
            volume = np.linalg.norm(indata) * 10
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
            self.streaming_asr.send_audio_chunk(audio_bytes)
    
    def start(self):
        """启动流式翻译"""
        if self.is_running:
            return
        
        print("=" * 60)
        print("启动本地流式翻译服务")
        print(f"源语言: {self.source_language}")
        print(f"目标语言: {self.target_language}")
        print(f"输入设备: {self.input_device or '默认'}")
        if self.virtual_mic_device is not None:
            try:
                device_info = sd.query_devices(self.virtual_mic_device)
                print(f"虚拟麦克风设备: {device_info['name']} (索引: {self.virtual_mic_device})")
            except Exception:
                print(f"虚拟麦克风设备: {self.virtual_mic_device}")
        print("=" * 60)
        print("持续监听麦克风，检测到句子完成时自动翻译")
        print("按 Ctrl+C 停止")
        print()
        if self.virtual_mic_device is not None:
            print("⚠️  重要提示：")
            print("   虚拟麦克风输出到虚拟音频设备，不会播放到扬声器")
            print("   如果听到麦克风翻译在扬声器播放，请检查系统配置：")
            print("   1. Windows 设置 → 系统 → 声音 → 输入 → 选择虚拟音频设备（如 VB-CABLE）")
            print("   2. Windows 设置 → 系统 → 声音 → 输出 → 选择您的扬声器")
            print("   3. 确保虚拟音频设备的输出不会路由到扬声器")
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
        
        self.is_running = True
        translation_thread = threading.Thread(target=self._process_translation_queue, daemon=True)
        translation_thread.start()
        
        # 启动虚拟麦克风输出线程
        if self.virtual_mic_device is not None:
            print(f"[虚拟麦克风] 启动虚拟麦克风输出线程...")
            virtual_mic_thread = threading.Thread(target=self._output_virtual_mic_worker, daemon=True)
            virtual_mic_thread.start()
        
        try:
            if self.input_device is not None:
                try:
                    device_info = sd.query_devices(self.input_device)
                    if device_info['max_input_channels'] == 0:
                        print(f"⚠️  警告: 设备 {self.input_device} 没有输入通道，尝试使用系统默认设备...")
                        self.input_device = None
                    else:
                        print(f"[音频输入] 使用设备: {device_info['name']} (索引: {self.input_device})")
                except Exception as e:
                    print(f"⚠️  警告: 无法查询设备 {self.input_device}: {e}")
                    print("尝试使用系统默认输入设备...")
                    self.input_device = None
            
            print(f"[音频输入] 启动音频流...")
            
            desired_sample_rate = self.sample_rate
            actual_sample_rate = desired_sample_rate
            try:
                sd.check_input_settings(
                    device=self.input_device,
                    samplerate=desired_sample_rate,
                    channels=1,
                    dtype='float32',
                )
                print(f"[音频输入] 设备支持 {desired_sample_rate} Hz，保持与 ASR 一致的采样率")
            except Exception as check_err:
                fallback_rate = None
                if self.input_device is not None:
                    try:
                        device_info = sd.query_devices(self.input_device)
                        fallback_rate = device_info.get('default_samplerate')
                    except Exception as query_err:
                        print(f"[音频输入] 查询设备采样率失败: {query_err}，继续尝试 {desired_sample_rate} Hz")
                if fallback_rate:
                    actual_sample_rate = int(fallback_rate)
                    print(
                        f"[音频输入] ⚠️  设备不支持 {desired_sample_rate} Hz ({check_err})，"
                        f"改用默认采样率 {actual_sample_rate} Hz"
                    )
                else:
                    print(
                        f"[音频输入] ⚠️  无法确认设备是否支持 {desired_sample_rate} Hz ({check_err})，"
                        "仍尝试使用该采样率"
                    )
            
            print(f"  采样率: {actual_sample_rate} Hz")
            print(f"  设备: {self.input_device or '系统默认'}")
            
            # 更新采样率（如果改变了）
            if actual_sample_rate != self.sample_rate:
                print(f"[音频输入] ⚠️  采样率已从 {self.sample_rate} Hz 调整为 {actual_sample_rate} Hz 以匹配设备")
                self.sample_rate = actual_sample_rate
                # 更新块大小以适应新的采样率
                self.DEFAULT_CHUNK_SIZE = int(self.sample_rate * 0.2)
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.input_device,
                callback=self.audio_callback,
                blocksize=self.DEFAULT_CHUNK_SIZE,
            )
            
            self.stream.start()
            print(f"[音频输入] ✓ 音频流已启动，正在监听麦克风...")
            
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


def translate_local_voice(
    voice: AudioInput,
    *,
    local_lang: str = "chinese",
    target_lang: str = "english",
    language_hint: Optional[str] = None,
) -> bytes:
    """
    翻译本地录制的音频到目标语言（非流式，一次性处理）。
    
    Returns:
        翻译后的音频字节数据
    """
    return voice_translator(
        voice,
        source_language=local_lang,
        target_language=target_lang,
    )
