"""Workflow helpers for translating audio from speaker output (online mode)."""

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
    from .voice_translator import asr_transcribe, translate_text, tts
except ImportError:
    from voice_translator import asr_transcribe, translate_text, tts

AudioInput = Union[str, Path, bytes]


class StreamingOnlineTranslator:
    """流式在线翻译器：持续监听扬声器输出，实时翻译并播放"""
    
    DEFAULT_SAMPLE_RATE = 16_000
    DEFAULT_CHUNK_SIZE = 3200  # ~0.2秒 @ 16kHz
    
    def __init__(
        self,
        source_language: str = "english",
        local_lang: str = "chinese",
        input_device: Optional[Union[int, str]] = None,  # 扬声器输出捕获设备（通常是虚拟音频设备的输入通道）
        output_device: Optional[Union[int, str]] = None,  # 播放设备（扬声器）
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
        self.local_lang = local_lang
        self.input_device = input_device  # 用于捕获扬声器输出的设备
        self.output_device = output_device  # 用于播放的设备
        self.sample_rate = sample_rate
        self.asr_sample_rate = 16_000  # ASR 需要的采样率（固定为 16000 Hz）
        
        # 流式 ASR
        self.streaming_asr = None
        
        # 翻译状态
        self.is_translating = False
        self.translation_queue = queue.Queue()  # 待翻译的文本队列
        
        # 音频流
        self.stream = None
        self.is_running = False
        
        # 播放队列
        self.playback_queue = queue.Queue(maxsize=10)
        
        # 去重机制：记录最近翻译的文本和时间戳，避免重复翻译
        self.recent_translations = {}  # {text_hash: timestamp}
        self.deduplication_window = 10.0  # 10秒内的相同内容不重复翻译
        self.last_playback_time = 0.0  # 上次播放的时间
        self.playback_cooldown = 5.0  # 播放后5秒内不处理新音频（避免捕获自己播放的内容）
    
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
        
        # 记录最近播放的翻译文本，用于识别并跳过来自虚拟麦克风的音频
        self.recent_played_texts = {}  # {text_hash: timestamp}
        self.played_text_window = 20.0  # 20秒内播放过的文本，如果再次出现则跳过（可能是来自虚拟麦克风）
        
        # 记录最近播放的英文原文，用于识别来自虚拟麦克风的英文音频
        self.recent_played_english = {}  # {english_text_hash: chinese_translation_hash}
    
    def _detect_language(self, text: str) -> Optional[str]:
        """快速语言检测：包含中文则判为中文，否则默认英文"""
        if not text or not text.strip():
            return None
        
        import re
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        return 'en'
    
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
        
        current_time = time.time()
        
        # 检查播放冷却时间：播放后一段时间内不处理新音频（避免捕获自己播放的内容）
        if current_time - self.last_playback_time < self.playback_cooldown:
            elapsed = current_time - self.last_playback_time
            print(f"[冷却] 播放冷却中（{elapsed:.1f}秒/{self.playback_cooldown}秒），跳过本次处理")
            return
        
        print(f"\n[在线翻译] 检测到句子完成: {text}")
        
        # 优先检查共享文件：判断是否是本地翻译器的翻译结果（最优先，避免不必要的处理）
        # 这样可以识别来自虚拟麦克风的音频，即使它们通过同一个 BlackHole 设备
        if self._check_shared_translation_file(text):
            return
        
        # 去重检查：避免短时间内重复翻译相同内容
        if self._is_duplicate(text):
            return
        
        # 检查是否是最近播放过的内容（可能是来自虚拟麦克风的音频）
        if self._is_recently_played(text):
            return
        
        # 检测语言：如果是中文，跳过翻译和播放（避免循环）
        detected_lang = self._detect_language(text)
        if detected_lang:
            # 检查是否是中文（包括 zh, zh-cn, zh-tw 等）
            if detected_lang.startswith('zh'):
                print(f"[在线翻译] 检测到中文内容，跳过翻译和播放（避免循环）")
                return
            # 检查是否是英文
            if detected_lang.startswith('en'):
                print(f"[在线翻译] 检测到英文内容，将进行翻译")
            else:
                print(f"[在线翻译] 检测到其他语言 ({detected_lang})，将进行翻译")
        else:
            # 如果无法检测语言，使用启发式方法：如果包含中文字符，跳过
            import re
            if re.search(r'[\u4e00-\u9fff]', text):
                print(f"[在线翻译] 文本包含中文字符，跳过翻译和播放（避免循环）")
                return
            print(f"[在线翻译] 无法检测语言，将尝试翻译")
        
        # 将文本加入翻译队列
        try:
            self.translation_queue.put_nowait(text)
        except queue.Full:
            print("[在线翻译] 翻译队列已满，跳过本次翻译")
    
    def _check_shared_translation_file(self, text: str) -> bool:
        """检查共享文件，判断文本是否是本地翻译器的翻译结果"""
        try:
            import json
            import hashlib
            from pathlib import Path
            
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
            
            # 检查文本是否在共享记录中
            if text_hash in translations:
                data = translations[text_hash]
                record_time = data.get('timestamp', 0)
                elapsed = current_time - record_time
                if elapsed < 30.0:  # 30秒内的记录都有效
                    print(f"[过滤] 检测到文本是本地翻译器的翻译结果（{elapsed:.1f}秒前），跳过以避免循环")
                    return True
            
            # 如果文本是英文，检查对应的中文翻译是否在记录中
            detected_lang = self._detect_language(text)
            if detected_lang and detected_lang.startswith('en'):
                # 遍历所有记录，查找匹配的英文文本
                for key, data in translations.items():
                    original = data.get('original', '').strip().lower()
                    translated = data.get('translated', '').strip().lower()
                    
                    # 精确匹配
                    if original == text_normalized or translated == text_normalized:
                        record_time = data.get('timestamp', 0)
                        elapsed = current_time - record_time
                        if elapsed < 30.0:
                            print(f"[过滤] 检测到英文文本是本地翻译器的翻译结果（{elapsed:.1f}秒前），跳过以避免循环")
                            return True
                    
                    # 模糊匹配：移除标点符号和空格后比较（处理 "Eat McDonald's today." vs "Eat McDonalds today." 的情况）
                    import re
                    text_normalized_no_punct = re.sub(r'[^\w\s]', '', text_normalized)
                    original_no_punct = re.sub(r'[^\w\s]', '', original)
                    translated_no_punct = re.sub(r'[^\w\s]', '', translated)
                    
                    if original_no_punct == text_normalized_no_punct or translated_no_punct == text_normalized_no_punct:
                        record_time = data.get('timestamp', 0)
                        elapsed = current_time - record_time
                        if elapsed < 30.0:
                            print(f"[过滤] 检测到英文文本是本地翻译器的翻译结果（模糊匹配，{elapsed:.1f}秒前），跳过以避免循环")
                            return True
                    
                    # 也检查哈希值匹配
                    original_hash = hashlib.md5(original.encode('utf-8')).hexdigest()
                    translated_hash = hashlib.md5(translated.encode('utf-8')).hexdigest()
                    if text_hash == original_hash or text_hash == translated_hash:
                        record_time = data.get('timestamp', 0)
                        elapsed = current_time - record_time
                        if elapsed < 30.0:
                            print(f"[过滤] 检测到文本是本地翻译器的翻译结果（{elapsed:.1f}秒前），跳过以避免循环")
                            return True
        except Exception:
            # 如果读取共享文件失败，继续正常流程
            pass
        
        return False
    
    def _record_played_translation_to_shared_file(self, original_text: str, translated_text: str):
        """记录播放的翻译结果到共享文件，用于本地翻译器识别并跳过（防止将播放的翻译音频发送回去）"""
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
            
            # 记录翻译结果（注意：这里记录的是扬声器播放的翻译，original是英文，translated是中文）
            current_time = time.time()
            original_normalized = original_text.strip().lower()
            translated_normalized = translated_text.strip().lower()
            original_hash = hashlib.md5(original_normalized.encode('utf-8')).hexdigest()
            translated_hash = hashlib.md5(translated_normalized.encode('utf-8')).hexdigest()
            
            # 记录：original是英文原文，translated是中文翻译结果（扬声器播放的内容）
            translations[original_hash] = {
                'original': original_normalized,
                'translated': translated_normalized,
                'translated_hash': translated_hash,
                'timestamp': current_time,
                'source': 'speaker'  # 标记来源是扬声器翻译器
            }
            translations[translated_hash] = {
                'original': original_normalized,
                'translated': translated_normalized,
                'original_hash': original_hash,
                'timestamp': current_time,
                'source': 'speaker'  # 标记来源是扬声器翻译器
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
        except Exception as e:
            # 如果记录失败，不影响主流程
            pass
    
    def _is_recently_played(self, text: str) -> bool:
        """检查文本是否是最近播放过的翻译结果（可能是来自虚拟麦克风的音频）"""
        import hashlib
        import json
        from pathlib import Path
        current_time = time.time()
        
        # 清理过期的记录
        expired_keys = [
            key for key, timestamp in self.recent_played_texts.items()
            if current_time - timestamp > self.played_text_window
        ]
        for key in expired_keys:
            del self.recent_played_texts[key]
        
        # 清理过期的英文记录
        expired_english_keys = [
            key for key in self.recent_played_english.keys()
            if key not in self.recent_played_texts  # 如果对应的中文翻译已过期，英文记录也过期
        ]
        for key in expired_english_keys:
            del self.recent_played_english[key]
        
        # 检查原始文本是否在最近播放的记录中
        text_normalized = text.strip().lower()
        text_hash = hashlib.md5(text_normalized.encode('utf-8')).hexdigest()
        
        if text_hash in self.recent_played_texts:
            last_time = self.recent_played_texts[text_hash]
            elapsed = current_time - last_time
            if elapsed < self.played_text_window:
                print(f"[过滤] 检测到最近播放过的内容（{elapsed:.1f}秒前），可能是来自虚拟麦克风，跳过")
                return True
        
        # 检查共享文件：读取本地翻译器记录的翻译结果
        try:
            shared_file = Path.home() / ".intime_voice_translations.json"
            if shared_file.exists():
                with open(shared_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                
                # 检查文本是否在共享记录中
                if text_hash in translations:
                    data = translations[text_hash]
                    record_time = data.get('timestamp', 0)
                    elapsed = current_time - record_time
                    if elapsed < 30.0:  # 30秒内的记录都有效
                        print(f"[过滤] 检测到文本是本地翻译器的翻译结果（{elapsed:.1f}秒前），跳过以避免循环")
                        return True
                
                # 如果文本是英文，检查对应的中文翻译是否在记录中
                detected_lang = self._detect_language(text)
                if detected_lang and detected_lang.startswith('en'):
                    # 检查英文文本对应的中文翻译是否在记录中
                    for key, data in translations.items():
                        if data.get('original_hash') == text_hash or data.get('translated_hash') == text_hash:
                            record_time = data.get('timestamp', 0)
                            elapsed = current_time - record_time
                            if elapsed < 30.0:
                                print(f"[过滤] 检测到英文文本是本地翻译器的翻译结果（{elapsed:.1f}秒前），跳过以避免循环")
                                return True
        except Exception:
            # 如果读取共享文件失败，继续正常流程
            pass
        
        # 如果文本是英文，检查是否在最近播放的英文记录中
        # 这样可以快速识别来自虚拟麦克风的英文音频（无需翻译）
        detected_lang = self._detect_language(text)
        if detected_lang and detected_lang.startswith('en'):
            # 检查英文文本是否在最近播放的英文记录中
            if text_hash in self.recent_played_english:
                chinese_hash = self.recent_played_english[text_hash]
                if chinese_hash in self.recent_played_texts:
                    last_time = self.recent_played_texts[chinese_hash]
                    elapsed = current_time - last_time
                    if elapsed < self.played_text_window:
                        print(f"[过滤] 检测到英文文本是最近播放过的翻译结果（{elapsed:.1f}秒前），可能是来自虚拟麦克风，跳过")
                        return True
            
            # 如果不在记录中，尝试翻译成中文，检查翻译结果是否是最近播放过的
            try:
                from .voice_translator import translate_text
                translated = translate_text(text, "english", "chinese")
                if translated:
                    translated_hash = hashlib.md5(translated.strip().lower().encode('utf-8')).hexdigest()
                    if translated_hash in self.recent_played_texts:
                        last_time = self.recent_played_texts[translated_hash]
                        elapsed = current_time - last_time
                        if elapsed < self.played_text_window:
                            # 记录这个英文文本和对应的中文翻译，以便下次快速识别
                            self.recent_played_english[text_hash] = translated_hash
                            print(f"[过滤] 检测到英文文本对应的中文翻译是最近播放过的（{elapsed:.1f}秒前），可能是来自虚拟麦克风，跳过")
                            return True
            except Exception as e:
                # 如果翻译失败，继续处理
                pass
        
        return False
    
    def _translate_and_play(self, text: str):
        """翻译文本并播放（在后台线程中执行）"""
        try:
            self.is_translating = True
            
            # 步骤1: 翻译
            print(f"[在线翻译] 开始翻译: {text}")
            translated_text = translate_text(text, self.source_language, self.local_lang)
            print(f"[在线翻译] 翻译结果: {translated_text}")
            
            if not translated_text or not translated_text.strip():
                print("[在线翻译] 翻译结果为空，跳过")
                return
            
            # 记录翻译后的文本和原始文本（用于识别来自虚拟麦克风的音频）
            import hashlib
            translated_hash = hashlib.md5(translated_text.strip().lower().encode('utf-8')).hexdigest()
            original_hash = hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()
            current_time = time.time()
            self.recent_played_texts[translated_hash] = current_time  # 记录翻译结果（中文）
            self.recent_played_texts[original_hash] = current_time  # 记录原始文本（英文）
            # 记录英文文本和对应的中文翻译的映射关系，用于快速识别
            self.recent_played_english[original_hash] = translated_hash
            
            # 记录到共享文件，让本地翻译器能够识别并跳过（防止将播放的翻译音频发送回去）
            self._record_played_translation_to_shared_file(text, translated_text)
            
            # 步骤2: TTS
            print(f"[在线翻译] 开始 TTS...")
            audio_bytes = tts(translated_text, self.local_lang)
            print(f"[在线翻译] TTS 完成，音频大小: {len(audio_bytes)} 字节")
            
            # 步骤3: 加入播放队列
            try:
                self.playback_queue.put_nowait(audio_bytes)
                print(f"[在线翻译] ✓ 音频已加入播放队列")
            except queue.Full:
                print("[在线翻译] 播放队列已满，跳过播放")
        except Exception as e:
            print(f"[在线翻译失败] {e}")
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
                    target=self._translate_and_play,
                    args=(text,),
                    daemon=True
                ).start()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[翻译队列处理错误] {e}")
    
    def _playback_worker(self):
        """播放工作线程"""
        while self.is_running:
            try:
                audio_bytes = self.playback_queue.get(timeout=0.1)
                print(f"[播放] 收到音频数据，大小: {len(audio_bytes)} 字节")
                self._play_audio_bytes(audio_bytes)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[播放错误] {e}")
                import traceback
                traceback.print_exc()
    
    def _validate_output_device(self, device) -> bool:
        """验证输出设备是否有效且可用"""
        if device is None:
            return True  # None 表示使用默认设备，总是有效
        try:
            device_info = sd.query_devices(device)
            # 检查设备是否有输出通道
            if device_info.get('max_output_channels', 0) == 0:
                print(f"[播放] ⚠️  警告: 设备 {device} ({device_info.get('name', 'Unknown')}) 没有输出通道")
                return False
            return True
        except Exception as e:
            print(f"[播放] ⚠️  警告: 设备 {device} 无效或不可用: {e}")
            return False
    
    def _get_device_supported_samplerate(self, device, target_rate: int) -> int:
        """获取设备支持的采样率，如果不支持目标采样率，返回设备默认采样率或常见采样率"""
        if device is None:
            # 默认设备，使用目标采样率或常见采样率
            return target_rate if target_rate in [44100, 48000, 16000] else 44100
        
        try:
            device_info = sd.query_devices(device)
            default_rate = device_info.get('default_samplerate', None)
            
            # 如果设备有默认采样率，优先使用（大多数设备支持其默认采样率）
            if default_rate:
                default_rate = int(default_rate)
                # 如果默认采样率与目标采样率相同，直接返回
                if default_rate == target_rate:
                    return default_rate
                # 否则使用设备默认采样率（通常更可靠）
                return default_rate
            
            # 如果没有默认采样率，根据目标采样率选择最接近的常见采样率
            # 常见的采样率列表（按优先级排序）
            common_rates = [48000, 44100, 32000, 24000, 22050, 16000, 11025, 8000]
            
            # 优先选择大于等于目标采样率的
            for rate in common_rates:
                if rate >= target_rate:
                    return rate
            
            # 如果目标采样率大于所有常见采样率，使用最高的
            return 48000
        except Exception:
            # 如果查询失败，使用常见采样率
            # 大多数设备支持 44100 或 48000
            if target_rate >= 24000:
                return 44100  # 对于高采样率，使用 44100
            else:
                return 16000  # 对于低采样率，使用 16000
    
    def _resample_audio(self, audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
        """重采样音频数据"""
        if original_rate == target_rate:
            return audio_data
        
        try:
            from scipy import signal
            # 计算重采样后的样本数
            num_samples = int(len(audio_data) * target_rate / original_rate)
            # 重采样
            if audio_data.ndim == 1:
                resampled = signal.resample(audio_data, num_samples, axis=0)
            else:
                resampled = signal.resample(audio_data, num_samples, axis=0)
            return resampled.astype(np.float32)
        except ImportError:
            # 如果没有 scipy，使用简单的线性插值
            num_samples = int(len(audio_data) * target_rate / original_rate)
            if audio_data.ndim == 1:
                indices = np.linspace(0, len(audio_data) - 1, num_samples)
                resampled = np.interp(indices, np.arange(len(audio_data)), audio_data)
            else:
                resampled = np.zeros((num_samples, audio_data.shape[1]), dtype=audio_data.dtype)
                for ch in range(audio_data.shape[1]):
                    indices = np.linspace(0, len(audio_data) - 1, num_samples)
                    resampled[:, ch] = np.interp(indices, np.arange(len(audio_data)), audio_data[:, ch])
            return resampled.astype(np.float32)
    
    def _play_audio_bytes(self, audio_bytes: bytes) -> None:
        """播放音频字节数据"""
        if sd is None or np is None:
            print("[播放跳过] sounddevice 或 numpy 未安装")
            return
        
        if not audio_bytes:
            print("[播放跳过] 音频数据为空")
            return
        
        # 定义目标安全采样率 (改为 44100 Hz，与虚拟音频设备的常用设置保持一致)
        TARGET_SAMPLE_RATE = 44100
        
        # === 新增：设备有效性检查 ===
        if self.output_device is not None:
            try:
                device_info = sd.query_devices(self.output_device)
                if device_info.get('max_output_channels', 0) == 0:
                    print(f"[播放] ❌ 错误: 指定设备 [{self.output_device}] {device_info['name']} 没有输出通道！")
                    print(" 💡 请运行 'python main.py --list-all-devices' 重新确认扬声器输出设备索引。")
                    return
            except Exception as e:
                if isinstance(e, (sd.PortAudioError, OSError)):
                    print(f"[播放] ❌ 严重错误: 无法查询到设备索引 [{self.output_device}]，索引可能已更改或设备不可用。")
                    print(" 💡 请检查该设备是否已被禁用或被其他应用独占。")
                    return
                else:
                    print(f"[播放] ⚠️  警告: 查询设备时出错: {e}，尝试使用默认设备")
                    self.output_device = None
        # =============================
        
        try:
            # 尝试使用 soundfile 读取（支持多种格式）
            if sf is not None:
                try:
                    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                    
                    # === 重采样逻辑修改：确保目标是 TARGET_SAMPLE_RATE ===
                    if sample_rate != TARGET_SAMPLE_RATE:
                        print(f"[播放] 设备不支持 {sample_rate} Hz，重采样到 {TARGET_SAMPLE_RATE} Hz")
                        
                        try:
                            from scipy.signal import resample
                            num_channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
                            new_length = int(audio_data.shape[0] * TARGET_SAMPLE_RATE / sample_rate)
                            
                            if num_channels == 1:
                                audio_data_resampled = resample(audio_data, new_length)
                            else:
                                audio_data_resampled = np.zeros((new_length, num_channels), dtype=audio_data.dtype)
                                for i in range(num_channels):
                                    audio_data_resampled[:, i] = resample(audio_data[:, i], new_length)
                            
                            audio_data = audio_data_resampled.astype(np.float32)
                            sample_rate = TARGET_SAMPLE_RATE
                            print(f"[播放] ✓ 重采样完成，新采样率: {sample_rate} Hz")
                        except ImportError:
                            print("[播放] ⚠️  警告: scipy 未安装，使用 numpy 线性插值重采样")
                            audio_data = self._resample_audio(audio_data, sample_rate, TARGET_SAMPLE_RATE)
                            sample_rate = TARGET_SAMPLE_RATE
                    # =======================================================
                    
                    channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
                    
                    if audio_data.ndim == 1:
                        audio_float = audio_data.reshape(-1, 1).astype(np.float32)
                    else:
                        audio_float = audio_data.astype(np.float32)
                    
                    audio_float = np.clip(audio_float, -1.0, 1.0)
                    
                    print(f"[播放] 播放音频: {sample_rate} Hz, {channels} 声道")
                    sd.play(audio_float, sample_rate, device=self.output_device)
                    sd.wait()  # 等待播放完成
                    # 更新播放时间（用于冷却机制，避免捕获自己播放的内容）
                    self.last_playback_time = time.time()
                    print("[播放] ✓ 播放完成")
                    return
                except Exception as e:
                    print(f"[播放] soundfile 读取失败，尝试 wave: {e}")
            
            # 使用 wave 读取 WAV 文件
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
                    
                    # === 重采样逻辑修改：确保目标是 TARGET_SAMPLE_RATE ===
                    if sample_rate != TARGET_SAMPLE_RATE:
                        print(f"[播放] 设备不支持 {sample_rate} Hz，重采样到 {TARGET_SAMPLE_RATE} Hz")
                        
                        try:
                            from scipy.signal import resample
                            num_channels = channels
                            new_length = int(len(audio_float) * TARGET_SAMPLE_RATE / sample_rate)
                            
                            if num_channels == 1:
                                audio_float = resample(audio_float.flatten(), new_length).reshape(-1, 1)
                            else:
                                audio_float_resampled = np.zeros((new_length, num_channels), dtype=np.float32)
                                for i in range(num_channels):
                                    audio_float_resampled[:, i] = resample(audio_float[:, i], new_length)
                                audio_float = audio_float_resampled
                            
                            sample_rate = TARGET_SAMPLE_RATE
                            print(f"[播放] ✓ 重采样完成，新采样率: {sample_rate} Hz")
                        except ImportError:
                            print("[播放] ⚠️  警告: scipy 未安装，使用 numpy 线性插值重采样")
                            audio_float = self._resample_audio(audio_float, sample_rate, TARGET_SAMPLE_RATE)
                            sample_rate = TARGET_SAMPLE_RATE
                    # =======================================================
                    
                    print(f"[播放] 播放音频: {sample_rate} Hz, {channels} 声道")
                    sd.play(audio_float, sample_rate, device=self.output_device)
                    sd.wait()  # 等待播放完成
                    # 更新播放时间（用于冷却机制，避免捕获自己播放的内容）
                    self.last_playback_time = time.time()
                    print("[播放] ✓ 播放完成")
                    return
            except wave.Error:
                print("[播放] WAV 解析失败，尝试按 PCM16 播放")
            
            # 最后尝试：按 PCM16 格式播放（假设 16kHz）
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32767.0
            audio_float = audio_float.reshape(-1, 1)
            
            sample_rate = 16000
            # === 重采样逻辑修改：确保目标是 TARGET_SAMPLE_RATE ===
            if sample_rate != TARGET_SAMPLE_RATE:
                print(f"[播放] 设备不支持 {sample_rate} Hz，重采样到 {TARGET_SAMPLE_RATE} Hz")
                
                try:
                    from scipy.signal import resample
                    new_length = int(len(audio_float) * TARGET_SAMPLE_RATE / sample_rate)
                    audio_float = resample(audio_float.flatten(), new_length).reshape(-1, 1)
                    sample_rate = TARGET_SAMPLE_RATE
                    print(f"[播放] ✓ 重采样完成，新采样率: {sample_rate} Hz")
                except ImportError:
                    print("[播放] ⚠️  警告: scipy 未安装，使用 numpy 线性插值重采样")
                    audio_float = self._resample_audio(audio_float, sample_rate, TARGET_SAMPLE_RATE)
                    sample_rate = TARGET_SAMPLE_RATE
            # =======================================================
            
            print(f"[播放] 按 PCM16 格式播放（{sample_rate} Hz）")
            sd.play(audio_float, sample_rate, device=self.output_device)
            sd.wait()  # 等待播放完成
            # 更新播放时间（用于冷却机制，避免捕获自己播放的内容）
            self.last_playback_time = time.time()
            print("[播放] ✓ 播放完成")
            
        except Exception as e:
            print(f"[播放失败] 无法播放音频: {e}")
            import traceback
            traceback.print_exc()
    
    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if status:
            if status.input_overflow:
                print(f"\n⚠️  [音频状态] 输入溢出，可能丢失数据")
            elif status.input_underflow:
                print(f"\n⚠️  [音频状态] 输入欠载")
        
        if np is not None:
            volume = np.linalg.norm(indata) * 10
            # 显示音量条，让用户知道是否捕获到音频
            volume_bar_length = 30
            volume_bar = "█" * int(min(volume * 2, volume_bar_length))
            volume_percent = min(volume * 100, 100)
            print(f"[音频捕获] 音量: {volume_percent:5.1f}% |{volume_bar:<{volume_bar_length}}|", end='\r')
        
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
        """启动流式在线翻译"""
        if self.is_running:
            return
        
        print("=" * 60)
        print("启动在线流式翻译服务")
        print(f"源语言（对方语言）: {self.source_language}")
        print(f"本地语言: {self.local_lang}")
        print(f"输入设备: {self.input_device or '默认'}")
        
        # 验证输出设备
        if self.output_device is not None:
            if not self._validate_output_device(self.output_device):
                print(f"⚠️  警告: 输出设备 {self.output_device} 无效或不可用，将使用默认设备")
                self.output_device = None
            else:
                try:
                    device_info = sd.query_devices(self.output_device)
                    print(f"输出设备（播放）: {device_info['name']} (索引: {self.output_device})")
                except Exception:
                    print(f"输出设备（播放）: {self.output_device}")
        else:
            print(f"输出设备（播放）: 默认")
        
        print("=" * 60)
        print("持续监听音频输入，检测到句子完成时自动翻译并播放")
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
        
        self.is_running = True
        translation_thread = threading.Thread(target=self._process_translation_queue, daemon=True)
        translation_thread.start()
        
        # 启动播放线程
        print("[播放] 启动播放线程...")
        playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        playback_thread.start()
        
        try:
            # 验证输入设备
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
                self.DEFAULT_CHUNK_SIZE = int(self.sample_rate * 0.2)
            
            print(f"  输入采样率: {self.sample_rate} Hz")
            print(f"  ASR处理采样率: {self.asr_sample_rate} Hz")
            print(f"  设备: {self.input_device or '系统默认'}")
            
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                device=self.input_device,
                callback=self.audio_callback,
                blocksize=self.DEFAULT_CHUNK_SIZE,
            )
            
            self.stream.start()
            device_name = "系统默认"
            if self.input_device is not None:
                try:
                    device_info = sd.query_devices(self.input_device)
                    device_name = device_info['name']
                except Exception:
                    pass
            print(f"[音频输入] ✓ 音频流已启动，正在监听: {device_name}")
            
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
        
        # 停止流式 ASR
        if self.streaming_asr:
            self.streaming_asr.stop()
            self.streaming_asr = None
        
        # 停止音频流
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        print("[已停止]")


def translate_online_voice(
    voice: AudioInput,
    *,
    local_lang: str = "chinese",
    source_language: Optional[str] = None,
    language_hint: Optional[str] = None,
) -> bytes:
    """
    翻译远程/在线音频到本地语言（一次性处理）。
    
    Args:
        voice: 输入的音频（文件路径、Path 对象或音频字节）
        local_lang: 本地语言（默认：chinese）
        source_language: 源语言（如果不提供，会通过 ASR 自动检测）
        language_hint: 可选的语言提示
    
    Returns:
        翻译后的音频字节数据
    """
    print("=" * 60)
    print("[在线翻译] 开始处理对方发送的语音")
    print(f"本地语言: {local_lang}")
    print("=" * 60)
    
    # 步骤 1: 语音识别 (ASR)
    if source_language is None:
        source_language = "english"
    
    print(f"\n[步骤 1/3] 语音识别 (ASR)...")
    print(f"假设源语言: {source_language}")
    
    transcribed_text = asr_transcribe(voice, source_language)
    print(f"识别结果: {transcribed_text}")
    
    if not transcribed_text or not transcribed_text.strip():
        print("⚠️  警告: ASR 未返回有效结果")
        return b""
    
    # 步骤 2: 文本翻译
    print(f"\n[步骤 2/3] 文本翻译 ({source_language} -> {local_lang})...")
    translated_text = translate_text(transcribed_text, source_language, local_lang)
    print(f"翻译结果: {translated_text}")
    
    if not translated_text or not translated_text.strip():
        print("⚠️  警告: 翻译结果为空")
        return b""
    
    # 步骤 3: 语音合成 (TTS)
    print(f"\n[步骤 3/3] 语音合成 (TTS)...")
    audio_bytes = tts(translated_text, local_lang)
    print(f"✓ 语音合成完成，音频大小: {len(audio_bytes)} 字节")
    print("=" * 60)
    
    return audio_bytes


def play_audio_bytes(audio_bytes: bytes, output_device: Optional[Union[int, str]] = None) -> None:
    """
    在扬声器播放音频字节数据。
    
    Args:
        audio_bytes: 音频字节数据
        output_device: 输出设备索引或名称（None 表示使用默认设备）
    """
    if sd is None or np is None:
        print("[播放跳过] sounddevice 或 numpy 未安装")
        return
    
    if not audio_bytes:
        print("[播放跳过] 音频数据为空")
        return
    
    print(f"[播放] 准备播放音频，大小: {len(audio_bytes)} 字节")
    
    try:
        # 尝试使用 soundfile 读取（支持多种格式）
        if sf is not None:
            try:
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]
                
                if audio_data.ndim == 1:
                    audio_float = audio_data.reshape(-1, 1).astype(np.float32)
                else:
                    audio_float = audio_data.astype(np.float32)
                
                audio_float = np.clip(audio_float, -1.0, 1.0)
                
                print(f"[播放] 使用 soundfile 读取音频: {sample_rate} Hz, {channels} 声道")
                sd.play(audio_float, sample_rate, device=output_device)
                sd.wait()  # 等待播放完成
                print("[播放] ✓ 播放完成")
                return
            except Exception as e:
                print(f"[播放] soundfile 读取失败，尝试 wave: {e}")
        
        # 使用 wave 读取 WAV 文件
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
                
                print(f"[播放] 使用 wave 读取音频: {sample_rate} Hz, {channels} 声道")
                sd.play(audio_float, sample_rate, device=output_device)
                sd.wait()  # 等待播放完成
                print("[播放] ✓ 播放完成")
                return
        except wave.Error:
            print("[播放] WAV 解析失败，尝试按 PCM16 播放")
        
        # 最后尝试：按 PCM16 格式播放（假设 16kHz）
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32767.0
        audio_float = audio_float.reshape(-1, 1)
        
        print("[播放] 按 PCM16 格式播放（假设 16kHz）")
        sd.play(audio_float, 16000, device=output_device)
        sd.wait()  # 等待播放完成
        print("[播放] ✓ 播放完成")
        
    except Exception as e:
        print(f"[播放失败] 无法播放音频: {e}")
        import traceback
        traceback.print_exc()
