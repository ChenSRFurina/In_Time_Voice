from __future__ import annotations

"""
SiliconFlow CosyVoice2 语音克隆与 TTS 辅助模块。
"""

import json
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import requests

_CONFIG_PATH = Path("voice_clone_config.json")
_DEFAULT_MODEL = os.getenv("SILICONFLOW_TTS_MODEL", "FunAudioLLM/CosyVoice2-0.5B")
_UPLOAD_URL = "https://api.siliconflow.cn/v1/uploads/audio/voice"
_SPEECH_URL = "https://api.siliconflow.cn/v1/audio/speech"
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".pcm", ".opus"}


def _load_config() -> Dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(cfg: Dict[str, Any]) -> None:
    try:
        _CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[音色克隆] 配置已保存: {_CONFIG_PATH.resolve()}")
    except Exception as exc:
        print(f"[音色克隆] 保存配置失败: {exc}")


def _get_api_key(cfg: Optional[Dict[str, Any]] = None) -> str:
    cfg = cfg or _load_config()
    api_key = os.getenv("SILICONFLOW_API_KEY") or cfg.get("sf_api_key")
    if not api_key:
        raise RuntimeError(
            "[音色克隆] 未检测到 SiliconFlow API Key，请设置环境变量 SILICONFLOW_API_KEY "
            "或调用 configure_siliconflow(api_key=...)."
        )
    return api_key


def configure_siliconflow(
    *,
    api_key: Optional[str] = None,
    voice_id: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    cfg = _load_config()
    if api_key:
        cfg["sf_api_key"] = api_key
    if voice_id:
        cfg["sf_voice_id"] = voice_id
    if model:
        cfg["sf_model"] = model
    if cfg:
        _save_config(cfg)


def upload_voice_sample(
    api_key: str,
    audio_path: str,
    sample_text: str,
    *,
    custom_name: str = "voice-clone",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """上传本地参考录音，创建 SiliconFlow 克隆音色。"""
    payload = {
        "model": model or _DEFAULT_MODEL,
        "text": sample_text,
        "customName": custom_name,
    }

    mime_type, _ = mimetypes.guess_type(audio_path)
    if not mime_type:
        mime_type = "audio/wav"

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with open(audio_path, "rb") as file_obj:
            files = {"file": (os.path.basename(audio_path), file_obj, mime_type)}
            resp = requests.post(_UPLOAD_URL, headers=headers, data=payload, files=files, timeout=300)
    except requests.RequestException as exc:  # pragma: no cover - 网络错误
        raise RuntimeError(f"[音色克隆] 上传参考录音失败: {exc}") from exc

    try:
        return resp.json()
    except ValueError:
        return {"status_code": resp.status_code, "text": resp.text}


def _extract_voice_id(response: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(response, dict):
        return None
    data = response.get("data") if isinstance(response.get("data"), dict) else {}
    return (
        response.get("voiceId")
        or response.get("voice_id")
        or data.get("voiceId")
        or data.get("voice_id")
        or response.get("uri")
    )


def set_clone_reference(
    audio_file: str,
    *,
    sample_text: str,
    custom_name: Optional[str] = None,
    apply_scope: str = "all", 
    force: bool = False,
) -> Dict[str, Any]:
    """
    使用 SiliconFlow 语音克隆接口上传本地录音并创建 voice_id。
    """
    original_path = Path(audio_file).expanduser().resolve()
    if not original_path.exists():
        raise FileNotFoundError(f"[音色克隆] 参考录音不存在: {original_path}")
    upload_path = _ensure_supported_audio(original_path)

    cfg = _load_config()
    cfg["mode"] = "siliconflow"
    cfg["reference_audio"] = str(original_path)
    cfg["apply_scope"] = apply_scope

    if cfg.get("sf_voice_id") and not force:
        print(f"[音色克隆] 已存在音色: {cfg['sf_voice_id']}，跳过重新创建（如需重新克隆请使用 force=True）")
        _save_config(cfg)
        return cfg

    api_key = _get_api_key(cfg)
    text = sample_text
    custom = custom_name or original_path.stem[:32] or "voice-clone"

    print(f"[音色克隆] 正在将参考录音上传到 SiliconFlow（名称: {custom}）")
    response = upload_voice_sample(
        api_key,
        str(upload_path),
        text,
        custom_name=custom,
        model=cfg.get("sf_model") or _DEFAULT_MODEL,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"[音色克隆] 上传接口返回异常数据: {response!r}")
    voice_id = _extract_voice_id(response)
    if not voice_id:
        raise RuntimeError(f"[音色克隆] 上传成功但未返回 voiceId，响应: {response}")

    cfg["sf_voice_id"] = voice_id
    if "sf_api_key" not in cfg:
        cfg["sf_api_key"] = api_key
    cfg["created_at"] = response.get("created_at") or response.get("timestamp")

    _save_config(cfg)
    print(f"[音色克隆] ✓ 克隆音色创建成功，voice_id={voice_id}")
    return cfg


def get_clone_reference() -> Optional[Path]:
    cfg = _load_config()
    ref = cfg.get("reference_audio")
    if not ref:
        return None
    path = Path(ref)
    if not path.exists():
        print(f"[音色克隆] 警告: 配置中的参考录音不存在: {path}")
        return None
    return path


def get_clone_voice_id() -> Optional[str]:
    cfg = _load_config()
    return cfg.get("sf_voice_id")


def synthesize_with_clone(
    text: str,
    *,
    voice_id: Optional[str] = None,
    audio_format: str = "mp3",
    timeout: int = 600,
) -> bytes:
    """
    使用 SiliconFlow 专属音色生成语音。
    """
    cfg = _load_config()
    api_key = _get_api_key(cfg)
    voice_id = voice_id or os.getenv("SILICONFLOW_VOICE_ID") or cfg.get("sf_voice_id")
    if not voice_id:
        raise RuntimeError("[音色克隆] 未找到任何 voice_id，请先执行 set_clone_reference 或配置 SILICONFLOW_VOICE_ID。")

    payload = {
        "model": cfg.get("sf_model") or _DEFAULT_MODEL,
        "input": text,
        "voice": voice_id,
        "format": audio_format,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "audio/mpeg, application/octet-stream",
    }

    resp = requests.post(_SPEECH_URL, headers=headers, data=json.dumps(payload, ensure_ascii=False), timeout=timeout)
    if resp.status_code == 200 and resp.content:
        print(f"[音色克隆] 使用 SiliconFlow 音色 {voice_id} 合成成功，音频长度 {len(resp.content)} 字节")
        return resp.content

    try:
        error_info = resp.json()
    except Exception:
        error_info = resp.text
    raise RuntimeError(f"[音色克隆] SiliconFlow 合成失败: {resp.status_code} - {error_info}")


def _ensure_supported_audio(path: Path) -> Path:
    if path.suffix.lower() in SUPPORTED_SUFFIXES:
        return path

    converted = path.with_suffix(".wav")
    print(f"[音色克隆] 参考录音格式 {path.suffix} 不受支持，正在自动转换为 WAV: {converted.name}")

    try:
        from pydub import AudioSegment  # type: ignore[import-not-found]

        audio = AudioSegment.from_file(path)
        audio.export(converted, format="wav")
    except ImportError:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "[音色克隆] 需要安装 pydub 或系统可用的 ffmpeg 才能自动转换音频格式。"
            )
        result = subprocess.run(
            [ffmpeg, "-y", "-i", str(path), str(converted)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"[音色克隆] ffmpeg 转换失败: {stderr}")

    if not converted.exists():
        raise RuntimeError("[音色克隆] 转换后的音频文件不存在，转换过程可能失败。")

    return converted

