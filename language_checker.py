"""Simple heuristics for detecting the language of an audio sample."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

AudioSource = Union[str, Path]

_LANGUAGE_ALIASES = {
    "chinese": {"zh", "cn", "mandarin", "chinese"},
    "english": {"en", "eng", "english"},
    "japanese": {"jp", "jpn", "japanese"},
    "korean": {"kr", "kor", "korean"},
}


def _normalise_language(language: str) -> Optional[str]:
    language = language.strip().lower()
    for canonical, aliases in _LANGUAGE_ALIASES.items():
        if language == canonical or language in aliases:
            return canonical
    return None


def _language_from_name(name: str) -> Optional[str]:
    tokens = (
        name.replace("-", " ")
        .replace("_", " ")
        .replace(".", " ")
        .lower()
        .split()
    )
    for token in tokens:
        normalised = _normalise_language(token)
        if normalised:
            return normalised
    return None


def check_language(
    audio_source: AudioSource,
    *,
    language_hint: Optional[str] = None,
    default: str = "english",
    **_unused: object,
) -> str:
    """
    Best-effort detection of the audio language.

    The function relies on light-weight heuristics (file name inspection, hints)
    so that the rest of the pipeline can operate without heavyweight models.
    Extra keyword arguments are ignored for backward compatibility.
    """

    if language_hint:
        hint = _normalise_language(language_hint)
        if hint:
            return hint

    path = Path(audio_source) if isinstance(audio_source, (str, Path)) else None
    if path:
        detected = _language_from_name(path.name)
        if detected:
            return detected

    return _normalise_language(default) or default