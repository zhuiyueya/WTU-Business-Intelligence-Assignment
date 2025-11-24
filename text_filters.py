from __future__ import annotations

import re

_NOISE_WORDS = [
    "object",
    "cq",
    "id",
    "face",
    "raw",
    "chaincount",
    "at",
    "qq",
    "reply",
    "forward",
    "com",
    "https",
    "qzone",
    "type",
    "amp",
    "groupphoto",
    "3d2",
    "file",
    "url",
    "size",
]

_NOISE_PATTERN = re.compile(
    r"(?:\b(?:"
    + "|".join(re.escape(word) for word in _NOISE_WORDS)
    + r")\b|\[cq:)",
    re.IGNORECASE,
)


def is_noise_message(text: str) -> bool:
    if not text:
        return True
    return bool(_NOISE_PATTERN.search(text))
