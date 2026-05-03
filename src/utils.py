"""
utils.py — Shared utility functions.
"""
from __future__ import annotations
import re
from typing import Optional


def is_valid_url(url: str) -> bool:
    """Quick URL format validation."""
    pattern = re.compile(
        r"^(https?://)"
        r"(([a-zA-Z0-9\-]+\.)+[a-zA-Z]{2,})"
        r"(:\d+)?(/.*)?$"
    )
    return bool(pattern.match(url.strip()))


def truncate(text: str, max_len: int = 120) -> str:
    """Truncate text with ellipsis."""
    return text[:max_len] + "…" if len(text) > max_len else text


def format_bytes(size: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def source_icon(source_type: str) -> str:
    """Return an emoji icon for a source type."""
    icons = {
        "pdf": "📄",
        "docx": "📝",
        "text": "📃",
        "web": "🌐",
        "unknown": "📌",
    }
    return icons.get(source_type, "📌")


def clean_url(url: str) -> str:
    """Normalise a URL string."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url
