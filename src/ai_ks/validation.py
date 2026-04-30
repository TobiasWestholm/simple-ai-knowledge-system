from __future__ import annotations


def normalize_user_text(value: str, *, field_name: str, max_chars: int) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty or whitespace.")
    if len(normalized) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters.")
    return normalized
