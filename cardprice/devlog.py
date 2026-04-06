"""
Persistent development log for the cardprice project.

Stores timestamped entries as JSONL at data/devlog.jsonl.
Entry types: bug, fix, note, eval_result, session_summary.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEVLOG_PATH = Path(__file__).resolve().parent.parent / "data" / "devlog.jsonl"


def _append(entry: dict) -> dict:
    """Append a single entry to the JSONL file and return it."""
    DEVLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEVLOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def _make_entry(entry_type: str, title: str, body: str,
                tags: list[str] = None, related_files: list[str] = None,
                **extra) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": entry_type,
        "title": title,
        "body": body,
        "tags": tags or [],
        "related_files": related_files or [],
    }
    entry.update(extra)
    return _append(entry)


def log_bug(title: str, body: str, tags: list[str] = None,
            related_files: list[str] = None) -> dict:
    return _make_entry("bug", title, body, tags, related_files)


def log_fix(title: str, body: str, tags: list[str] = None,
            related_files: list[str] = None) -> dict:
    return _make_entry("fix", title, body, tags, related_files)


def log_note(title: str, body: str, tags: list[str] = None,
             related_files: list[str] = None) -> dict:
    return _make_entry("note", title, body, tags, related_files)


def log_eval(accuracy: float, total: int, failures: list[str] = None,
             notes: str = "") -> dict:
    return _make_entry(
        "eval_result",
        f"Eval: {accuracy:.1%} ({int(accuracy * total)}/{total})",
        notes,
        tags=["eval"],
        failures=failures or [],
    )


def log_session(title: str, body: str, tags: list[str] = None,
                related_files: list[str] = None) -> dict:
    return _make_entry("session_summary", title, body, tags, related_files)


def _read_all() -> list[dict]:
    """Read all entries from the JSONL file."""
    if not DEVLOG_PATH.exists():
        return []
    entries = []
    with open(DEVLOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_recent(n: int = 20, type_filter: str = None) -> list[dict]:
    """Return the most recent n entries, optionally filtered by type."""
    entries = _read_all()
    if type_filter:
        entries = [e for e in entries if e.get("type") == type_filter]
    return entries[-n:]


def search(keyword: str) -> list[dict]:
    """Return entries where keyword appears in title or body (case-insensitive)."""
    keyword = keyword.lower()
    return [
        e for e in _read_all()
        if keyword in e.get("title", "").lower()
        or keyword in e.get("body", "").lower()
    ]
