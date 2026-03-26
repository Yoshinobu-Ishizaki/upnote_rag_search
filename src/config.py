"""Configuration management for UpNote RAG Search.

Reads from config.ini (settings) and .env (secrets).
"""
import configparser
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"
ENV_PATH = PROJECT_ROOT / ".env"


def _read_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH, encoding="utf-8")
    return cfg


def get_backup_path() -> Path:
    cfg = _read_config()
    raw = cfg.get("settings", "UPNOTE_BACKUP_PATH", fallback="")
    if raw:
        p = Path(raw)
        # If relative, resolve against project root
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        return p
    return PROJECT_ROOT / "UpNote" / "General Space"


def get_top_k() -> int:
    cfg = _read_config()
    return int(cfg.get("settings", "TOP_K_RESULTS", fallback="10"))


def get_max_context_chars() -> int:
    cfg = _read_config()
    return int(cfg.get("settings", "MAX_CONTEXT_CHARS", fallback="50000"))


def get_api_key() -> str:
    """Read ANTHROPIC_API_KEY from .env, then environment variables."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    return os.environ.get("ANTHROPIC_API_KEY", "")


def get_claude_model() -> str:
    cfg = _read_config()
    return cfg.get("settings", "CLAUDE_MODEL", fallback="claude-haiku-4-5-20251001")


def get_embedding_provider() -> str:
    """Return 'google' or 'local'."""
    cfg = _read_config()
    return cfg.get("settings", "EMBEDDING_PROVIDER", fallback="local").strip().lower()


def get_gemini_api_key() -> str:
    """Read GEMINI_API_KEY from .env, then environment variables."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    return os.environ.get("GEMINI_API_KEY", "")


def save_backup_path(path: str) -> None:
    """Save backup path to config.ini."""
    cfg = _read_config()
    if not cfg.has_section("settings"):
        cfg.add_section("settings")
    cfg.set("settings", "UPNOTE_BACKUP_PATH", path)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)


VALID_DATE_MODES = ["すべて", "以前", "以降", "範囲"]


def get_default_date_mode() -> str:
    cfg = _read_config()
    value = cfg.get("settings", "DEFAULT_DATE_MODE", fallback="すべて").strip()
    return value if value in VALID_DATE_MODES else "すべて"


def get_default_categories() -> list[str]:
    cfg = _read_config()
    raw = cfg.get("settings", "DEFAULT_CATEGORIES", fallback="").strip()
    return [c.strip() for c in raw.split(",") if c.strip()] if raw else []


def get_default_tags() -> list[str]:
    cfg = _read_config()
    raw = cfg.get("settings", "DEFAULT_TAGS", fallback="").strip()
    return [t.strip() for t in raw.split(",") if t.strip()] if raw else []


def get_default_start_date() -> "datetime.date":
    import datetime
    cfg = _read_config()
    raw = cfg.get("settings", "DEFAULT_START_DATE", fallback="").strip()
    try:
        return datetime.date.fromisoformat(raw) if raw else datetime.date(1998, 1, 1)
    except ValueError:
        return datetime.date(1998, 1, 1)


def get_default_end_date() -> "datetime.date":
    import datetime
    cfg = _read_config()
    raw = cfg.get("settings", "DEFAULT_END_DATE", fallback="").strip()
    try:
        return datetime.date.fromisoformat(raw) if raw else datetime.date.today()
    except ValueError:
        return datetime.date.today()


def get_auto_preprocess() -> bool:
    cfg = _read_config()
    return cfg.get("settings", "AUTO_PREPROCESS", fallback="false").strip().lower() == "true"


def save_auto_preprocess(enabled: bool) -> None:
    cfg = _read_config()
    if not cfg.has_section("settings"):
        cfg.add_section("settings")
    cfg.set("settings", "AUTO_PREPROCESS", "true" if enabled else "false")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        cfg.write(f)


def save_api_key(key: str) -> None:
    """Save API key to .env file."""
    lines = []
    found = False
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                lines.append(f"ANTHROPIC_API_KEY={key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"ANTHROPIC_API_KEY={key}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_gemini_api_key(key: str) -> None:
    """Save Gemini API key to .env file."""
    lines = []
    found = False
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                lines.append(f"GEMINI_API_KEY={key}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"GEMINI_API_KEY={key}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
