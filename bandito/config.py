"""Config loader — reads ~/.bandito/config.toml and env vars."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".bandito"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_BASE_URL = "https://bandito-api.onrender.com"


@dataclass
class BanditoConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL
    data_storage: str = "local"  # "local" or "cloud"


def _escape_toml_value(value: str) -> str:
    """Escape a string for use as a TOML basic string value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def load_config() -> BanditoConfig:
    """Load config from TOML file, falling back to env vars."""
    config = BanditoConfig()

    # TOML file first
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
            config.api_key = data.get("api_key", config.api_key)
            config.base_url = data.get("base_url", config.base_url)
            config.data_storage = data.get("data_storage", config.data_storage)
        except tomllib.TOMLDecodeError:
            logger.warning(
                "Failed to parse %s — falling back to env vars", CONFIG_FILE
            )

    # Env vars override
    env_key = os.environ.get("BANDITO_API_KEY")
    if env_key:
        config.api_key = env_key
    env_url = os.environ.get("BANDITO_BASE_URL")
    if env_url:
        config.base_url = env_url
    env_storage = os.environ.get("BANDITO_DATA_STORAGE")
    if env_storage:
        config.data_storage = env_storage

    return config


def save_config(
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    data_storage: str = "local",
) -> None:
    """Write config to ~/.bandito/config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    lines = [f'api_key = "{_escape_toml_value(api_key)}"']
    if base_url != DEFAULT_BASE_URL:
        lines.append(f'base_url = "{_escape_toml_value(base_url)}"')
    lines.append(f'data_storage = "{_escape_toml_value(data_storage)}"')
    CONFIG_FILE.write_text("\n".join(lines) + "\n")
    CONFIG_FILE.chmod(0o600)
