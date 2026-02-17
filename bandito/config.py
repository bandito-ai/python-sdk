"""Config loader — reads ~/.bandito/config.toml and env vars."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR = Path.home() / ".bandito"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_BASE_URL = "http://localhost:8000"


@dataclass
class BanditoConfig:
    api_key: str | None = None
    base_url: str = DEFAULT_BASE_URL


def load_config() -> BanditoConfig:
    """Load config from TOML file, falling back to env vars."""
    config = BanditoConfig()

    # TOML file first
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "rb") as f:
            data = tomllib.load(f)
        config.api_key = data.get("api_key", config.api_key)
        config.base_url = data.get("base_url", config.base_url)

    # Env vars override
    env_key = os.environ.get("BANDITO_API_KEY")
    if env_key:
        config.api_key = env_key
    env_url = os.environ.get("BANDITO_BASE_URL")
    if env_url:
        config.base_url = env_url

    return config


def save_config(api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
    """Write config to ~/.bandito/config.toml."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    lines = [f'api_key = "{api_key}"']
    if base_url != DEFAULT_BASE_URL:
        lines.append(f'base_url = "{base_url}"')
    CONFIG_FILE.write_text("\n".join(lines) + "\n")
    CONFIG_FILE.chmod(0o600)
