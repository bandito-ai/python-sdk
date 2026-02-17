"""Backwards-compat shim — config moved to bandito.config."""

from bandito.config import (  # noqa: F401
    BanditoConfig,
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_BASE_URL,
    load_config,
    save_config,
)
