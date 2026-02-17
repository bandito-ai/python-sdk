"""Tests for TUI config loader."""

import os
from pathlib import Path

import pytest

from bandito.config import BanditoConfig, load_config, save_config, DEFAULT_BASE_URL


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Redirect config to a temp directory."""
    import bandito.config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.toml")
    return tmp_path


class TestLoadConfig:
    def test_no_file_no_env(self, config_dir, monkeypatch):
        monkeypatch.delenv("BANDITO_API_KEY", raising=False)
        monkeypatch.delenv("BANDITO_BASE_URL", raising=False)
        cfg = load_config()
        assert cfg.api_key is None
        assert cfg.base_url == DEFAULT_BASE_URL

    def test_reads_toml(self, config_dir):
        toml_file = config_dir / "config.toml"
        toml_file.write_text('api_key = "bnd_test123"\nbase_url = "http://localhost:8000"\n')
        cfg = load_config()
        assert cfg.api_key == "bnd_test123"
        assert cfg.base_url == "http://localhost:8000"

    def test_env_vars_override_toml(self, config_dir, monkeypatch):
        toml_file = config_dir / "config.toml"
        toml_file.write_text('api_key = "from_toml"\n')
        monkeypatch.setenv("BANDITO_API_KEY", "from_env")
        cfg = load_config()
        assert cfg.api_key == "from_env"

    def test_env_base_url(self, config_dir, monkeypatch):
        monkeypatch.setenv("BANDITO_BASE_URL", "http://custom:9000")
        cfg = load_config()
        assert cfg.base_url == "http://custom:9000"


class TestSaveConfig:
    def test_saves_toml(self, config_dir):
        save_config("bnd_abc", "http://custom:9000")
        content = (config_dir / "config.toml").read_text()
        assert 'api_key = "bnd_abc"' in content
        assert 'base_url = "http://custom:9000"' in content

    def test_saves_default_url_omits_base_url(self, config_dir):
        save_config("bnd_abc")
        content = (config_dir / "config.toml").read_text()
        assert "api_key" in content
        assert "base_url" not in content

    def test_creates_directory(self, tmp_path, monkeypatch):
        import bandito.config as cfg
        nested = tmp_path / "sub" / "dir"
        monkeypatch.setattr(cfg, "CONFIG_DIR", nested)
        monkeypatch.setattr(cfg, "CONFIG_FILE", nested / "config.toml")
        save_config("bnd_test")
        assert (nested / "config.toml").exists()
