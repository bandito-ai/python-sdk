"""Tests for `bandito init` command."""

import getpass
from unittest.mock import patch

import httpx
import pytest
import respx

from bandito.config import DEFAULT_BASE_URL


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    """Redirect config to a temp directory."""
    import bandito.config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.toml")
    # Also patch cli_init's import reference
    import bandito.cli_init as cli_mod

    monkeypatch.setattr(cli_mod, "CONFIG_FILE", tmp_path / "config.toml")
    monkeypatch.setattr(cli_mod, "save_config", cfg.save_config)
    return tmp_path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("BANDITO_API_KEY", raising=False)
    monkeypatch.delenv("BANDITO_BASE_URL", raising=False)


class TestInitSavesConfig:
    @respx.mock
    def test_saves_config_from_prompt(self, config_dir, monkeypatch):
        """Interactive flow: prompt for key + default URL → config saved."""
        monkeypatch.setattr(getpass, "getpass", lambda _: "bnd_test123")
        monkeypatch.setattr("builtins.input", lambda _: "")  # accept default URL

        route = respx.get(f"{DEFAULT_BASE_URL}/api/v1/bandits").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        from bandito.cli_init import run_init

        run_init()

        assert route.called
        content = (config_dir / "config.toml").read_text()
        assert 'api_key = "bnd_test123"' in content
        assert "base_url" not in content  # default URL omitted

    @respx.mock
    def test_saves_config_custom_url(self, config_dir, monkeypatch):
        monkeypatch.setattr(getpass, "getpass", lambda _: "bnd_key")
        monkeypatch.setattr("builtins.input", lambda _: "http://custom:9000")

        respx.get("http://custom:9000/api/v1/bandits").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        from bandito.cli_init import run_init

        run_init()

        content = (config_dir / "config.toml").read_text()
        assert 'base_url = "http://custom:9000"' in content


class TestInitUsesEnvVar:
    @respx.mock
    def test_uses_env_var(self, config_dir, monkeypatch):
        monkeypatch.setenv("BANDITO_API_KEY", "bnd_from_env")
        monkeypatch.setattr("builtins.input", lambda _: "")

        respx.get(f"{DEFAULT_BASE_URL}/api/v1/bandits").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        from bandito.cli_init import run_init

        run_init()

        content = (config_dir / "config.toml").read_text()
        assert 'api_key = "bnd_from_env"' in content


class TestInitHandles401:
    @respx.mock
    def test_invalid_key_exits(self, config_dir, monkeypatch):
        monkeypatch.setattr(getpass, "getpass", lambda _: "bad_key")
        monkeypatch.setattr("builtins.input", lambda _: "")

        respx.get(f"{DEFAULT_BASE_URL}/api/v1/bandits").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid API key"})
        )

        from bandito.cli_init import run_init

        with pytest.raises(SystemExit) as exc_info:
            run_init()
        assert exc_info.value.code == 1
        assert not (config_dir / "config.toml").exists()


class TestInitHandlesConnectionError:
    @respx.mock
    def test_connect_error_exits(self, config_dir, monkeypatch):
        monkeypatch.setattr(getpass, "getpass", lambda _: "bnd_key")
        monkeypatch.setattr("builtins.input", lambda _: "")

        respx.get(f"{DEFAULT_BASE_URL}/api/v1/bandits").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        from bandito.cli_init import run_init

        with pytest.raises(SystemExit) as exc_info:
            run_init()
        assert exc_info.value.code == 1


class TestInitIdempotent:
    @respx.mock
    def test_overwrite_existing(self, config_dir, monkeypatch):
        """Existing config + user says 'y' → overwrite."""
        (config_dir / "config.toml").write_text('api_key = "old_key"\n')

        inputs = iter(["y", ""])  # overwrite=y, default URL
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setattr(getpass, "getpass", lambda _: "bnd_new_key")

        respx.get(f"{DEFAULT_BASE_URL}/api/v1/bandits").mock(
            return_value=httpx.Response(200, json={"items": [], "total": 0})
        )

        from bandito.cli_init import run_init

        run_init()

        content = (config_dir / "config.toml").read_text()
        assert 'api_key = "bnd_new_key"' in content

    @respx.mock
    def test_abort_existing(self, config_dir, monkeypatch):
        """Existing config + user says 'n' → abort, keep old config."""
        (config_dir / "config.toml").write_text('api_key = "old_key"\n')

        monkeypatch.setattr("builtins.input", lambda _: "n")

        from bandito.cli_init import run_init

        run_init()

        content = (config_dir / "config.toml").read_text()
        assert 'api_key = "old_key"' in content
