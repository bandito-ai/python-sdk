"""Tests for TUI screens using Textual's Pilot."""

import pytest
from unittest.mock import MagicMock, patch

from textual.widgets import DataTable, Input, Button, Static, ListView

from bandito.tui.app import BanditoApp
from bandito.tui.screens.setup import SetupScreen
from bandito.tui.screens.bandit_select import BanditSelectScreen
from bandito.tui.screens.help import HelpScreen


@pytest.fixture
def mock_config_no_key():
    """Patch load_config to return no API key."""
    from bandito.config import BanditoConfig
    with patch("bandito.tui.app.load_config", return_value=BanditoConfig()):
        yield


@pytest.fixture
def mock_config_with_key():
    """Patch load_config to return a valid API key."""
    from bandito.config import BanditoConfig
    cfg = BanditoConfig(api_key="bnd_test", base_url="http://localhost:8000")
    with patch("bandito.tui.app.load_config", return_value=cfg):
        yield


@pytest.fixture
def mock_api():
    """Mock TuiAPI for testing."""
    api = MagicMock()
    api.list_bandits.return_value = {
        "items": [
            {"id": 1, "name": "prod-bot", "type": "online", "arm_count": 3,
             "total_pull_count": 500, "optimization_mode": "maximize"},
            {"id": 2, "name": "experiment", "type": "experiment", "arm_count": 2,
             "total_pull_count": 50, "optimization_mode": "explore"},
        ],
        "total": 2,
    }
    api.get_stats.return_value = {
        "bandit_id": 1, "total_events": 100, "total_rewarded": 80,
        "avg_computed_reward": 0.75, "total_cost": 5.0, "budget": 50.0,
    }
    api.get_arm_performance.return_value = {
        "bandit_id": 1, "total_events": 100,
        "arms": [
            {"arm_id": 1, "model_name": "gpt-4", "model_provider": "openai",
             "event_count": 60, "pull_share": 0.6, "avg_computed_reward": 0.8,
             "avg_cost": 0.05, "human_reward_count": 10, "review_ratio": 0.17},
        ],
    }
    api.list_events.return_value = {"items": [], "total": 0}
    return api


class TestAppLaunch:
    @pytest.mark.asyncio
    async def test_no_key_shows_setup(self, mock_config_no_key):
        async with BanditoApp().run_test() as pilot:
            assert isinstance(pilot.app.screen, SetupScreen)

    @pytest.mark.asyncio
    async def test_with_key_shows_bandit_select(self, mock_config_with_key, mock_api):
        app = BanditoApp()
        with patch.object(app, "_init_api"):
            app.api = mock_api
            async with app.run_test() as pilot:
                assert isinstance(pilot.app.screen, BanditSelectScreen)


class TestSetupScreen:
    @pytest.mark.asyncio
    async def test_empty_key_shows_error(self, mock_config_no_key):
        async with BanditoApp().run_test() as pilot:
            await pilot.click("#setup-submit")
            await pilot.pause()
            error = pilot.app.screen.query_one("#setup-error", Static)
            # Check that error is visible and contains the right message
            assert error.styles.display == "block"


class TestHelpScreen:
    @pytest.mark.asyncio
    async def test_help_action(self, mock_config_no_key):
        """Verify help screen can be opened via action."""
        async with BanditoApp().run_test() as pilot:
            pilot.app.action_help()
            await pilot.pause()
            assert isinstance(pilot.app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(pilot.app.screen, SetupScreen)


class TestBanditSelectScreen:
    @pytest.mark.asyncio
    async def test_shows_bandits(self, mock_config_with_key, mock_api):
        app = BanditoApp()
        with patch.object(app, "_init_api"):
            app.api = mock_api
            async with app.run_test() as pilot:
                await pilot.pause()
                # Wait for worker to complete
                await pilot.app.workers.wait_for_complete()
                await pilot.pause()
                table = pilot.app.screen.query_one("#bandit-table", DataTable)
                assert table.row_count == 2
