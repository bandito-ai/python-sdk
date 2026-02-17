"""Tests for `bandito create` command."""

from unittest.mock import patch

from bandito.cli_create import DASHBOARD_URL, run_create


class TestCreate:
    def test_opens_browser(self):
        with patch("bandito.cli_create.webbrowser.open") as mock_open:
            run_create()
            mock_open.assert_called_once_with(DASHBOARD_URL)
