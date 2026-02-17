"""Arm performance table — shows which arm is winning."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import DataTable, Static


class ArmTable(Static):
    """DataTable showing per-arm performance metrics.

    Args:
        compact: If True, show only 3 columns (Model, Pull%, Reward)
                 for sidebar use. Default False shows full 8 columns.
    """

    DEFAULT_CSS = """
    ArmTable {
        height: 1fr;
        padding: 0 1;
    }

    #arm-perf-table {
        height: 1fr;
    }

    #arm-empty {
        padding: 2;
        text-align: center;
        color: $text-muted;
        display: none;
    }
    """

    def __init__(
        self,
        compact: bool = False,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._compact = compact

    def compose(self) -> ComposeResult:
        yield DataTable(id="arm-perf-table", cursor_type="row")
        yield Static("No arms configured.", id="arm-empty")

    def on_mount(self) -> None:
        table = self.query_one("#arm-perf-table", DataTable)
        if self._compact:
            table.add_columns("Model", "Pull %", "Reward")
        else:
            table.add_columns(
                "Model", "Provider", "Events", "Pull %", "Avg Reward",
                "Avg Cost", "Reviewed", "Review %",
            )

    def update_arms(self, data: dict[str, Any]) -> None:
        table = self.query_one("#arm-perf-table", DataTable)
        table.clear()

        arms = data.get("arms", [])
        if not arms:
            self.query_one("#arm-empty").styles.display = "block"
            return

        self.query_one("#arm-empty").styles.display = "none"

        # Sort by pull_share descending (winning arm at top)
        arms = sorted(arms, key=lambda a: a.get("pull_share", 0), reverse=True)

        for arm in arms:
            pull_share = arm.get("pull_share")
            avg_reward = arm.get("avg_computed_reward")

            if self._compact:
                table.add_row(
                    arm.get("model_name", "?"),
                    f"{pull_share:.0%}" if pull_share is not None else "—",
                    f"{avg_reward:.3f}" if avg_reward is not None else "—",
                )
            else:
                avg_cost = arm.get("avg_cost")
                review_ratio = arm.get("review_ratio")

                table.add_row(
                    arm.get("model_name", "?"),
                    arm.get("model_provider", "?"),
                    str(arm.get("event_count", 0)),
                    f"{pull_share:.0%}" if pull_share is not None else "—",
                    f"{avg_reward:.3f}" if avg_reward is not None else "—",
                    f"${avg_cost:.4f}" if avg_cost is not None else "—",
                    str(arm.get("human_reward_count", 0)),
                    f"{review_ratio:.0%}" if review_ratio is not None else "—",
                )
