"""Stats summary panel — bandit-level metrics."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static


class StatCard(Static):
    """Single metric display."""

    DEFAULT_CSS = """
    StatCard {
        width: 1fr;
        height: 5;
        padding: 1 2;
        border: round $accent;
        margin: 0 1;
    }
    """

    def __init__(self, label: str, value: str = "—", **kwargs) -> None:
        super().__init__(f"[bold]{value}[/]\n[dim]{label}[/]", **kwargs)
        self._label = label
        self._value = value

    def set_value(self, value: str) -> None:
        self._value = value
        self.update(f"[bold]{self._value}[/]\n[dim]{self._label}[/]")


class StatsPanel(Static):
    """Row of stat cards showing bandit-level metrics.

    Args:
        layout: "horizontal" (default) for card row, "vertical" for
                compact key-value lines (sidebar use).
    """

    DEFAULT_CSS = """
    StatsPanel {
        height: auto;
        padding: 1 0;
    }

    #stats-row {
        height: auto;
    }

    .stat-line {
        padding: 0 1;
    }
    """

    def __init__(
        self,
        layout: str = "horizontal",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._layout_mode = layout

    def compose(self) -> ComposeResult:
        if self._layout_mode == "vertical":
            with Vertical(id="stats-col"):
                yield Static("[dim]Events:[/] —", id="vstat-events", classes="stat-line")
                yield Static("[dim]Rewarded:[/] —", id="vstat-rewarded", classes="stat-line")
                yield Static("[dim]Avg Reward:[/] —", id="vstat-reward", classes="stat-line")
                yield Static("[dim]Cost:[/] —", id="vstat-cost", classes="stat-line")
                yield Static("[dim]Budget:[/] —", id="vstat-budget", classes="stat-line")
        else:
            with Horizontal(id="stats-row"):
                yield StatCard("Total Events", id="stat-events")
                yield StatCard("Rewarded", id="stat-rewarded")
                yield StatCard("Avg Reward", id="stat-reward")
                yield StatCard("Total Cost", id="stat-cost")
                yield StatCard("Budget", id="stat-budget")

    def update_stats(self, data: dict[str, Any]) -> None:
        if self._layout_mode == "vertical":
            self._update_vertical(data)
        else:
            self._update_horizontal(data)

    @staticmethod
    def _budget_color(ratio: float) -> str:
        """Return a Rich color tag based on budget usage ratio."""
        if ratio > 1.0:
            return "bold red"
        elif ratio > 0.9:
            return "red"
        elif ratio > 0.7:
            return "yellow"
        return "green"

    @staticmethod
    def _format_budget(cost: float | None, budget: float | None) -> str:
        """Format the budget value string (no label prefix).

        When both cost and budget exist: ``$10.00 (25% used)`` with color.
        When only budget exists: ``$10.00``.
        Otherwise: ``—``.
        """
        if cost is not None and budget is not None and budget > 0:
            ratio = cost / budget
            color = StatsPanel._budget_color(ratio)
            return f"[{color}]${budget:.2f} ({ratio:.0%} used)[/]"
        if budget is not None:
            return f"${budget:.2f}"
        return "—"

    def _update_horizontal(self, data: dict[str, Any]) -> None:
        self.query_one("#stat-events", StatCard).set_value(
            f"{data.get('total_events', 0):,}"
        )
        self.query_one("#stat-rewarded", StatCard).set_value(
            f"{data.get('total_rewarded', 0):,}"
        )
        avg = data.get("avg_reward")
        self.query_one("#stat-reward", StatCard).set_value(
            f"{avg:.3f}" if avg is not None else "—"
        )
        cost = data.get("total_cost")
        budget = data.get("budget")
        self.query_one("#stat-cost", StatCard).set_value(
            f"${cost:.2f}" if cost is not None else "—"
        )
        self.query_one("#stat-budget", StatCard).set_value(
            self._format_budget(cost, budget)
        )

    def _update_vertical(self, data: dict[str, Any]) -> None:
        events = data.get("total_events", 0)
        self.query_one("#vstat-events", Static).update(
            f"[dim]Events:[/] {events:,}"
        )
        rewarded = data.get("total_rewarded", 0)
        self.query_one("#vstat-rewarded", Static).update(
            f"[dim]Rewarded:[/] {rewarded:,}"
        )
        avg = data.get("avg_reward")
        self.query_one("#vstat-reward", Static).update(
            f"[dim]Avg Reward:[/] {avg:.3f}" if avg is not None else "[dim]Avg Reward:[/] —"
        )
        cost = data.get("total_cost")
        budget = data.get("budget")
        self.query_one("#vstat-cost", Static).update(
            f"[dim]Cost:[/] ${cost:.2f}" if cost is not None else "[dim]Cost:[/] —"
        )
        self.query_one("#vstat-budget", Static).update(
            f"[dim]Budget:[/] {self._format_budget(cost, budget)}"
        )
