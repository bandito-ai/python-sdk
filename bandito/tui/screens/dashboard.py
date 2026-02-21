"""Dashboard — grading-first split pane with toggleable sidebar."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import ListView, Markdown, Static
from textual import work
from textual.worker import Worker, WorkerState

from bandito.tui.widgets.arm_table import ArmTable
from bandito.tui.utils import format_response
from bandito.tui.widgets.event_card import EventCard
from bandito.tui.widgets.stats_panel import StatsPanel


class DashboardScreen(Screen):
    """Grading-first split pane dashboard for a selected bandit."""

    CSS = """
    DashboardScreen {
        layout: vertical;
    }

    #dash-header {
        text-style: bold;
        padding: 1 2;
        height: auto;
    }

    #main-area {
        height: 1fr;
    }

    #list-pane {
        width: 30;
        min-width: 24;
        border-right: solid $accent;
    }

    #grading-list {
        height: 1fr;
    }

    #detail-pane {
        width: 1fr;
        padding: 1 2;
    }

    #detail-meta {
        height: auto;
        padding: 0 0 1 0;
        border-bottom: solid $accent;
        margin-bottom: 1;
    }

    .detail-section-header {
        text-style: bold;
        margin-top: 1;
        color: $accent;
    }

    .detail-text {
        margin: 0 0 1 0;
    }

    .detail-markdown {
        margin: 0 0 1 0;
    }

    #detail-hint {
        height: auto;
        padding: 1 0 0 0;
        border-top: solid $accent;
        margin-top: 1;
        color: $text-muted;
    }

    #sidebar {
        width: 36;
        border-left: solid $accent;
        display: none;
        padding: 1;
    }

    #sidebar.visible {
        display: block;
    }

    #sidebar-stats-header, #sidebar-arms-header {
        text-style: bold;
        color: $accent;
        padding: 0 0 0 1;
    }

    #empty-state {
        padding: 4 2;
        text-align: center;
        color: $text-muted;
        display: none;
    }

    #footer-bar {
        height: auto;
        padding: 0 2;
        border-top: solid $accent;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("t", "toggle_sidebar", "Sidebar", show=True),
        Binding("y", "grade_good", "Good", show=True, key_display="y"),
        Binding("n", "grade_bad", "Bad", show=True, key_display="n"),
        Binding("s", "skip", "Skip", show=True),
        Binding("space", "toggle_select", "Select", show=True),
        Binding("a", "select_all", "All", show=True),
        Binding("g", "toggle_graded", "Graded", show=True),
        Binding("j", "cursor_down", show=False),
        Binding("k", "cursor_up", show=False),
        Binding("d", "download_event", "Download", show=True),
        Binding("question_mark", "show_help", "Help", show=True, key_display="?"),
        Binding("q", "quit_app", "Quit", show=True),
    ]

    def __init__(self, bandit: dict[str, Any]) -> None:
        super().__init__()
        self._bandit = bandit
        self._bandit_id: int = bandit["id"]
        self._events: list[dict[str, Any]] = []
        self._skipped_uuids: set[str] = set()
        self._selected_uuids: set[str] = set()
        self._sidebar_loaded: bool = False
        self._show_graded: bool = False

    def compose(self) -> ComposeResult:
        name = self._bandit.get("name", "Unknown")
        yield Static(f"{name}", id="dash-header")

        with Horizontal(id="main-area"):
            with Vertical(id="list-pane"):
                yield ListView(id="grading-list")
            with VerticalScroll(id="detail-pane"):
                yield Static("", id="detail-meta")
                yield Static("USER INPUT", classes="detail-section-header")
                yield Markdown("", id="detail-query-text", classes="detail-markdown")
                yield Static("RESPONSE", classes="detail-section-header")
                yield Markdown("", id="detail-response-text", classes="detail-markdown")
                yield Static("SYSTEM PROMPT", classes="detail-section-header", id="detail-prompt-header")
                yield Markdown("", id="detail-prompt-text", classes="detail-markdown")
                yield Static(
                    "[bold green]y[/] Good  [bold red]n[/] Bad  "
                    "[bold]s[/] Skip  [bold]Space[/] Select",
                    id="detail-hint",
                )
            with Vertical(id="sidebar"):
                yield Static("Stats", id="sidebar-stats-header")
                yield StatsPanel(layout="vertical", id="sidebar-stats")
                yield Static("Arms", id="sidebar-arms-header")
                yield ArmTable(compact=True, id="sidebar-arms")

        yield Static(
            "All caught up! No events to review.",
            id="empty-state",
        )
        yield Static(
            "[dim]esc[/] Back  [dim]r[/] Refresh  [dim]t[/] Sidebar  "
            "[dim]d[/] Download  [dim]g[/] Graded  [dim]?[/] Help  [dim]q[/] Quit",
            id="footer-bar",
        )

    def on_mount(self) -> None:
        # Hide prompt section initially
        self.query_one("#detail-prompt-header", Static).styles.display = "none"
        self.query_one("#detail-prompt-text", Markdown).styles.display = "none"
        self._refresh_grading_queue()
        # Auto-focus the event list so keyboard navigation works immediately
        self.query_one("#grading-list", ListView).focus()

    # ── Data loading (cloud) ───────────────────────────────────────

    @work(thread=True, exit_on_error=False)
    def _load_stats(self) -> dict[str, Any]:
        return self.app.api.get_stats(self._bandit_id)

    @work(thread=True, exit_on_error=False)
    def _load_arms(self) -> dict[str, Any]:
        return self.app.api.get_arm_performance(self._bandit_id)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        name = event.worker.name

        if name == "_load_stats":
            if event.state == WorkerState.SUCCESS:
                self.query_one("#sidebar-stats", StatsPanel).update_stats(
                    event.worker.result
                )
            elif event.state == WorkerState.ERROR:
                self.app.notify(
                    self._friendly_error(event.worker.error),
                    title="Stats", severity="error",
                )

        elif name == "_load_arms":
            if event.state == WorkerState.SUCCESS:
                self.query_one("#sidebar-arms", ArmTable).update_arms(
                    event.worker.result
                )
            elif event.state == WorkerState.ERROR:
                self.app.notify(
                    self._friendly_error(event.worker.error),
                    title="Arm data", severity="error",
                )

        elif name == "_fetch_events":
            if event.state == WorkerState.SUCCESS:
                self._render_grading_queue(event.worker.result)
            elif event.state == WorkerState.ERROR:
                self._update_header(len(self._events))
                self.app.notify(
                    self._friendly_error(event.worker.error),
                    title="Events", severity="error",
                )

        elif name == "_submit_grade":
            if event.state == WorkerState.SUCCESS:
                self.app.notify("Grade submitted", severity="information")
            elif event.state == WorkerState.ERROR:
                self.app.notify(
                    self._friendly_error(event.worker.error),
                    title="Grade failed", severity="error",
                )

    @staticmethod
    def _friendly_error(error: BaseException | None) -> str:
        import httpx
        if isinstance(error, httpx.ConnectError):
            return "Cannot connect to server."
        if isinstance(error, httpx.HTTPStatusError):
            return f"Server returned {error.response.status_code}."
        return str(error) if error else "Unknown error."

    # ── Grading queue (cloud metadata + local text) ─────────────────

    def _refresh_grading_queue(self) -> None:
        """Fetch events from cloud API (metadata) and merge local text."""
        self._fetch_events()

    @work(thread=True, exit_on_error=False, exclusive=True)
    def _fetch_events(self) -> list[dict[str, Any]]:
        """Cloud fetch + local text merge, runs in worker thread."""
        has_human = None if self._show_graded else False
        data = self.app.api.list_events(
            self._bandit_id, has_grade=has_human, limit=50,
        )
        events = data.get("items", [])

        # Merge local text (query_text, response) from SQLite
        store = self.app.store
        if store is not None and events:
            uuids = [e.get("local_event_uuid", "") for e in events]
            local_text = store.get_text(uuids)
            for ev in events:
                uuid = ev.get("local_event_uuid", "")
                if uuid in local_text:
                    text = local_text[uuid]
                    # Local text wins when present (cloud may also have it)
                    if text.get("query_text") is not None:
                        ev["query_text"] = text["query_text"]
                    if text.get("response") is not None:
                        ev["response"] = text["response"]

        return events

    def _render_grading_queue(self, events: list[dict[str, Any]]) -> None:
        # Separate skipped from unskipped, new events go before skipped
        skipped = [e for e in events if e.get("local_event_uuid") in self._skipped_uuids]
        unskipped = [e for e in events if e.get("local_event_uuid") not in self._skipped_uuids]
        self._events = unskipped + skipped

        listview = self.query_one("#grading-list", ListView)
        listview.clear()

        if not self._events:
            self.query_one("#empty-state").styles.display = "block"
            self.query_one("#main-area").styles.display = "none"
            self._update_header(0)
        else:
            self.query_one("#empty-state").styles.display = "none"
            self.query_one("#main-area").styles.display = "block"
            for ev in self._events:
                uuid = ev.get("local_event_uuid", "")
                card = EventCard(ev)
                if ev.get("grade") is not None:
                    card._graded = True
                if uuid in self._skipped_uuids:
                    card._skipped = True
                if uuid in self._selected_uuids:
                    card._selected = True
                listview.append(card)
            listview.index = 0
            self._update_header(len(self._events))
            # Show detail for first event
            self._update_detail_pane(self._events[0])

    def _update_header(self, count: int) -> None:
        name = self._bandit.get("name", "Unknown")
        parts = [name]

        unskipped_count = count - len(
            self._skipped_uuids & {e.get("local_event_uuid") for e in self._events}
        )
        if count > 0:
            parts.append(
                f"  [dim]{unskipped_count} event{'s' if unskipped_count != 1 else ''} awaiting review[/]"
            )

        if self._show_graded:
            parts.append("  [dim italic]showing all[/]")

        if self._selected_uuids:
            parts.append(
                f"  [bold yellow]BATCH: {len(self._selected_uuids)} selected[/]"
            )

        self.query_one("#dash-header", Static).update("".join(parts))

    # ── Detail pane ────────────────────────────────────────────────

    def _update_detail_pane(self, event_data: dict[str, Any]) -> None:
        self._current_event = event_data

        uuid = event_data.get("local_event_uuid", "?")
        model = event_data.get("model_name", "?")
        provider = event_data.get("model_provider", "?")

        cost = event_data.get("cost")
        cost_str = f"${cost:.4f}" if cost is not None else "—"

        latency = event_data.get("latency")
        lat_str = f"{latency:.0f}ms" if latency is not None else "—"

        self.query_one("#detail-meta", Static).update(
            f"[bold]{model}[/] / {provider}  "
            f"[dim]cost:[/] {cost_str}  [dim]latency:[/] {lat_str}  "
            f"[dim]{uuid[:8]}[/]"
        )

        self.query_one("#detail-query-text", Markdown).update(
            str(event_data.get("query_text") or "*no query text*")
        )
        self.query_one("#detail-response-text", Markdown).update(
            format_response(event_data.get("response"))
        )

        prompt = event_data.get("system_prompt")
        if prompt:
            self.query_one("#detail-prompt-header").styles.display = "block"
            self.query_one("#detail-prompt-text").styles.display = "block"
            self.query_one("#detail-prompt-text", Markdown).update(str(prompt))
        else:
            self.query_one("#detail-prompt-header").styles.display = "none"
            self.query_one("#detail-prompt-text").styles.display = "none"

        # Scroll detail pane to top
        self.query_one("#detail-pane", VerticalScroll).scroll_home(animate=False)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Update detail pane as cursor moves."""
        if isinstance(event.item, EventCard):
            self._update_detail_pane(event.item.event_data)

    # ── Grading ────────────────────────────────────────────────────

    def _grade_focused_or_batch(self, reward: float) -> None:
        if self._selected_uuids:
            self._grade_batch(reward)
        else:
            self._grade_focused(reward)

    def _grade_focused(self, reward: float) -> None:
        listview = self.query_one("#grading-list", ListView)
        if listview.index is None:
            return
        highlighted = listview.highlighted_child
        if not isinstance(highlighted, EventCard):
            return

        uuid = highlighted.event_data.get("local_event_uuid")
        if not uuid:
            return

        self._do_grade(uuid, reward)

        # Remove from list
        listview.pop(listview.index)
        self._events = [e for e in self._events if e.get("local_event_uuid") != uuid]
        self._skipped_uuids.discard(uuid)

        self._after_grade_update()

    def _grade_batch(self, reward: float) -> None:
        uuids_to_grade = set(self._selected_uuids)
        count = len(uuids_to_grade)

        for uuid in uuids_to_grade:
            self._do_grade(uuid, reward)

        # Remove graded from list
        self._events = [
            e for e in self._events
            if e.get("local_event_uuid") not in uuids_to_grade
        ]
        self._skipped_uuids -= uuids_to_grade
        self._selected_uuids.clear()

        # Re-render list
        listview = self.query_one("#grading-list", ListView)
        listview.clear()
        for ev in self._events:
            uuid = ev.get("local_event_uuid", "")
            card = EventCard(ev)
            if ev.get("grade") is not None:
                card._graded = True
            if uuid in self._skipped_uuids:
                card._skipped = True
            listview.append(card)
        if self._events:
            listview.index = 0

        self.app.notify(
            f"Graded {count} event{'s' if count != 1 else ''}",
            severity="information",
        )
        self._after_grade_update()

    def _do_grade(self, uuid: str, reward: float) -> None:
        """Mark graded locally + submit to cloud."""
        store = self.app.store
        if store is not None:
            store.mark_graded(uuid, reward)
        self._submit_grade(uuid, reward)

    def _after_grade_update(self) -> None:
        """Update UI state after grading one or more events."""
        if not self._events:
            self.query_one("#empty-state").styles.display = "block"
            self.query_one("#main-area").styles.display = "none"
            self._update_header(0)
        else:
            self._update_header(len(self._events))
            # Update detail pane to show current highlighted item
            listview = self.query_one("#grading-list", ListView)
            highlighted = listview.highlighted_child
            if isinstance(highlighted, EventCard):
                self._update_detail_pane(highlighted.event_data)

    @work(thread=True, exit_on_error=False)
    def _submit_grade(self, uuid: str, reward: float) -> None:
        self.app.api.submit_grade(uuid, reward)

    # ── Skip ───────────────────────────────────────────────────────

    def _skip_focused(self) -> None:
        listview = self.query_one("#grading-list", ListView)
        if listview.index is None:
            return
        highlighted = listview.highlighted_child
        if not isinstance(highlighted, EventCard):
            return

        uuid = highlighted.event_data.get("local_event_uuid")
        if not uuid or uuid in self._skipped_uuids:
            return

        self._skipped_uuids.add(uuid)

        # Move event to end of list
        event_data = highlighted.event_data
        idx = listview.index
        self._events = [e for e in self._events if e.get("local_event_uuid") != uuid]
        self._events.append(event_data)

        # Remove and re-add at end
        listview.pop(idx)
        card = EventCard(event_data)
        card._skipped = True
        listview.append(card)
        # After mounting, set skipped visual
        card.set_skipped(True)

        self._update_header(len(self._events))

    # ── Batch select ───────────────────────────────────────────────

    def _toggle_select(self) -> None:
        listview = self.query_one("#grading-list", ListView)
        if listview.index is None:
            return
        highlighted = listview.highlighted_child
        if not isinstance(highlighted, EventCard):
            return

        uuid = highlighted.event_data.get("local_event_uuid")
        if not uuid:
            return

        if uuid in self._selected_uuids:
            self._selected_uuids.discard(uuid)
            highlighted.set_selected(False)
        else:
            self._selected_uuids.add(uuid)
            highlighted.set_selected(True)

        self._update_header(len(self._events))

    def _select_all(self) -> None:
        listview = self.query_one("#grading-list", ListView)
        for child in listview.children:
            if isinstance(child, EventCard):
                uuid = child.event_data.get("local_event_uuid")
                if uuid and uuid not in self._skipped_uuids:
                    self._selected_uuids.add(uuid)
                    child.set_selected(True)
        self._update_header(len(self._events))

    # ── Sidebar ────────────────────────────────────────────────────

    def _toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.toggle_class("visible")

        if not self._sidebar_loaded and sidebar.has_class("visible"):
            self._sidebar_loaded = True
            self._load_stats()
            self._load_arms()

    # ── Actions ─────────────────────────────────────────────────────

    def action_grade_good(self) -> None:
        self._grade_focused_or_batch(1.0)

    def action_grade_bad(self) -> None:
        self._grade_focused_or_batch(0.0)

    def action_skip(self) -> None:
        self._skip_focused()

    def action_toggle_select(self) -> None:
        self._toggle_select()

    def action_select_all(self) -> None:
        self._select_all()

    def action_toggle_graded(self) -> None:
        self._show_graded = not self._show_graded
        self._selected_uuids.clear()
        # Clear current list so the new fetch always triggers a re-render
        self._events = []
        self._refresh_grading_queue()
        label = "all events" if self._show_graded else "ungraded only"
        self.app.notify(f"Showing {label}")

    def action_download_event(self) -> None:
        ev = getattr(self, "_current_event", None)
        if ev is None:
            return
        from bandito.tui.utils import save_event_json
        path = save_event_json(ev)
        self.app.notify(f"Saved to {path}", severity="information")

    def action_toggle_sidebar(self) -> None:
        self._toggle_sidebar()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        name = self._bandit.get("name", "Unknown")
        self.query_one("#dash-header", Static).update(
            f"{name}  [dim italic]refreshing...[/]"
        )
        self._refresh_grading_queue()
        if self._sidebar_loaded:
            self._load_stats()
            self._load_arms()

    def action_cursor_down(self) -> None:
        listview = self.query_one("#grading-list", ListView)
        listview.action_cursor_down()

    def action_cursor_up(self) -> None:
        listview = self.query_one("#grading-list", ListView)
        listview.action_cursor_up()

    def action_show_help(self) -> None:
        from bandito.tui.screens.help import HelpScreen
        self.app.push_screen(HelpScreen())

    def action_quit_app(self) -> None:
        self.app.exit()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Open event detail on Enter."""
        if isinstance(event.item, EventCard):
            from bandito.tui.screens.event_detail import EventDetailScreen
            self.app.push_screen(
                EventDetailScreen(event.item.event_data),
                callback=self._on_detail_dismiss,
            )

    def _on_detail_dismiss(self, grade: float | None) -> None:
        """Handle grade returned from event detail modal."""
        if grade is None:
            return
        self._grade_focused(grade)
