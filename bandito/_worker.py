"""Background daemon thread — periodic sync heartbeat + event flush."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from bandito.http import BanditoHTTP
    from bandito.store import EventStore

logger = logging.getLogger("bandito")


class BackgroundWorker:
    """Single daemon thread that handles:

    1. Flushing pending events from SQLite to cloud (every flush_interval seconds)
    2. Periodic heartbeat / state refresh (every sync_interval seconds)

    Both survive HTTP errors — the thread logs warnings and retries next cycle.
    """

    def __init__(
        self,
        http: BanditoHTTP,
        store: EventStore,
        on_sync: Callable[[dict], None],
        *,
        sync_interval: float = 30.0,
        flush_interval: float = 5.0,
    ) -> None:
        self._http = http
        self._store = store
        self._on_sync = on_sync
        self._sync_interval = sync_interval
        self._flush_interval = flush_interval
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._wake_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()  # wake up so it exits promptly
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def trigger_flush(self) -> None:
        """Wake the worker to flush events immediately."""
        self._wake_event.set()

    def _run(self) -> None:
        last_sync = time.monotonic()
        while not self._stop_event.is_set():
            # Sleep until timeout or trigger_flush() wakes us
            self._wake_event.wait(timeout=self._flush_interval)
            self._wake_event.clear()

            if self._stop_event.is_set():
                break

            # Flush pending events
            self._flush()

            # Heartbeat on schedule (uses actual elapsed time, not assumed interval)
            if time.monotonic() - last_sync >= self._sync_interval:
                self._heartbeat()
                last_sync = time.monotonic()

    def _flush(self) -> None:
        try:
            pending = self._store.pending()
            if not pending:
                return
            result = self._http.ingest_events(pending)
            # Mark all sent events as flushed (server handles dedup)
            uuids = [e["local_event_uuid"] for e in pending]
            self._store.mark_flushed(uuids)
            logger.debug(
                "Flushed %d events (accepted=%d, duplicates=%d)",
                len(pending),
                result.get("accepted", 0),
                result.get("duplicates", 0),
            )
        except Exception:
            logger.warning("Event flush failed — will retry next cycle", exc_info=True)

    def _heartbeat(self) -> None:
        try:
            data = self._http.heartbeat()
            self._on_sync(data)
            logger.debug("Heartbeat sync: %d bandits", len(data.get("bandits", [])))
        except Exception:
            logger.warning("Heartbeat failed — continuing with stale weights", exc_info=True)
