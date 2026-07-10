"""
Event logger: publishes recommendation-served and interaction events to
Kafka for offline evaluation, per the architecture diagram
("Event Logger (Kafka) -> offline evaluation").

Since this project is meant to run locally with `uvicorn app.main:app`
without requiring a running Kafka broker, this module tries to connect to
Kafka and transparently falls back to appending JSON lines to a local
file if the broker isn't reachable. In a real deployment you'd remove the
fallback and let connection failures page someone - logging silently
dropping events is a real production bug class, so we log loudly (via a
`degraded` flag surfaced on the logger) rather than pretending it's fine.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

logger = logging.getLogger("event_logger")

LOCAL_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "events.jsonl"
CONNECT_TIMEOUT_S = 2.0
SEND_TIMEOUT_S = 1.0


class EventLogger:
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "recommendation_events"):
        self.topic = topic
        self.degraded = False
        self._producer = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

        def _connect():
            from kafka import KafkaProducer  # imported lazily so kafka-python isn't a hard requirement to boot

            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=1500,
                max_block_ms=1500,
                api_version=(2, 0, 0),
            )
            # bootstrap_connected() is optimistic in kafka-python (it can return True
            # before ever proving a broker is reachable), so we don't trust it alone -
            # everything here runs behind a hard wall-clock timeout instead. If there's
            # truly no broker, this either raises quickly or we hit the executor timeout.
            producer.partitions_for(topic)
            return producer

        try:
            future = self._executor.submit(_connect)
            self._producer = future.result(timeout=CONNECT_TIMEOUT_S)
        except (FutureTimeoutError, Exception) as exc:  # noqa: BLE001 - broad on purpose, this is a connectivity probe
            logger.warning("Kafka unavailable (%s) - falling back to local JSONL log at %s", exc, LOCAL_LOG_PATH)
            self._producer = None
            self.degraded = True
            LOCAL_LOG_PATH.parent.mkdir(exist_ok=True, parents=True)

    def log_event(self, event: dict) -> None:
        event = {**event, "logged_at": time.time()}
        if self._producer is not None:
            try:
                future = self._executor.submit(self._producer.send, self.topic, value=event)
                future.result(timeout=SEND_TIMEOUT_S)
                return
            except (FutureTimeoutError, Exception) as exc:  # noqa: BLE001
                logger.error("Kafka publish failed/timed out (%s), falling back to local log for this event", exc)
                self.degraded = True
                self._producer = None  # stop retrying a broker that's not there

        with self._lock:
            with open(LOCAL_LOG_PATH, "a") as f:
                f.write(json.dumps(event) + "\n")

    def log_recommendation_served(self, user_id: int, variant: str, item_ids: list[int], request_id: str) -> None:
        self.log_event(
            {
                "event_type": "recommendation_served",
                "user_id": user_id,
                "variant": variant,
                "item_ids": item_ids,
                "request_id": request_id,
            }
        )

    def log_interaction(self, user_id: int, item_id: int, action: str, variant: str, request_id: str | None = None) -> None:
        self.log_event(
            {
                "event_type": "interaction",
                "user_id": user_id,
                "item_id": item_id,
                "action": action,
                "variant": variant,
                "request_id": request_id,
            }
        )


_singleton: EventLogger | None = None


def get_event_logger() -> EventLogger:
    global _singleton
    if _singleton is None:
        _singleton = EventLogger()
    return _singleton
