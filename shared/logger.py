"""Structured JSON logging configured with call/stream ID context for correlating logs."""

import logging
import os
import sys

from pythonjsonlogger import jsonlogger

_ROOT_LOGGER_NAME = "voicebot"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(module)s %(message)s"

_configured = False


def _configure_root() -> None:
    """Attach a single stdout JSON handler to the root voicebot logger, once per process."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            _LOG_FORMAT,
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(os.environ.get("AGENT_LOG_LEVEL", "INFO").upper())
    root.addHandler(handler)
    root.propagate = False

    _configured = True


class CallContextAdapter(logging.LoggerAdapter):
    """LoggerAdapter that injects bound stream_id/room_name into every emitted log record."""

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.update(self.extra)
        return msg, kwargs


def get_logger(
    name: str,
    stream_id: str | None = None,
    room_name: str | None = None,
) -> logging.LoggerAdapter:
    """Return a JSON-structured logger for `name` with stream_id/room_name bound, if given."""
    _configure_root()

    context = {}
    if stream_id is not None:
        context["stream_id"] = stream_id
    if room_name is not None:
        context["room_name"] = room_name

    logger = logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
    return CallContextAdapter(logger, context)
