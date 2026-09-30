"""Tails the agent's JSON logs for the duration of a load-test run, so scripts/load/harness.py
can correlate what it observed at the connection level (did the room join succeed, did it time
out) with what the agent itself logged for that same room (per-turn latency, provider errors).

Every agent log line is JSON (see shared/logger.py), whether it comes from this project's own
loggers (get_logger("turn_latency", room_name=...) -> shared/timing.py's TurnLatencyTracker) or
from livekit-agents' own internal logger - both are still one-JSON-object-per-line, just with
different field sets. That's all this module relies on: parse every line as JSON, keep the ones
that parse, and let callers filter/aggregate.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

# Log lines whose message/logger name suggests a provider (STT/LLM/TTS) call failed - used to
# surface "where Groq/Cartesia rate limits kick in" even though those errors aren't tagged with
# a room_name (they come from livekit-agents' own internal logger, not ours - see module
# docstring). Matched case-insensitively against the whole raw JSON line.
_PROVIDER_PROBLEM_KEYWORDS = (
    "ratelimit",
    "rate limit",
    "rate_limit_exceeded",
    "429",
    "413",  # Groq's code for "request too large for this tier's tokens-per-minute limit"
    "apistatuserror",
    "apiconnectionerror",
    "apitimeouterror",
    "llmerror",
    "ttserror",
    "stterror",
)


@dataclass
class LogRecord:
    received_at: float  # time.monotonic() when this harness process read the line
    data: dict


@dataclass
class LogTail:
    """Runs `cmd` as a subprocess and parses its stdout as one JSON object per line for as
    long as the tail is running. Works equally with `docker compose logs -f --no-log-prefix
    <service>` (this class doesn't know or care that it's Docker) or `tail -F` on a log file
    the agent's own stdout was redirected to."""

    cmd: list[str]
    records: list[LogRecord] = field(default_factory=list)
    _proc: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _pump_task: asyncio.Task | None = field(default=None, repr=False)

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        async for raw_line in self._proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON output (deprecation warnings, tracebacks, ...) - ignore
            if not isinstance(data, dict):
                continue
            self.records.append(LogRecord(received_at=time.monotonic(), data=data))

    async def stop(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()

    def turn_latency_records_for_rooms(self, room_names: set[str]) -> list[dict]:
        return [
            r.data
            for r in self.records
            if r.data.get("event") == "turn_latency" and r.data.get("room_name") in room_names
        ]

    def turn_latency_summary_records_for_rooms(self, room_names: set[str]) -> list[dict]:
        return [
            r.data
            for r in self.records
            if r.data.get("event") == "turn_latency_summary" and r.data.get("room_name") in room_names
        ]

    def rooms_with_any_activity(self, room_names: set[str]) -> set[str]:
        """Rooms that appear anywhere in agent logs at all (e.g. "agent starting") - used to
        tell "the agent never picked this call up" apart from "it picked it up but never
        completed a turn"."""
        seen = set()
        for r in self.records:
            room_name = r.data.get("room_name")
            if room_name in room_names:
                seen.add(room_name)
        return seen

    def provider_problem_count_in_window(self, start: float, end: float) -> int:
        """Count of ERROR/WARNING-level log lines that look provider-related (see
        _PROVIDER_PROBLEM_KEYWORDS), received between `start` and `end` (time.monotonic()
        timestamps) - a level-wide signal since these lines aren't tagged with room_name.

        Matches against the "message" field only, not the whole JSON record: matching the
        full record (including "timestamp") let a millisecond-precision timestamp like
        "...10:31:07.413491+00:00" false-positive-match the "413" keyword - a real bug caught
        by testing this against live logs, not a hypothetical."""
        count = 0
        for r in self.records:
            if not (start <= r.received_at <= end):
                continue
            if r.data.get("level") not in ("ERROR", "WARNING"):
                continue
            haystack = str(r.data.get("message", "")).lower()
            if any(keyword in haystack for keyword in _PROVIDER_PROBLEM_KEYWORDS):
                count += 1
        return count
