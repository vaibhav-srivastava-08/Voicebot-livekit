"""Reusable per-call latency instrumentation.

`RollingStats` keeps a bounded window of samples for one stage and reports p50/p95.
`TurnLatencyTracker` hooks an `AgentSession`'s own events - `user_input_transcribed`,
`metrics_collected`, `function_tools_executed`, `conversation_item_added` - and the
`ChatMessage.metrics` LiveKit Agents already attaches to each turn, to emit one structured
JSON log line per conversational turn. No STT/LLM/TTS provider is wrapped or patched
directly; everything comes from the pipeline's own event stream.
"""

from __future__ import annotations

import statistics
from collections import deque
from typing import Any


class RollingStats:
    """Bounded rolling window of float samples for one named stage."""

    def __init__(self, maxlen: int = 200) -> None:
        self._samples: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._samples.append(value)

    def percentiles(self) -> dict[str, float] | None:
        """Return {count, p50, p95}, or None if no samples have been recorded yet."""
        if not self._samples:
            return None
        data = sorted(self._samples)
        if len(data) == 1:
            return {"count": 1, "p50": data[0], "p95": data[0]}
        quantiles = statistics.quantiles(data, n=100, method="inclusive")
        return {"count": len(data), "p50": quantiles[49], "p95": quantiles[94]}


# Stages tracked in the rolling summary - the duration-shaped fields of a turn record.
# (stt_final_transcript_at is an absolute timestamp, not a duration, so it isn't in here.)
_STAGES = ("stt_transcription_delay", "llm_ttft", "llm_duration", "tts_ttfb", "total_turn_latency")


class TurnLatencyTracker:
    """Tracks per-turn latency for one call by listening to one AgentSession's events.

    A "turn" runs from the user's final transcript through the agent's completed reply:
      user_input_transcribed(is_final) -> ... -> conversation_item_added(user message)
      -> [metrics_collected(llm_metrics) / function_tools_executed]* while the agent thinks
      -> conversation_item_added(assistant message)  <- turn boundary, record emitted here

    Attach once per call via `attach(session)`; a structured "turn_latency" log line is
    emitted after every completed assistant turn, and a "turn_latency_summary" line (rolling
    p50/p95 per stage) is emitted when the session closes.
    """

    def __init__(self, logger: Any, rolling_window: int = 200) -> None:
        self._logger = logger
        self._stats = {stage: RollingStats(maxlen=rolling_window) for stage in _STAGES}
        self._reset_pending()

    def _reset_pending(self) -> None:
        self._stt_final_transcript_at: float | None = None
        self._stt_transcription_delay: float | None = None
        self._llm_duration_total: float = 0.0
        self._tool_called: bool = False
        self._tool_names: list[str] = []

    def attach(self, session: Any) -> None:
        """Register this tracker's listeners on `session`. Call once per AgentSession."""
        session.on("user_input_transcribed", self._on_user_input_transcribed)
        session.on("metrics_collected", self._on_metrics_collected)
        session.on("function_tools_executed", self._on_function_tools_executed)
        session.on("conversation_item_added", self._on_conversation_item_added)
        session.on("close", lambda ev: self.log_summary())

    def _on_user_input_transcribed(self, ev: Any) -> None:
        if ev.is_final:
            self._stt_final_transcript_at = ev.created_at

    def _on_metrics_collected(self, ev: Any) -> None:
        if ev.metrics.type == "llm_metrics":
            self._llm_duration_total += ev.metrics.duration

    def _on_function_tools_executed(self, ev: Any) -> None:
        self._tool_called = True
        self._tool_names.extend(call.name for call in ev.function_calls)

    def _on_conversation_item_added(self, ev: Any) -> None:
        item = ev.item
        role = getattr(item, "role", None)
        if getattr(item, "type", None) != "message" or role not in ("user", "assistant"):
            return

        if role == "user":
            delay = item.metrics.get("transcription_delay")
            if delay is not None:
                self._stt_transcription_delay = delay
            return

        # Assistant turn complete - this is the turn boundary.
        metrics = item.metrics
        record = {
            "event": "turn_latency",
            "stt_final_transcript_at": self._stt_final_transcript_at,
            "stt_transcription_delay_s": self._stt_transcription_delay,
            "llm_ttft_s": metrics.get("llm_node_ttft"),
            "llm_duration_s": self._llm_duration_total or None,
            "tts_ttfb_s": metrics.get("tts_node_ttfb"),
            "total_turn_latency_s": metrics.get("e2e_latency"),
            "tool_called": self._tool_called,
            "tool_names": list(self._tool_names),
        }
        self._logger.info("turn latency", extra=record)

        stage_values = {
            "stt_transcription_delay": self._stt_transcription_delay,
            "llm_ttft": record["llm_ttft_s"],
            "llm_duration": record["llm_duration_s"],
            "tts_ttfb": record["tts_ttfb_s"],
            "total_turn_latency": record["total_turn_latency_s"],
        }
        for stage, value in stage_values.items():
            if value is not None:
                self._stats[stage].add(value)

        self._reset_pending()

    def log_summary(self) -> None:
        """Log the rolling p50/p95-per-stage summary. Call at end of call (wired to the
        session's own "close" event by attach(), or call directly)."""
        summary = {stage: stats.percentiles() for stage, stats in self._stats.items()}
        self._logger.info(
            "turn latency call summary",
            extra={"event": "turn_latency_summary", "stages": summary},
        )
