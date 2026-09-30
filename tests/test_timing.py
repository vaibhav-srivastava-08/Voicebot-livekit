"""Tests for shared.timing: RollingStats percentiles and TurnLatencyTracker's event hooks.

TurnLatencyTracker is exercised against a plain `rtc.EventEmitter` standing in for an
AgentSession - the same base class AgentSession itself extends - and real LiveKit event/model
objects (UserInputTranscribedEvent, ChatMessage, etc.), so no live LiveKit connection, room, or
provider is needed. The logger passed in is a MagicMock, so log records are inspected directly
via call_args rather than parsing formatted log output.
"""

from unittest.mock import MagicMock

from livekit import rtc
from livekit.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.metrics import LLMMetrics
from livekit.agents.voice.events import (
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
)

from shared.timing import RollingStats, TurnLatencyTracker


def _llm_metrics(duration: float) -> LLMMetrics:
    return LLMMetrics(
        label="test-llm",
        request_id="req-1",
        timestamp=0.0,
        duration=duration,
        ttft=0.0,
        cancelled=False,
        completion_tokens=10,
        prompt_tokens=10,
        prompt_cached_tokens=0,
        total_tokens=20,
        tokens_per_second=10.0,
    )


def _user_message(transcription_delay: float | None) -> ChatMessage:
    metrics = {"transcription_delay": transcription_delay} if transcription_delay is not None else {}
    return ChatMessage(role="user", content=["hello"], metrics=metrics)


def _assistant_message(*, ttft: float, ttfb: float, e2e_latency: float) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=["hi there"],
        metrics={
            "llm_node_ttft": ttft,
            "tts_node_ttfb": ttfb,
            "e2e_latency": e2e_latency,
        },
    )


class TestRollingStats:
    def test_percentiles_is_none_when_empty(self):
        assert RollingStats().percentiles() is None

    def test_percentiles_with_single_sample(self):
        stats = RollingStats()
        stats.add(1.5)
        assert stats.percentiles() == {"count": 1, "p50": 1.5, "p95": 1.5}

    def test_percentiles_shape_with_many_samples(self):
        stats = RollingStats()
        for value in range(1, 101):  # 1..100
            stats.add(float(value))
        result = stats.percentiles()
        assert result["count"] == 100
        # p50 should sit near the middle, p95 near the top, and p50 < p95.
        assert 40 <= result["p50"] <= 60
        assert 90 <= result["p95"] <= 100
        assert result["p50"] < result["p95"]

    def test_rolling_window_is_bounded(self):
        stats = RollingStats(maxlen=5)
        for value in range(10):
            stats.add(float(value))
        assert stats.percentiles()["count"] == 5


class TestTurnLatencyTracker:
    def _make_tracker(self):
        logger = MagicMock()
        tracker = TurnLatencyTracker(logger=logger)
        session = rtc.EventEmitter()
        tracker.attach(session)
        return tracker, session, logger

    def test_full_turn_emits_a_well_formed_turn_latency_record(self):
        tracker, session, logger = self._make_tracker()

        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="hel", is_final=False),
        )
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="hello there", is_final=True),
        )
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(item=_user_message(transcription_delay=0.12)),
        )
        session.emit(
            "function_tools_executed",
            FunctionToolsExecutedEvent(
                function_calls=[
                    FunctionCall(call_id="call-1", arguments="{}", name="search_knowledge_base")
                ],
                function_call_outputs=[
                    FunctionCallOutput(call_id="call-1", output="ok", is_error=False)
                ],
            ),
        )
        session.emit("metrics_collected", MetricsCollectedEvent(metrics=_llm_metrics(0.85)))
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(
                item=_assistant_message(ttft=0.31, ttfb=0.09, e2e_latency=1.4)
            ),
        )

        turn_calls = [
            call for call in logger.info.call_args_list if call.args[0] == "turn latency"
        ]
        assert len(turn_calls) == 1
        record = turn_calls[0].kwargs["extra"]

        assert record["event"] == "turn_latency"
        assert record["stt_final_transcript_at"] is not None
        assert record["stt_transcription_delay_s"] == 0.12
        assert record["llm_ttft_s"] == 0.31
        assert record["llm_duration_s"] == 0.85
        assert record["tts_ttfb_s"] == 0.09
        assert record["total_turn_latency_s"] == 1.4
        assert record["tool_called"] is True
        assert record["tool_names"] == ["search_knowledge_base"]

    def test_turn_without_tool_call_reports_tool_called_false(self):
        tracker, session, logger = self._make_tracker()

        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="hi", is_final=True),
        )
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(item=_user_message(transcription_delay=0.05)),
        )
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(
                item=_assistant_message(ttft=0.2, ttfb=0.1, e2e_latency=0.9)
            ),
        )

        record = logger.info.call_args_list[-1].kwargs["extra"]
        assert record["tool_called"] is False
        assert record["tool_names"] == []

    def test_pending_state_resets_between_turns(self):
        tracker, session, logger = self._make_tracker()

        def run_turn(delay, ttft, ttfb, e2e, with_tool):
            session.emit(
                "user_input_transcribed",
                UserInputTranscribedEvent(transcript="x", is_final=True),
            )
            session.emit(
                "conversation_item_added",
                ConversationItemAddedEvent(item=_user_message(transcription_delay=delay)),
            )
            if with_tool:
                session.emit(
                    "function_tools_executed",
                    FunctionToolsExecutedEvent(
                        function_calls=[
                            FunctionCall(call_id="c", arguments="{}", name="search_knowledge_base")
                        ],
                        function_call_outputs=[
                            FunctionCallOutput(call_id="c", output="ok", is_error=False)
                        ],
                    ),
                )
            session.emit(
                "conversation_item_added",
                ConversationItemAddedEvent(item=_assistant_message(ttft=ttft, ttfb=ttfb, e2e_latency=e2e)),
            )

        run_turn(0.1, 0.2, 0.05, 0.5, with_tool=True)
        run_turn(0.15, 0.25, 0.06, 0.6, with_tool=False)

        second_record = logger.info.call_args_list[-1].kwargs["extra"]
        assert second_record["tool_called"] is False
        assert second_record["tool_names"] == []
        assert second_record["stt_transcription_delay_s"] == 0.15

    def test_session_close_logs_rolling_summary(self):
        tracker, session, logger = self._make_tracker()

        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="hi", is_final=True),
        )
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(item=_user_message(transcription_delay=0.1)),
        )
        session.emit(
            "conversation_item_added",
            ConversationItemAddedEvent(
                item=_assistant_message(ttft=0.2, ttfb=0.1, e2e_latency=0.7)
            ),
        )

        session.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))

        summary_calls = [
            call
            for call in logger.info.call_args_list
            if call.args[0] == "turn latency call summary"
        ]
        assert len(summary_calls) == 1
        summary = summary_calls[0].kwargs["extra"]
        assert summary["event"] == "turn_latency_summary"
        assert summary["stages"]["llm_ttft"] == {"count": 1, "p50": 0.2, "p95": 0.2}
        assert summary["stages"]["total_turn_latency"] == {"count": 1, "p50": 0.7, "p95": 0.7}
