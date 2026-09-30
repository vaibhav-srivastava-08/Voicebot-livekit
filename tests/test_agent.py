"""Tests for agent.assistant: the Assistant class and build_session() wiring.

All external speech/LLM clients (Deepgram, Cartesia, Groq, Silero, the turn detector model)
are mocked - no real API calls or model downloads happen when these tests run.
"""

import asyncio
import os
from unittest.mock import MagicMock

# agent.config builds its settings singleton at import time, which agent.assistant triggers
# transitively. Seed harmless placeholder values so that import never depends on a real .env
# or real secrets being present in the environment these tests run in.
os.environ.setdefault("LIVEKIT_URL", "wss://example.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test-livekit-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-livekit-secret")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram-key")
os.environ.setdefault("CARTESIA_API_KEY", "test-cartesia-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("CARTESIA_VOICE_ID", "test-voice-id")

import pytest
from livekit.agents import AgentSession, UserInputTranscribedEvent
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput, is_function_tool

from agent import assistant


def _ctx_with_user_message(text: str) -> ChatContext:
    ctx = ChatContext()
    ctx.add_message(role="user", content=text)
    return ctx


def _ctx_after_kb_call(user_text: str, kb_output: str) -> ChatContext:
    """A chat context shaped like a real turn that called search_knowledge_base: a user
    message, then the tool's call+output, matching what _recent_kb_context looks for."""
    ctx = ChatContext()
    ctx.add_message(role="user", content=user_text)
    ctx.insert(
        [
            FunctionCall(call_id="call-1", arguments="{}", name="search_knowledge_base"),
            FunctionCallOutput(
                call_id="call-1", name="search_knowledge_base", output=kb_output, is_error=False
            ),
        ]
    )
    return ctx


@pytest.fixture
def mocked_plugins(monkeypatch):
    """Replace every speech/LLM plugin build_session() touches with a mock, so no real
    API calls or model downloads (e.g. silero.VAD.load()) ever happen in tests.

    build_llm/build_stt (agent.llm_factory.build_llm / agent.stt_factory.build_stt, imported
    into assistant's namespace) are mocked as single units rather than reaching into the
    provider-specific plugin classes they can construct (groq.LLM/google.LLM/openai.LLM,
    deepgram.STT/soniox.STT) - those are covered in isolation by tests/test_llm_factory.py and
    tests/test_stt_factory.py instead."""
    monkeypatch.setattr(assistant, "build_stt", MagicMock(return_value=MagicMock(name="stt")))
    monkeypatch.setattr(assistant.cartesia, "TTS", MagicMock(return_value=MagicMock(name="tts")))
    monkeypatch.setattr(assistant, "build_llm", MagicMock(return_value=MagicMock(name="llm")))
    monkeypatch.setattr(
        assistant.silero,
        "VAD",
        MagicMock(load=MagicMock(return_value=MagicMock(name="vad"))),
    )
    monkeypatch.setattr(
        assistant, "MultilingualModel", MagicMock(return_value=MagicMock(name="turn_detector"))
    )


def test_build_session_returns_agent_session_instance(mocked_plugins):
    session = assistant.build_session()
    assert isinstance(session, AgentSession)


def test_build_session_wires_mocked_components(mocked_plugins):
    assistant.build_session()

    assistant.build_stt.assert_called_once_with()
    assistant.build_llm.assert_called_once_with(assistant.settings.smart_llm_model)
    assistant.cartesia.TTS.assert_called_once()
    assistant.silero.VAD.load.assert_called_once()
    assistant.MultilingualModel.assert_called_once()


def test_assistant_instantiates_without_error():
    agent_instance = assistant.Assistant()
    assert isinstance(agent_instance, assistant.Assistant)


def test_search_knowledge_base_is_a_function_tool():
    assert is_function_tool(assistant.search_knowledge_base)


def test_assistant_instance_exposes_search_knowledge_base_tool():
    agent_instance = assistant.Assistant()
    tool_names = {tool.info.name for tool in agent_instance.tools if is_function_tool(tool)}
    assert "search_knowledge_base" in tool_names


def test_assistant_instance_exposes_end_call_toolset():
    agent_instance = assistant.Assistant()
    assert any(isinstance(tool, EndCallTool) for tool in agent_instance.tools)


def test_assistant_accepts_per_call_instructions():
    custom_instructions = "Custom per-call instructions with {{lead_name}} filled in already."
    agent_instance = assistant.Assistant(instructions=custom_instructions)
    assert agent_instance.instructions == custom_instructions


def test_build_session_passes_preemptive_generation_setting(mocked_plugins, monkeypatch):
    mock_session_cls = MagicMock(return_value=MagicMock(name="session"))
    monkeypatch.setattr(assistant, "AgentSession", mock_session_cls)
    monkeypatch.setattr(assistant.settings, "preemptive_generation_enabled", False)

    assistant.build_session()

    _, kwargs = mock_session_cls.call_args
    assert kwargs["turn_handling"]["preemptive_generation"] == {"enabled": False}


def test_build_session_passes_endpointing_min_delay_setting(mocked_plugins, monkeypatch):
    mock_session_cls = MagicMock(return_value=MagicMock(name="session"))
    monkeypatch.setattr(assistant, "AgentSession", mock_session_cls)
    monkeypatch.setattr(assistant.settings, "endpointing_min_delay_s", 0.8)

    assistant.build_session()

    _, kwargs = mock_session_cls.call_args
    assert kwargs["turn_handling"]["endpointing"] == {"min_delay": 0.8}


def test_build_session_forces_vad_interruption_mode(mocked_plugins, monkeypatch):
    # "adaptive" mode needs LiveKit's hosted inference gateway, unreachable from this
    # self-hosted deployment - see the comment in agent.assistant.build_session.
    mock_session_cls = MagicMock(return_value=MagicMock(name="session"))
    monkeypatch.setattr(assistant, "AgentSession", mock_session_cls)

    assistant.build_session()

    _, kwargs = mock_session_cls.call_args
    assert kwargs["turn_handling"]["interruption"] == {"mode": "vad"}


def test_build_session_tts_text_transforms_include_speech_normalization(mocked_plugins, monkeypatch):
    mock_session_cls = MagicMock(return_value=MagicMock(name="session"))
    monkeypatch.setattr(assistant, "AgentSession", mock_session_cls)

    assistant.build_session()

    _, kwargs = mock_session_cls.call_args
    transforms = kwargs["tts_text_transforms"]
    # filter_markdown/filter_emoji are livekit-agents' own defaults - specifying
    # tts_text_transforms at all replaces rather than appends to them, so they must be listed
    # explicitly alongside this project's own normalize_for_speech (see agent/tts_text.py).
    assert transforms == ["filter_markdown", "filter_emoji", assistant.normalize_for_speech]


def test_build_session_sets_userdata_with_no_detected_language_yet(mocked_plugins):
    session = assistant.build_session()
    assert isinstance(session.userdata, assistant.VoicebotUserdata)
    assert session.userdata.last_detected_language is None


def test_build_session_tracks_final_transcript_language(mocked_plugins):
    session = assistant.build_session()

    session.emit(
        "user_input_transcribed",
        UserInputTranscribedEvent(transcript="namaste", is_final=True, language="hi"),
    )

    assert session.userdata.last_detected_language == "hi"


def test_build_session_ignores_interim_transcripts_for_language_tracking(mocked_plugins):
    session = assistant.build_session()

    session.emit(
        "user_input_transcribed",
        UserInputTranscribedEvent(transcript="nam", is_final=False, language="hi"),
    )

    assert session.userdata.last_detected_language is None


class TestTurnNeedsKnowledgeBase:
    """Covers the routing decision's pure keyword-screen logic in isolation."""

    def test_true_for_a_price_question(self):
        ctx = _ctx_with_user_message("What's the price of the Brezza?")
        assert assistant._turn_needs_knowledge_base(ctx) is True

    def test_true_for_a_bare_model_name_mention(self):
        ctx = _ctx_with_user_message("Tell me about the Swift")
        assert assistant._turn_needs_knowledge_base(ctx) is True

    def test_true_for_a_comparison_question(self):
        ctx = _ctx_with_user_message("What's the difference between Brezza and Fronx?")
        assert assistant._turn_needs_knowledge_base(ctx) is True

    def test_false_for_confirming_identity(self):
        ctx = _ctx_with_user_message("Yes, that's me speaking.")
        assert assistant._turn_needs_knowledge_base(ctx) is False

    def test_false_for_giving_a_name(self):
        ctx = _ctx_with_user_message("My name is Rahul.")
        assert assistant._turn_needs_knowledge_base(ctx) is False

    def test_false_when_no_user_message_present(self):
        ctx = ChatContext()
        ctx.add_message(role="assistant", content="Hello!")
        assert assistant._turn_needs_knowledge_base(ctx) is False

    def test_uses_only_the_most_recent_user_message(self):
        ctx = ChatContext()
        ctx.add_message(role="user", content="What's the price of the Brezza?")
        ctx.add_message(role="assistant", content="It's approximately eight lakh.")
        ctx.add_message(role="user", content="Okay thanks, that's all.")
        assert assistant._turn_needs_knowledge_base(ctx) is False


class TestTwoTierLLMRouting:
    """Covers Assistant's two-tier routing: fast-LLM construction and per-turn selection."""

    def test_fast_llm_is_none_when_two_tier_disabled(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "enable_two_tier_llm", False)
        agent_instance = assistant.Assistant()
        assert agent_instance._fast_llm is None

    def test_fast_llm_is_constructed_with_configured_model_when_two_tier_enabled(
        self, monkeypatch
    ):
        monkeypatch.setattr(assistant.settings, "enable_two_tier_llm", True)
        monkeypatch.setattr(
            assistant, "build_llm", MagicMock(return_value=MagicMock(name="fast-llm"))
        )

        agent_instance = assistant.Assistant()

        assert agent_instance._fast_llm is not None
        assistant.build_llm.assert_called_once_with(assistant.settings.fast_llm_model)

    def test_select_fast_llm_returns_none_when_two_tier_disabled(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "enable_two_tier_llm", False)
        agent_instance = assistant.Assistant()

        ctx = _ctx_with_user_message("Yes, that's me.")
        assert agent_instance._select_fast_llm(ctx) is None

    def test_select_fast_llm_picks_fast_tier_for_non_kb_turn(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "enable_two_tier_llm", True)
        monkeypatch.setattr(
            assistant, "build_llm", MagicMock(return_value=MagicMock(name="fast-llm"))
        )
        agent_instance = assistant.Assistant()

        ctx = _ctx_with_user_message("Yes, that's me speaking.")
        assert agent_instance._select_fast_llm(ctx) is agent_instance._fast_llm

    def test_select_fast_llm_picks_smart_tier_for_kb_turn(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "enable_two_tier_llm", True)
        monkeypatch.setattr(
            assistant, "build_llm", MagicMock(return_value=MagicMock(name="fast-llm"))
        )
        agent_instance = assistant.Assistant()

        ctx = _ctx_with_user_message("What's the price of the Brezza?")
        assert agent_instance._select_fast_llm(ctx) is None


_BREZZA_PRICE_CONTEXT = (
    "Brezza - ARENA SUV: Ex-showroom price range: 8 lakh 11 thousand 400 to 13 lakh 1 thousand 300"
)


class TestResponseGuard:
    """Covers the number-verification safety net wired into Assistant.llm_node: it must catch
    a hallucinated price/spec that doesn't appear in the KB text retrieved that turn, while
    leaving turns that didn't call search_knowledge_base (and legitimate KB-backed replies)
    untouched."""

    def _mock_default_llm_node(self, monkeypatch, reply_text: str):
        async def fake_default_llm_node(agent, chat_ctx, tools, model_settings):
            yield reply_text

        monkeypatch.setattr(assistant.Agent.default, "llm_node", fake_default_llm_node)
        # These tests exercise the guard against the DEFAULT llm_node path specifically - two-
        # tier routing must stay off regardless of the real environment's ENABLE_TWO_TIER_LLM
        # (e.g. a deployment testing that feature), or a non-KB turn would route to a real
        # _fast_llm/_stream_from instead of the mock above, and fail on the missing session
        # activity _stream_from needs.
        monkeypatch.setattr(assistant.settings, "enable_two_tier_llm", False)

    @pytest.mark.asyncio
    async def test_guard_catches_a_hallucinated_price(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "response_guard_enabled", True)
        self._mock_default_llm_node(
            monkeypatch, "The Brezza starts at approximately seven lakh fifty thousand rupees."
        )
        agent_instance = assistant.Assistant()
        ctx = _ctx_after_kb_call("What's the Brezza price?", _BREZZA_PRICE_CONTEXT)

        chunks = [chunk async for chunk in agent_instance.llm_node(ctx, [], MagicMock())]

        assert len(chunks) == 1
        assert "seven lakh fifty thousand" not in chunks[0]
        assert "advisor" in chunks[0].lower() or "confirm" in chunks[0].lower()

    @pytest.mark.asyncio
    async def test_guard_passes_through_a_verified_price_unmodified(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "response_guard_enabled", True)
        reply = "The Brezza starts at approximately eight lakh eleven thousand rupees."
        self._mock_default_llm_node(monkeypatch, reply)
        agent_instance = assistant.Assistant()
        ctx = _ctx_after_kb_call("What's the Brezza price?", _BREZZA_PRICE_CONTEXT)

        chunks = [chunk async for chunk in agent_instance.llm_node(ctx, [], MagicMock())]

        assert chunks == [reply]

    @pytest.mark.asyncio
    async def test_guard_does_not_intervene_when_turn_never_called_the_kb_tool(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "response_guard_enabled", True)
        reply = "Sure, I can help with that. May I know your name please?"
        self._mock_default_llm_node(monkeypatch, reply)
        agent_instance = assistant.Assistant()
        ctx = _ctx_with_user_message("Hi, is this Rahul?")

        chunks = [chunk async for chunk in agent_instance.llm_node(ctx, [], MagicMock())]

        assert chunks == [reply]

    @pytest.mark.asyncio
    async def test_guard_disabled_lets_a_hallucinated_price_through(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "response_guard_enabled", False)
        reply = "The Brezza starts at approximately seven lakh fifty thousand rupees."
        self._mock_default_llm_node(monkeypatch, reply)
        agent_instance = assistant.Assistant()
        ctx = _ctx_after_kb_call("What's the Brezza price?", _BREZZA_PRICE_CONTEXT)

        chunks = [chunk async for chunk in agent_instance.llm_node(ctx, [], MagicMock())]

        assert chunks == [reply]

    @pytest.mark.asyncio
    async def test_guard_logs_a_warning_when_it_fires(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "response_guard_enabled", True)
        self._mock_default_llm_node(
            monkeypatch, "The Brezza starts at approximately seven lakh fifty thousand rupees."
        )
        mock_logger = MagicMock()
        monkeypatch.setattr(assistant, "logger", mock_logger)
        agent_instance = assistant.Assistant()
        ctx = _ctx_after_kb_call("What's the Brezza price?", _BREZZA_PRICE_CONTEXT)

        [chunk async for chunk in agent_instance.llm_node(ctx, [], MagicMock())]

        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_guard_does_not_log_when_reply_is_verified(self, monkeypatch):
        monkeypatch.setattr(assistant.settings, "response_guard_enabled", True)
        self._mock_default_llm_node(
            monkeypatch, "The Brezza starts at approximately eight lakh eleven thousand rupees."
        )
        mock_logger = MagicMock()
        monkeypatch.setattr(assistant, "logger", mock_logger)
        agent_instance = assistant.Assistant()
        ctx = _ctx_after_kb_call("What's the Brezza price?", _BREZZA_PRICE_CONTEXT)

        [chunk async for chunk in agent_instance.llm_node(ctx, [], MagicMock())]

        mock_logger.warning.assert_not_called()


class TestBargeInCancellation:
    """Covers barge-in at the one layer this codebase actually controls: when AgentSession
    cancels the in-flight generation task on interruption (VAD detects the customer talking
    over the agent - see build_session's turn_handling.interruption), the task running
    Assistant.llm_node's generator must unwind promptly and cleanly, not swallow/delay the
    cancellation or leave the upstream LLM stream unclosed. The rest of barge-in (VAD
    detection, in-flight TTS audio truncation, resuming STT) is livekit-agents' own
    audio-pipeline responsibility, not something this custom generator code could break -
    this test targets the one thing that IS ours: cancellation-safety of the wrapping
    generator sitting between the LLM stream and TTS.
    """

    @pytest.mark.asyncio
    async def test_verify_and_stream_propagates_cancellation_and_closes_upstream(self):
        cleanup_ran = False

        async def fake_upstream():
            nonlocal cleanup_ran
            try:
                for i in range(1000):
                    yield f"chunk {i} "
                    await asyncio.sleep(10)  # still "mid-turn" whenever cancellation arrives
            finally:
                cleanup_ran = True

        agent_instance = assistant.Assistant()

        async def consume():
            async for _ in agent_instance._verify_and_stream(fake_upstream(), ["some context"]):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # let it start iterating fake_upstream
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert cleanup_ran is True
