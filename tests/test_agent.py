"""Tests for agent.assistant: the Assistant class and build_session() wiring.

All external speech/LLM clients (Deepgram, Cartesia, Groq, Silero, the turn detector model)
are mocked - no real API calls or model downloads happen when these tests run.
"""

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
from livekit.agents import AgentSession
from livekit.agents.llm import is_function_tool

from agent import assistant
from agent.prompts import HUMAN_TRANSFER_PHRASE


@pytest.fixture
def mocked_plugins(monkeypatch):
    """Replace every speech/LLM plugin build_session() touches with a mock, so no real
    API calls or model downloads (e.g. silero.VAD.load()) ever happen in tests."""
    monkeypatch.setattr(assistant.deepgram, "STT", MagicMock(return_value=MagicMock(name="stt")))
    monkeypatch.setattr(assistant.cartesia, "TTS", MagicMock(return_value=MagicMock(name="tts")))
    monkeypatch.setattr(assistant.groq, "LLM", MagicMock(return_value=MagicMock(name="llm")))
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

    assistant.deepgram.STT.assert_called_once_with(model="nova-3")
    assistant.groq.LLM.assert_called_once_with(
        model="llama-3.3-70b-versatile", api_key=assistant.settings.groq_api_key
    )
    assistant.cartesia.TTS.assert_called_once()
    assistant.silero.VAD.load.assert_called_once()
    assistant.MultilingualModel.assert_called_once()


def test_assistant_instantiates_without_error():
    agent_instance = assistant.Assistant()
    assert isinstance(agent_instance, assistant.Assistant)


def test_get_customer_info_is_a_function_tool():
    assert is_function_tool(assistant.Assistant.get_customer_info)


def test_transfer_to_human_is_a_function_tool():
    assert is_function_tool(assistant.Assistant.transfer_to_human)


def test_assistant_instance_exposes_both_tools():
    agent_instance = assistant.Assistant()
    tool_names = {tool.info.name for tool in agent_instance.tools if is_function_tool(tool)}
    assert {"get_customer_info", "transfer_to_human"} <= tool_names


@pytest.mark.asyncio
async def test_get_customer_info_returns_mock_customer():
    agent_instance = assistant.Assistant()
    result = await agent_instance.get_customer_info(None, "555-123-4567")
    assert result == {"name": "Test Customer", "account_status": "active"}


@pytest.mark.asyncio
async def test_transfer_to_human_returns_transfer_phrase():
    agent_instance = assistant.Assistant()
    result = await agent_instance.transfer_to_human(None)
    assert result == HUMAN_TRANSFER_PHRASE
