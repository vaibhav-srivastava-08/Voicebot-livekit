"""Tests for agent.llm_factory.build_llm: provider selection and validation.

Provider plugin classes (groq.LLM, google.LLM, openai.LLM) are mocked - no real API calls or
model downloads happen when these tests run.
"""

import os
from unittest.mock import MagicMock

# agent.config builds its settings singleton at import time. Seed harmless placeholder values
# so import never depends on a real .env or real secrets in the environment these tests run in.
os.environ.setdefault("LIVEKIT_URL", "wss://example.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test-livekit-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-livekit-secret")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram-key")
os.environ.setdefault("CARTESIA_API_KEY", "test-cartesia-key")
os.environ.setdefault("CARTESIA_VOICE_ID", "test-voice-id")

import pytest

from agent import llm_factory


class TestBuildLlmGroq:
    def test_builds_groq_client_with_configured_key(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "groq")
        monkeypatch.setattr(llm_factory.settings, "groq_api_key", "gsk_test")
        monkeypatch.setattr(
            llm_factory.groq, "LLM", MagicMock(return_value=MagicMock(name="groq-llm"))
        )

        result = llm_factory.build_llm("openai/gpt-oss-120b")

        llm_factory.groq.LLM.assert_called_once_with(
            model="openai/gpt-oss-120b", api_key="gsk_test"
        )
        assert result is llm_factory.groq.LLM.return_value

    def test_raises_clear_error_when_groq_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "groq")
        monkeypatch.setattr(llm_factory.settings, "groq_api_key", None)

        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            llm_factory.build_llm("openai/gpt-oss-120b")


class TestBuildLlmGoogle:
    def test_builds_google_client_with_configured_key(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "google")
        monkeypatch.setattr(llm_factory.settings, "google_use_vertexai", False)
        monkeypatch.setattr(llm_factory.settings, "google_api_key", "gemini_test")
        monkeypatch.setattr(
            llm_factory.google, "LLM", MagicMock(return_value=MagicMock(name="google-llm"))
        )

        result = llm_factory.build_llm("gemini-2.5-flash")

        llm_factory.google.LLM.assert_called_once_with(
            model="gemini-2.5-flash", api_key="gemini_test"
        )
        assert result is llm_factory.google.LLM.return_value

    def test_raises_clear_error_when_google_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "google")
        monkeypatch.setattr(llm_factory.settings, "google_use_vertexai", False)
        monkeypatch.setattr(llm_factory.settings, "google_api_key", None)

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            llm_factory.build_llm("gemini-2.5-flash")

    def test_builds_google_client_with_vertexai_service_account(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "google")
        monkeypatch.setattr(llm_factory.settings, "google_use_vertexai", True)
        monkeypatch.setattr(llm_factory.settings, "google_vertex_project", "swiftex-development")
        monkeypatch.setattr(llm_factory.settings, "google_vertex_location", "us-central1")
        monkeypatch.setattr(
            llm_factory.google, "LLM", MagicMock(return_value=MagicMock(name="google-llm"))
        )

        result = llm_factory.build_llm("gemini-2.5-flash")

        llm_factory.google.LLM.assert_called_once_with(
            model="gemini-2.5-flash",
            vertexai=True,
            project="swiftex-development",
            location="us-central1",
        )
        assert result is llm_factory.google.LLM.return_value


class TestBuildLlmOpenAI:
    def test_builds_openai_client_with_configured_key(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "openai")
        monkeypatch.setattr(llm_factory.settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(llm_factory.settings, "llm_base_url", None)
        monkeypatch.setattr(
            llm_factory.openai, "LLM", MagicMock(return_value=MagicMock(name="openai-llm"))
        )

        result = llm_factory.build_llm("gpt-4o-mini")

        _, kwargs = llm_factory.openai.LLM.call_args
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["api_key"] == "sk-test"
        assert result is llm_factory.openai.LLM.return_value

    def test_passes_custom_base_url_when_set(self, monkeypatch):
        # Covers "any OpenAI-compatible endpoint" (OpenRouter, Together, a local vLLM/Ollama
        # server, ...) - the whole point of not requiring a dedicated plugin per provider.
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "openai")
        monkeypatch.setattr(llm_factory.settings, "openai_api_key", "sk-test")
        monkeypatch.setattr(llm_factory.settings, "llm_base_url", "https://openrouter.ai/api/v1")
        monkeypatch.setattr(
            llm_factory.openai, "LLM", MagicMock(return_value=MagicMock(name="openai-llm"))
        )

        llm_factory.build_llm("some/model")

        _, kwargs = llm_factory.openai.LLM.call_args
        assert kwargs["base_url"] == "https://openrouter.ai/api/v1"

    def test_raises_clear_error_when_openai_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "openai")
        monkeypatch.setattr(llm_factory.settings, "openai_api_key", None)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm_factory.build_llm("gpt-4o-mini")


class TestBuildLlmUnknownProvider:
    def test_raises_value_error_for_unrecognized_provider(self, monkeypatch):
        monkeypatch.setattr(llm_factory.settings, "llm_provider", "anthropic")

        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            llm_factory.build_llm("claude-3")
