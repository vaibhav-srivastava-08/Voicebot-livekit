"""Tests for agent.stt_factory.build_stt: provider selection and validation.

Provider plugin classes (deepgram.STT, soniox.STT) are mocked - no real API calls or model
downloads happen when these tests run.
"""

import os
from unittest.mock import MagicMock

# agent.config builds its settings singleton at import time. Seed harmless placeholder values
# so import never depends on a real .env or real secrets in the environment these tests run in.
os.environ.setdefault("LIVEKIT_URL", "wss://example.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test-livekit-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-livekit-secret")
os.environ.setdefault("CARTESIA_API_KEY", "test-cartesia-key")
os.environ.setdefault("CARTESIA_VOICE_ID", "test-voice-id")

import pytest

from agent import stt_factory


class TestBuildSttDeepgram:
    def test_builds_deepgram_client_with_configured_key(self, monkeypatch):
        monkeypatch.setattr(stt_factory.settings, "stt_provider", "deepgram")
        monkeypatch.setattr(stt_factory.settings, "deepgram_api_key", "dg_test")
        monkeypatch.setattr(
            stt_factory.deepgram, "STT", MagicMock(return_value=MagicMock(name="deepgram-stt"))
        )

        result = stt_factory.build_stt()

        stt_factory.deepgram.STT.assert_called_once_with(
            model="nova-3", language="multi", api_key="dg_test"
        )
        assert result is stt_factory.deepgram.STT.return_value

    def test_raises_clear_error_when_deepgram_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(stt_factory.settings, "stt_provider", "deepgram")
        monkeypatch.setattr(stt_factory.settings, "deepgram_api_key", None)

        with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
            stt_factory.build_stt()


class TestBuildSttSoniox:
    def test_builds_soniox_client_with_configured_key(self, monkeypatch):
        monkeypatch.setattr(stt_factory.settings, "stt_provider", "soniox")
        monkeypatch.setattr(stt_factory.settings, "soniox_api_key", "snx_test")
        monkeypatch.setattr(
            stt_factory.soniox, "STT", MagicMock(return_value=MagicMock(name="soniox-stt"))
        )

        result = stt_factory.build_stt()

        assert stt_factory.soniox.STT.call_args.kwargs["api_key"] == "snx_test"
        params = stt_factory.soniox.STT.call_args.kwargs["params"]
        assert params.language_hints == ["en", "hi"]
        assert params.enable_language_identification is True
        assert result is stt_factory.soniox.STT.return_value

    def test_raises_clear_error_when_soniox_api_key_missing(self, monkeypatch):
        monkeypatch.setattr(stt_factory.settings, "stt_provider", "soniox")
        monkeypatch.setattr(stt_factory.settings, "soniox_api_key", None)

        with pytest.raises(RuntimeError, match="SONIOX_API_KEY"):
            stt_factory.build_stt()


def test_raises_value_error_for_unknown_provider(monkeypatch):
    monkeypatch.setattr(stt_factory.settings, "stt_provider", "not-a-real-provider")

    with pytest.raises(ValueError, match="STT_PROVIDER"):
        stt_factory.build_stt()
