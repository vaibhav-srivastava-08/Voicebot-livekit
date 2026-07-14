"""Tests for agent.config: AgentSettings loading and required-field validation.

Every test drives environment state through monkeypatch and runs from a throwaway tmp_path
directory (never the project's own .env), so results only ever reflect the env vars a given
test explicitly sets.
"""

import importlib

import pytest
from pydantic import ValidationError

REQUIRED_ENV = {
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_API_KEY": "test-livekit-key",
    "LIVEKIT_API_SECRET": "test-livekit-secret",
    "DEEPGRAM_API_KEY": "test-deepgram-key",
    "CARTESIA_API_KEY": "test-cartesia-key",
    "GROQ_API_KEY": "test-groq-key",
    "CARTESIA_VOICE_ID": "test-voice-id",
}


def _reload_config_module(monkeypatch, tmp_path, remove=()):
    """Set required env vars, chdir away from any real .env, and (re)load agent.config fresh."""
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    for key in remove:
        monkeypatch.delenv(key, raising=False)

    import agent.config as config_module

    return importlib.reload(config_module)


def test_settings_loads_when_all_required_env_vars_are_present(monkeypatch, tmp_path):
    config_module = _reload_config_module(monkeypatch, tmp_path)
    settings = config_module.settings

    assert settings.livekit_url == REQUIRED_ENV["LIVEKIT_URL"]
    assert settings.livekit_api_key == REQUIRED_ENV["LIVEKIT_API_KEY"]
    assert settings.livekit_api_secret == REQUIRED_ENV["LIVEKIT_API_SECRET"]
    assert settings.deepgram_api_key == REQUIRED_ENV["DEEPGRAM_API_KEY"]
    assert settings.cartesia_api_key == REQUIRED_ENV["CARTESIA_API_KEY"]
    assert settings.groq_api_key == REQUIRED_ENV["GROQ_API_KEY"]
    assert settings.cartesia_voice_id == REQUIRED_ENV["CARTESIA_VOICE_ID"]

    # Infra fields are safe to default and should fall back when not set in the environment.
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.server_host == "0.0.0.0"
    assert settings.server_port == 8080
    assert settings.agent_log_level == "INFO"


@pytest.mark.parametrize("missing_var", sorted(REQUIRED_ENV))
def test_missing_required_env_var_raises_clear_error(monkeypatch, tmp_path, missing_var):
    with pytest.raises(ValidationError) as exc_info:
        _reload_config_module(monkeypatch, tmp_path, remove=[missing_var])

    # The error should clearly identify which field is missing.
    assert missing_var.lower() in str(exc_info.value).lower()
