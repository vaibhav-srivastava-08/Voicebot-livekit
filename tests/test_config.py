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
    "CARTESIA_API_KEY": "test-cartesia-key",
    "CARTESIA_VOICE_ID": "test-voice-id",
}
# NOT in REQUIRED_ENV: GROQ_API_KEY/GOOGLE_API_KEY/OPENAI_API_KEY (LLM) and
# DEEPGRAM_API_KEY/SONIOX_API_KEY (STT) are all optional at the settings level - only the key
# matching LLM_PROVIDER/STT_PROVIDER needs to be set, and agent.llm_factory/agent.stt_factory
# (not AgentSettings) are what raise a clear error if it's missing - see test_llm_factory.py.


@pytest.fixture(autouse=True)
def _clear_llm_provider_env_leaks(monkeypatch):
    """Other test modules (test_agent.py, test_tools.py) seed GROQ_API_KEY etc. via
    os.environ.setdefault(...) at import time - a real, non-monkeypatched mutation - which can
    leak into these tests when run in the same pytest process. test_bridge.py is worse: it
    imports bridge.server, which calls load_dotenv() unconditionally, loading this project's own
    real .env (LLM_PROVIDER, STT_PROVIDER, and whatever provider keys a given developer has set
    locally) into the real process environment for the rest of the pytest run. Clear all of it
    before every test here (before the test body's own monkeypatch.setenv calls, since autouse
    fixtures run first) so these tests only ever see what they explicitly set, regardless of
    import/collection order or what's in the developer's local .env."""
    for key in (
        "LLM_PROVIDER",
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "LLM_BASE_URL",
        "STT_PROVIDER",
        "DEEPGRAM_API_KEY",
        "SONIOX_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


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
    assert settings.cartesia_api_key == REQUIRED_ENV["CARTESIA_API_KEY"]
    assert settings.cartesia_voice_id == REQUIRED_ENV["CARTESIA_VOICE_ID"]

    # LLM provider defaults to groq with no key set - agent.llm_factory.build_llm is what
    # raises a clear error at session-build time if the matching key is missing, not this
    # settings object (see test_llm_factory.py).
    assert settings.llm_provider == "groq"
    assert settings.groq_api_key is None
    assert settings.google_api_key is None
    assert settings.openai_api_key is None
    assert settings.llm_base_url is None

    # STT provider defaults to deepgram with no key set - agent.stt_factory.build_stt is what
    # raises a clear error at session-build time if the matching key is missing, not this
    # settings object (see test_stt_factory.py).
    assert settings.stt_provider == "deepgram"
    assert settings.deepgram_api_key is None
    assert settings.soniox_api_key is None

    # Infra fields are safe to default and should fall back when not set in the environment.
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.server_host == "0.0.0.0"
    assert settings.server_port == 8080
    assert settings.agent_log_level == "INFO"
    assert settings.preemptive_generation_enabled is True
    assert settings.endpointing_min_delay_s == 0.5

    # Worker scaling knobs default to reproducing this project's previous hardcoded behavior
    # (see agent.main) unchanged, so local/dev usage is unaffected until explicitly overridden.
    assert settings.num_idle_processes == 1
    assert settings.worker_load_threshold is None
    assert settings.job_memory_limit_mb == 0
    assert settings.drain_timeout == 3600


def test_preemptive_generation_enabled_can_be_toggled_off(monkeypatch, tmp_path):
    monkeypatch.setenv("PREEMPTIVE_GENERATION_ENABLED", "false")
    config_module = _reload_config_module(monkeypatch, tmp_path)
    assert config_module.settings.preemptive_generation_enabled is False


def test_llm_provider_and_keys_can_be_switched_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    config_module = _reload_config_module(monkeypatch, tmp_path)

    assert config_module.settings.llm_provider == "google"
    assert config_module.settings.google_api_key == "test-google-key"
    # Switching providers doesn't require the other providers' keys to be set.
    assert config_module.settings.groq_api_key is None


def test_stt_provider_and_keys_can_be_switched_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("STT_PROVIDER", "soniox")
    monkeypatch.setenv("SONIOX_API_KEY", "test-soniox-key")
    config_module = _reload_config_module(monkeypatch, tmp_path)

    assert config_module.settings.stt_provider == "soniox"
    assert config_module.settings.soniox_api_key == "test-soniox-key"
    # Switching providers doesn't require the other provider's key to be set.
    assert config_module.settings.deepgram_api_key is None


def test_llm_base_url_can_be_set_for_openai_compatible_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    config_module = _reload_config_module(monkeypatch, tmp_path)

    assert config_module.settings.llm_base_url == "https://openrouter.ai/api/v1"


def test_endpointing_min_delay_can_be_overridden_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ENDPOINTING_MIN_DELAY_S", "0.8")
    config_module = _reload_config_module(monkeypatch, tmp_path)
    assert config_module.settings.endpointing_min_delay_s == 0.8


def test_worker_scaling_knobs_can_be_overridden_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NUM_IDLE_PROCESSES", "4")
    monkeypatch.setenv("WORKER_LOAD_THRESHOLD", "0.75")
    monkeypatch.setenv("JOB_MEMORY_LIMIT_MB", "1500")
    monkeypatch.setenv("DRAIN_TIMEOUT", "120")
    config_module = _reload_config_module(monkeypatch, tmp_path)

    assert config_module.settings.num_idle_processes == 4
    assert config_module.settings.worker_load_threshold == 0.75
    assert config_module.settings.job_memory_limit_mb == 1500
    assert config_module.settings.drain_timeout == 120


@pytest.mark.parametrize("missing_var", sorted(REQUIRED_ENV))
def test_missing_required_env_var_raises_clear_error(monkeypatch, tmp_path, missing_var):
    with pytest.raises(ValidationError) as exc_info:
        _reload_config_module(monkeypatch, tmp_path, remove=[missing_var])

    # The error should clearly identify which field is missing.
    assert missing_var.lower() in str(exc_info.value).lower()
