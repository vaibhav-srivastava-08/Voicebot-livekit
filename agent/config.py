"""Centralized configuration for the LiveKit agent, loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Typed application settings sourced from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LiveKit server connection — required, no safe default.
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # Speech/AI provider keys.
    cartesia_api_key: str
    cartesia_voice_id: str

    # --- STT provider selection (see agent/stt_factory.py) ---
    # Which STT provider build_session() builds a client from. Switching providers is purely a
    # .env change - see .env.example. Only the key matching stt_provider needs to actually be
    # set; agent.stt_factory raises a clear error at session-build time if it's missing.
    stt_provider: str = "deepgram"
    deepgram_api_key: str | None = None
    soniox_api_key: str | None = None

    # --- LLM provider selection (see agent/llm_factory.py) ---
    # Which LLM provider build_session()/Assistant's two-tier routing build clients from.
    # Switching providers is purely a .env change - see .env.example for the full list of
    # supported values and working per-provider model-name examples. Model name FORMAT differs
    # per provider (fast_llm_model/smart_llm_model below) - changing llm_provider without also
    # updating the model names to match will fail at call time, not at startup.
    llm_provider: str = "groq"

    # Only the key matching llm_provider needs to actually be set; agent.llm_factory raises a
    # clear error at session-build time if it's missing, rather than failing mid-call. All
    # optional here (rather than one required field per provider) since a given deployment only
    # ever uses one.
    groq_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None
    # Set to use a GCP service account (Vertex AI) instead of a Gemini API key - set
    # GOOGLE_APPLICATION_CREDENTIALS (path to the service account JSON key file) alongside this;
    # google_api_key above is then unused. google_vertex_project is only needed if it can't be
    # inferred from the key file itself.
    google_use_vertexai: bool = False
    google_vertex_project: str | None = None
    google_vertex_location: str = "us-central1"
    # Only meaningful when llm_provider=openai and you're pointing at something other than
    # api.openai.com - any OpenAI-compatible endpoint (OpenRouter, Together, a local
    # vLLM/Ollama server, ...) works by setting this alongside openai_api_key.
    llm_base_url: str | None = None

    # Infrastructure — safe defaults for local development.
    redis_url: str = "redis://localhost:6379/0"
    server_host: str = "0.0.0.0"
    server_port: int = 8080

    agent_log_level: str = "INFO"

    # Whether the LLM/TTS may start generating a reply before the user's turn is confirmed
    # done. Speeds up perceived response time; disable if it destabilizes turn-taking on a
    # given deployment (see agent.assistant.build_session).
    preemptive_generation_enabled: bool = True

    # LLM model tiers - model name FORMAT is provider-specific (see llm_provider above), so
    # these must be updated together with it. Defaults below are Groq model strings (Groq's own
    # recommended replacements for llama-3.1-8b-instant / llama-3.3-70b-versatile, which Groq
    # deprecated 2026-06-17 with a 2026-08-16 shutdown date -
    # https://console.groq.com/docs/deprecations). Working examples for other providers (see
    # .env.example): google -> "gemini-2.5-flash-lite" / "gemini-2.5-flash"; openai ->
    # "gpt-4o-mini" / "gpt-4o".
    fast_llm_model: str = "openai/gpt-oss-20b"
    smart_llm_model: str = "openai/gpt-oss-120b"

    # When True, route each turn to fast_llm_model or smart_llm_model based on whether it looks
    # like it needs a knowledge-base lookup (see agent.assistant._turn_needs_knowledge_base).
    # Off by default so behavior is unchanged unless explicitly enabled.
    enable_two_tier_llm: bool = False

    # When True, verify any price/spec numbers in a reply that followed a search_knowledge_base
    # call against the KB text actually retrieved that turn, substituting a deflection phrase
    # if a number can't be verified (see agent.assistant's response guard). On by default -
    # this is a safety net against hallucinated prices, not an optional quality tweak.
    response_guard_enabled: bool = True

    # Minimum silence (seconds) after the customer stops talking before the turn detector will
    # consider their turn over (see agent.assistant.build_session's turn_handling.endpointing).
    # This is the main "how quickly does Priya jump in" knob:
    #   - Too SHORT: a customer who pauses mid-thought - to find a word, or to switch between
    #     Hindi and English mid-sentence (this bot's expected code-switching pattern) - gets
    #     cut off and interrupted before they meant to yield the turn.
    #   - Too LONG: every response feels laggy, since the agent waits out the full delay even
    #     after the customer has clearly finished, on every single turn.
    # livekit-agents' own default for a streaming turn detector (MultilingualModel, our case)
    # is 0.3s, tuned for fast native-English turn-taking. 0.5s here is a deliberately slightly
    # more patient default: Hindi/English code-switching speech tends to have marginally longer
    # natural pauses (word-finding, language-switch transitions) than monolingual English, and
    # the turn detector model's own end-of-turn probability (not just this timer) is doing most
    # of the actual judgment - this value is a floor/debounce underneath that, not the sole
    # signal. Treat this as a reasoned starting point, not a measured constant - tune it against
    # real call recordings for your customer base.
    endpointing_min_delay_s: float = 0.5

    # --- Worker scaling knobs (see DEPLOY.md's "Scaling knobs" section) ---
    # Defaults below reproduce this project's existing behavior unchanged (see agent.main), so
    # local/dev usage is unaffected; production deployments should override these via env.

    # Pre-warmed idle worker processes (VAD already loaded, see agent.main.prewarm) this
    # process keeps ready so a job never waits on process spin-up + model load. This is a
    # *standing* pool size, not a hard concurrency cap - livekit-agents spins up additional
    # processes on demand for extra concurrent jobs, gated by worker_load_threshold below.
    # livekit-agents' own production default is min(cpu_count, 4); we default to 1 to match
    # this project's existing behavior.
    num_idle_processes: int = 1

    # Fraction (0.0-1.0) of measured CPU load past which this worker reports itself
    # unavailable for new job dispatch, so LiveKit routes the next call to a less-loaded
    # replica instead - this is the real per-replica concurrency gate (see DEPLOY.md).
    # None defers to livekit-agents' own mode-aware default (0.7 under the `start` subcommand,
    # unlimited under `dev`) instead of silently overriding it.
    worker_load_threshold: float | None = None

    # Per-job memory ceiling in MB; that job's process is killed if it's exceeded (0 =
    # disabled). Bounds a single runaway call to itself instead of it starving every other
    # concurrent call on the same worker replica.
    job_memory_limit_mb: float = 0

    # Seconds a worker waits for in-flight calls to finish before exiting on SIGTERM/SIGINT.
    # Keep this in sync with your orchestrator's own graceful-shutdown grace period (e.g.
    # Kubernetes' terminationGracePeriodSeconds) - see DEPLOY.md.
    drain_timeout: int = 3600


settings = AgentSettings()
