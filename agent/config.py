"""Centralized configuration for the LiveKit agent, loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """Typed application settings sourced from environment variables and .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LiveKit server connection — required, no safe default.
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    # Speech/AI provider keys — required, no safe default.
    deepgram_api_key: str
    cartesia_api_key: str
    groq_api_key: str
    cartesia_voice_id: str

    # Infrastructure — safe defaults for local development.
    redis_url: str = "redis://localhost:6379/0"
    server_host: str = "0.0.0.0"
    server_port: int = 8080

    agent_log_level: str = "INFO"


settings = AgentSettings()
