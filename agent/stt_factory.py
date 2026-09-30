"""Builds the configured STT client, based on agent.config.settings.stt_provider - so switching
STT providers (Deepgram or Soniox) is purely a .env change, never a code change. See
.env.example for the full list of settings.

build_session() (agent.assistant) goes through build_stt() so there is exactly one place that
knows how to turn a provider name into a livekit.agents.stt.STT instance.
"""

from livekit.agents import stt
from livekit.plugins import deepgram, soniox

from agent.config import settings

SUPPORTED_PROVIDERS = ("deepgram", "soniox")


def build_stt() -> stt.STT:
    """Construct an stt.STT using the configured provider (settings.stt_provider).

    Raises a clear, actionable RuntimeError/ValueError at session-build time if the provider is
    unrecognized or its required API key isn't set, rather than failing confusingly mid-call.
    """
    provider = settings.stt_provider

    if provider == "deepgram":
        if not settings.deepgram_api_key:
            raise RuntimeError("STT_PROVIDER=deepgram but DEEPGRAM_API_KEY is not set")
        # language="multi" enables Deepgram's code-switching mode (Hindi/English mixed speech).
        # Without it, STT defaults to en-US only and mishears/drops non-English speech entirely.
        return deepgram.STT(
            model="nova-3", language="multi", api_key=settings.deepgram_api_key
        )

    if provider == "soniox":
        if not settings.soniox_api_key:
            raise RuntimeError(
                "STT_PROVIDER=soniox but SONIOX_API_KEY is not set - get one at "
                "https://console.soniox.com"
            )
        # language_hints (rather than a hard language pin) plus language identification is
        # Soniox's equivalent of Deepgram's language="multi": it lets Hindi/English
        # code-switching speech be transcribed without locking to a single language.
        return soniox.STT(
            api_key=settings.soniox_api_key,
            params=soniox.STTOptions(
                language_hints=["en", "hi"],
                enable_language_identification=True,
            ),
        )

    raise ValueError(
        f"Unknown STT_PROVIDER={provider!r} - supported values: {', '.join(SUPPORTED_PROVIDERS)}"
    )
