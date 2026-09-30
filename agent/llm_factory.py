"""Builds the configured LLM client for a given model name, based on
agent.config.settings.llm_provider - so switching LLM providers (Groq, Google Gemini, or any
OpenAI-compatible endpoint: OpenRouter, Together, a local vLLM/Ollama server, ...) is purely a
.env change, never a code change. See .env.example for the full list of settings and working
per-provider model-name examples.

Both call sites that need an LLM (see agent.assistant: build_session()'s main LLM, and
Assistant's two-tier fast-tier LLM when ENABLE_TWO_TIER_LLM is on) go through build_llm() so
there is exactly one place that knows how to turn "a provider name + a model string" into a
livekit.agents.llm.LLM instance. Both tiers always use the SAME provider - only the model name
differs per tier - since there's no clear use case here for mixing providers within one call.
"""

from livekit.agents import NOT_GIVEN, llm
from livekit.plugins import google, groq, openai

from agent.config import settings

SUPPORTED_PROVIDERS = ("groq", "google", "openai")


def build_llm(model: str) -> llm.LLM:
    """Construct an llm.LLM for `model` using the configured provider
    (settings.llm_provider).

    Raises a clear, actionable RuntimeError/ValueError at session-build time if the provider is
    unrecognized or its required API key isn't set, rather than failing confusingly mid-call.
    """
    provider = settings.llm_provider

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set - get one at "
                "https://console.groq.com/keys"
            )
        return groq.LLM(model=model, api_key=settings.groq_api_key)

    if provider == "google":
        if settings.google_use_vertexai:
            return google.LLM(
                model=model,
                vertexai=True,
                project=settings.google_vertex_project or NOT_GIVEN,
                location=settings.google_vertex_location,
            )
        if not settings.google_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=google but neither GOOGLE_API_KEY nor GOOGLE_USE_VERTEXAI is "
                "set - get an API key at https://aistudio.google.com/apikey, or set "
                "GOOGLE_USE_VERTEXAI=true with GOOGLE_APPLICATION_CREDENTIALS for a service "
                "account"
            )
        return google.LLM(model=model, api_key=settings.google_api_key)

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. This provider also covers "
                "any OpenAI-compatible endpoint (OpenRouter, Together, a local vLLM/Ollama "
                "server, ...) via LLM_BASE_URL - set that too if you're pointing somewhere "
                "other than api.openai.com."
            )
        return openai.LLM(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url or NOT_GIVEN,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r} - supported values: "
        f"{', '.join(SUPPORTED_PROVIDERS)}"
    )
