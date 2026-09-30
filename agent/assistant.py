"""Agent class definition for the LiveKit voice assistant."""

import random
from dataclasses import dataclass

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentSession,
    FlushSentinel,
    ModelSettings,
    UserInputTranscribedEvent,
    llm,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import cartesia, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.config import settings
from agent.llm_factory import build_llm
from agent.stt_factory import build_stt
from agent.prompts import NUMBER_VERIFICATION_FALLBACK_PHRASES, SYSTEM_PROMPT
from agent.tools import search_knowledge_base
from agent.tts_text import normalize_for_speech
from shared.logger import get_logger
from shared.number_verification import find_unverified_claims

logger = get_logger(__name__)


@dataclass
class VoicebotUserdata:
    """Per-session state shared with tools via RunContext.userdata (see agent.tools)."""

    last_detected_language: str | None = None


# Keyword screen for "this turn probably needs search_knowledge_base" (price/spec/variant/
# colour/comparison questions) vs. small talk, confirmations, or name capture. Not exhaustive
# and not meant to be: it's a cheap pre-filter, not a classifier. See _turn_needs_knowledge_base
# and the tradeoff note on Assistant.__init__ for why "simple and occasionally wrong" is fine.
_KB_KEYWORDS = (
    # pricing / finance
    "price", "cost", "lakh", "rupee", "rs.", "emi", "loan", "finance", "down payment",
    # specs / features customers ask about
    "mileage", "kmpl", "km/l", "spec", "feature", "variant", "colour", "color",
    "airbag", "safety", "safe", "cng", "hybrid", "sunroof", "camera", "engine",
    "seat", "seating", "boot space", "ground clearance", "warranty", "adas",
    # comparisons
    "compare", "comparison", "vs", "versus", "difference", "better than",
    # model names (Arena + Nexa lineup from the system prompt)
    "alto", "s-presso", "spresso", "celerio", "wagon", "swift", "dzire", "eeco", "brezza",
    "ertiga", "victoris", "ignis", "baleno", "fronx", "grand vitara", "xl6", "xl-6", "jimny",
    "invicto", "e-vitara", "e vitara", "ciaz",
)


def _turn_needs_knowledge_base(chat_ctx: llm.ChatContext) -> bool:
    """Cheap keyword screen on the latest user message: does this turn look like it needs a
    KB lookup? False negatives just mean the fast model handles it directly (it can still call
    search_knowledge_base itself); false positives just mean an easy turn gets the slower
    model. Neither is a correctness problem, only a latency/cost one - see Assistant.__init__.
    """
    for message in reversed(chat_ctx.messages()):
        if message.role == "user":
            text = (message.text_content or "").lower()
            return any(keyword in text for keyword in _KB_KEYWORDS)
    return False


def _recent_kb_context(chat_ctx: llm.ChatContext) -> list[str] | None:
    """Return the raw text of any search_knowledge_base tool outputs since the customer's last
    message, or None if this generation doesn't follow one (nothing to verify against)."""
    outputs: list[str] = []
    for item in reversed(chat_ctx.items):
        if isinstance(item, llm.ChatMessage) and item.role == "user":
            break
        if isinstance(item, llm.FunctionCallOutput) and item.name == "search_knowledge_base":
            outputs.append(item.output)
    return outputs or None


def _chunk_text(chunk: llm.ChatChunk | str | FlushSentinel) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
        return chunk.delta.content
    return ""


def _pick_deflection_phrase(language: str | None) -> str:
    phrases = (
        NUMBER_VERIFICATION_FALLBACK_PHRASES["hi"]
        if (language or "").lower().startswith("hi")
        else NUMBER_VERIFICATION_FALLBACK_PHRASES["en"]
    )
    return random.choice(phrases)


class Assistant(Agent):
    """LiveKit Agent subclass configured with Priya's persona, KB lookup, and call-ending tool."""

    def __init__(self, instructions: str = SYSTEM_PROMPT) -> None:
        # instructions is per-call: agent.main builds it from SYSTEM_PROMPT + that call's
        # dynamic variables (objective, lead_name, ...) via prompts.build_system_prompt().
        # delete_room=True disconnects every participant on hangup, including SIP/Exotel callers.
        super().__init__(
            instructions=instructions,
            tools=[EndCallTool(delete_room=True), search_knowledge_base],
        )

        # Two-tier LLM routing (see llm_node below). Only construct the fast-tier client when
        # the feature is enabled - its presence/absence also doubles as the on/off flag.
        #
        # Tradeoff: most turns (greetings, "yes"/"no", giving a name, general chit-chat) don't
        # need careful reasoning, so routing them to a small/fast model cuts latency and cost
        # with no real quality loss. Turns that look like they need search_knowledge_base
        # (price/spec/variant/comparison questions) go to the larger/smarter model instead,
        # since converting the KB's raw numbers into the prompt's exact spoken word-format
        # (Hardcoded Rule Two) and deciding to call the tool correctly both benefit from a
        # stronger model. The keyword screen below is deliberately simple, not a classifier -
        # it will occasionally misroute (miss a KB-worthy question, or send an easy one to the
        # smart model). That's an acceptable latency/cost tradeoff, not a correctness one: the
        # fast model can still call the tool itself if it decides it needs to.
        self._fast_llm = (
            build_llm(settings.fast_llm_model)
            if settings.enable_two_tier_llm
            else None
        )

    def _select_fast_llm(self, chat_ctx: llm.ChatContext) -> llm.LLM | None:
        """Return the fast-tier LLM for this turn, or None to use the session's default
        (smart-tier) LLM. Only meaningful when two-tier routing is enabled."""
        if self._fast_llm is None or _turn_needs_knowledge_base(chat_ctx):
            return None
        return self._fast_llm

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        """Overrides Agent's default LLM node for two-tier model routing and the response
        guard. Falls straight through to the framework's default (which uses the session's
        configured LLM) when two-tier routing is disabled or this turn is routed to the smart
        tier; wraps the result in the response guard when this turn follows a
        search_knowledge_base call and the guard is enabled."""
        fast_llm = self._select_fast_llm(chat_ctx)
        if fast_llm is None:
            upstream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        else:
            upstream = self._stream_from(fast_llm, chat_ctx, tools, model_settings)

        if not settings.response_guard_enabled:
            return upstream

        kb_context = _recent_kb_context(chat_ctx)
        if kb_context is None:
            return upstream
        return self._verify_and_stream(upstream, kb_context)

    async def _stream_from(
        self,
        llm_instance: llm.LLM,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ):
        """Mirrors Agent.default.llm_node's implementation, but against an explicitly chosen
        LLM instance instead of the session's configured one - needed for per-turn routing."""
        tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
        conn_options = self._get_activity_or_raise().session.conn_options.llm_conn_options
        async with llm_instance.chat(
            chat_ctx=chat_ctx, tools=tools, tool_choice=tool_choice, conn_options=conn_options
        ) as stream:
            async for chunk in stream:
                yield chunk

    async def _verify_and_stream(self, upstream, kb_context: list[str]):
        """Response guard: buffer `upstream`'s full text output, verify any monetary/spec
        numbers against `kb_context` (the KB text actually retrieved this turn), and either
        replay it unmodified or substitute a deflection phrase if a number can't be verified.

        Buffering sacrifices token-by-token streaming for this turn only - acceptable since it
        only applies to turns that just called search_knowledge_base (already a slower,
        tool-augmented turn), not every turn.
        """
        buffered = []
        text_parts = []
        async for chunk in upstream:
            buffered.append(chunk)
            text_parts.append(_chunk_text(chunk))

        full_text = "".join(text_parts)
        unverified = find_unverified_claims(full_text, kb_context)

        if unverified:
            try:
                language = getattr(
                    self._get_activity_or_raise().session.userdata, "last_detected_language", None
                )
            except Exception:
                # Language lookup is a nice-to-have for picking en/hi phrasing, never a reason
                # to fail the whole guard - fall back to the English deflection phrase.
                language = None
            logger.warning(
                "response guard fired: unverified number(s) %s in reply %r "
                "(retrieved context: %d chunk(s)) - substituting deflection phrase",
                unverified,
                full_text,
                len(kb_context),
            )
            yield _pick_deflection_phrase(language)
            return

        for chunk in buffered:
            yield chunk


def build_session(vad: silero.VAD | None = None) -> AgentSession:
    """Build an AgentSession wired with STT, LLM, TTS, VAD, and turn detection. Caller starts it.

    `vad` should be a prewarmed instance (see agent.main.prewarm) so the blocking ONNX model
    load happens once per worker process instead of once per call. Falls back to loading fresh
    if omitted, for standalone callers (e.g. tests) that don't go through job prewarming.
    """
    session = AgentSession(
        # Provider comes from settings.stt_provider - see agent/stt_factory.py. Both supported
        # providers are configured there for Hindi/English code-switching (this bot's expected
        # speech pattern) rather than single-language transcription.
        stt=build_stt(),
        # Session default LLM - the "smart" tier. Used for every turn when two-tier routing is
        # disabled, and for turns routed to the smart tier when it's enabled (see
        # Assistant.llm_node). Provider comes from settings.llm_provider - see agent/llm_factory.py.
        llm=build_llm(settings.smart_llm_model),
        tts=cartesia.TTS(voice=settings.cartesia_voice_id),
        vad=vad or silero.VAD.load(),
        userdata=VoicebotUserdata(),
        # filter_markdown/filter_emoji are livekit-agents' own defaults - repeated explicitly
        # here (rather than left implicit) because specifying tts_text_transforms at all
        # REPLACES the default list rather than appending to it; normalize_for_speech is this
        # project's own addition (see agent/tts_text.py) that collapses repeated punctuation
        # ("Great!!") and stray whitespace/newlines before the text reaches Cartesia, so
        # prosody reflects what SYSTEM_PROMPT's brevity rules already ask for: short,
        # conversational, lightly-punctuated sentences, not read-aloud document formatting.
        tts_text_transforms=["filter_markdown", "filter_emoji", normalize_for_speech],
        turn_handling={
            "turn_detection": MultilingualModel(),
            # "adaptive" mode routes audio to LiveKit's hosted inference gateway for ML-based
            # interruption detection (see livekit.agents.inference.AdaptiveInterruptionDetector)
            # - unreachable from this self-hosted deployment. Forced to "vad" explicitly rather
            # than left as the framework default so behavior doesn't depend on which CLI mode
            # (`start` vs `dev`) happens to run it - livekit-agents already auto-disables
            # adaptive interruption under `start` in a non-Cloud-hosted deployment, but not
            # under `dev`.
            "interruption": {"mode": "vad"},
            # Minimum silence before the turn detector considers the customer done talking -
            # see agent.config.settings.endpointing_min_delay_s for the full tradeoff writeup
            # (too short interrupts code-switching pauses, too long feels laggy).
            "endpointing": {"min_delay": settings.endpointing_min_delay_s},
            # Preemptive generation speculatively starts the LLM/TTS before the user's turn is
            # confirmed done. Toggle via PREEMPTIVE_GENERATION_ENABLED if it ever destabilizes
            # turn-taking on a given deployment.
            "preemptive_generation": {"enabled": settings.preemptive_generation_enabled},
        },
    )

    def _on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        # Tracks the STT-detected language of the customer's last utterance so tools (e.g.
        # search_knowledge_base's filler phrase) can match it - real signal from the STT
        # provider's code-switching mode (see agent/stt_factory.py), not a guess.
        if ev.is_final and ev.language:
            session.userdata.last_detected_language = ev.language

    session.on("user_input_transcribed", _on_user_input_transcribed)

    return session
