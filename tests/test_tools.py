"""Tests for agent.tools: search_knowledge_base's language-aware filler-phrase wiring.

Uses a minimal fake RunContext (not a MagicMock) so with_filler's async-context-manager
protocol behaves correctly and the callable `source` argument it's invoked with can be
inspected directly - no live AgentSession, pipeline, or LiveKit connection needed.
"""

import os
from contextlib import asynccontextmanager

os.environ.setdefault("LIVEKIT_URL", "wss://example.livekit.cloud")
os.environ.setdefault("LIVEKIT_API_KEY", "test-livekit-key")
os.environ.setdefault("LIVEKIT_API_SECRET", "test-livekit-secret")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram-key")
os.environ.setdefault("CARTESIA_API_KEY", "test-cartesia-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("CARTESIA_VOICE_ID", "test-voice-id")

import pytest

from agent import tools
from agent.prompts import TOOL_FILLER_PHRASES


class _FakeUserdata:
    def __init__(self, last_detected_language=None):
        self.last_detected_language = last_detected_language


class _FakeRunContext:
    """Minimal stand-in for RunContext: just enough for with_filler's async-with protocol."""

    def __init__(self, last_detected_language=None):
        self.userdata = _FakeUserdata(last_detected_language)
        self.with_filler_sources = []

    @asynccontextmanager
    async def with_filler(self, source, **kwargs):
        self.with_filler_sources.append(source)
        yield


class TestPickFillerPhrase:
    def test_defaults_to_english_when_no_language_detected(self):
        assert tools._pick_filler_phrase(None) in TOOL_FILLER_PHRASES["en"]

    def test_defaults_to_english_for_non_hindi_language(self):
        assert tools._pick_filler_phrase("en") in TOOL_FILLER_PHRASES["en"]

    def test_picks_hindi_for_hindi_language_code(self):
        assert tools._pick_filler_phrase("hi") in TOOL_FILLER_PHRASES["hi"]

    def test_language_match_is_case_insensitive(self):
        assert tools._pick_filler_phrase("HI") in TOOL_FILLER_PHRASES["hi"]


class TestSearchKnowledgeBase:
    @pytest.mark.asyncio
    async def test_wraps_the_search_in_exactly_one_filler_context(self):
        ctx = _FakeRunContext(last_detected_language="en")

        result = await tools.search_knowledge_base(ctx, "Brezza price")

        assert isinstance(result, str)
        assert len(ctx.with_filler_sources) == 1

    @pytest.mark.asyncio
    async def test_filler_source_matches_detected_language(self):
        ctx = _FakeRunContext(last_detected_language="hi")

        await tools.search_knowledge_base(ctx, "Brezza price")

        source = ctx.with_filler_sources[0]
        assert source(0) in TOOL_FILLER_PHRASES["hi"]

    @pytest.mark.asyncio
    async def test_filler_source_defaults_to_english_when_language_unknown(self):
        ctx = _FakeRunContext(last_detected_language=None)

        await tools.search_knowledge_base(ctx, "Brezza price")

        source = ctx.with_filler_sources[0]
        assert source(0) in TOOL_FILLER_PHRASES["en"]

    @pytest.mark.asyncio
    async def test_returns_real_knowledge_base_search_result(self):
        ctx = _FakeRunContext()

        result = await tools.search_knowledge_base(ctx, "Brezza price")

        assert "brezza" in result.lower()
