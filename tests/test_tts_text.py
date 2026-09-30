"""Tests for agent.tts_text.normalize_for_speech: the streaming-safe punctuation/whitespace
normalization applied just before text reaches Cartesia (see agent.assistant.build_session).
"""

import pytest

from agent.tts_text import normalize_for_speech


async def _achunks(*chunks: str):
    for chunk in chunks:
        yield chunk


async def _collect(text_iterable) -> str:
    return "".join([chunk async for chunk in normalize_for_speech(text_iterable)])


class TestNormalizeForSpeech:
    @pytest.mark.asyncio
    async def test_plain_conversational_text_passes_through_unchanged(self):
        text = "Brezza is a great choice. Would you like to know the price?"
        assert await _collect(_achunks(text)) == text

    @pytest.mark.asyncio
    async def test_collapses_repeated_exclamation_marks(self):
        assert await _collect(_achunks("Great choice!!!")) == "Great choice!"

    @pytest.mark.asyncio
    async def test_collapses_repeated_question_marks(self):
        assert await _collect(_achunks("Really???")) == "Really?"

    @pytest.mark.asyncio
    async def test_collapses_mixed_punctuation_run_to_first_mark(self):
        assert await _collect(_achunks("Wait, seriously?!")) == "Wait, seriously?"

    @pytest.mark.asyncio
    async def test_collapses_whitespace_run_to_single_space(self):
        assert await _collect(_achunks("Sure,    let me check.")) == "Sure, let me check."

    @pytest.mark.asyncio
    async def test_collapses_newlines_to_a_single_space(self):
        assert await _collect(_achunks("Sure.\n\nLet me check that.")) == "Sure. Let me check that."

    @pytest.mark.asyncio
    async def test_mixed_punctuation_and_whitespace_keeps_word_separation(self):
        # The bug this guards against: collapsing "! \n" down to bare "!" would merge the
        # following word onto the exclamation mark with no space ("Great!Next").
        assert await _collect(_achunks("Great! \nNext up is the price.")) == (
            "Great! Next up is the price."
        )

    @pytest.mark.asyncio
    async def test_trailing_whitespace_at_end_of_stream_is_dropped(self):
        assert await _collect(_achunks("All set.\n")) == "All set."

    @pytest.mark.asyncio
    async def test_purely_whitespace_stream_yields_nothing(self):
        assert await _collect(_achunks("   \n\n  ")) == ""

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        assert await _collect(_achunks()) == ""

    @pytest.mark.asyncio
    async def test_run_split_across_chunk_boundary_still_collapses(self):
        # "Great!!" arrives as two separate chunks, split mid-run.
        assert await _collect(_achunks("Great!", "!! Next.")) == "Great! Next."

    @pytest.mark.asyncio
    async def test_whitespace_run_split_across_chunk_boundary_still_collapses(self):
        assert await _collect(_achunks("Sure,  ", "   let me check.")) == "Sure, let me check."

    @pytest.mark.asyncio
    async def test_single_punctuation_mark_is_left_alone(self):
        text = "Is that alright with you?"
        assert await _collect(_achunks(text)) == text
