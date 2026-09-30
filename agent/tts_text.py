"""Text normalization applied to the LLM's output stream just before it reaches Cartesia, so
synthesized speech sounds conversational rather than read-aloud-from-a-document.

SYSTEM_PROMPT already does most of this work at the source (short sentences, no bullets/tables,
digits spelled as words - see prompts.py's BREVITY and HARDCODED RULE TWO sections), and the
framework already strips markdown/emoji by default (see agent.assistant.build_session's
tts_text_transforms). This module is a narrow, defense-in-depth safety net for the handful of
formatting artifacts an LLM can still slip in even when told not to - repeated punctuation
("Great!!", "Really??") and literal newlines/runs of whitespace - both of which read to a
streaming TTS engine as unnatural pauses or emphasis that no one actually spoke.

Deliberately NOT attempted here: rewriting sentence structure, splitting run-on sentences, or
any transform that changes what the text *says* rather than how it's spaced/punctuated - that's
a much riskier, meaning-changing operation better left to prompt engineering (SYSTEM_PROMPT)
than a mechanical post-process.
"""

from collections.abc import AsyncIterable, AsyncIterator

# Characters where a run of 2+ collapses to a single character: repeated "!!!"/"???" read as
# over-emphatic to a TTS engine ("Great!!!" has no prosody advantage over "Great!"), and
# literal whitespace/newlines in a row are LLM formatting artifacts (paragraph breaks, indents)
# that don't correspond to anything a speaker actually paused for.
_COLLAPSIBLE = "!? \t\n\r"


def _collapse_runs(text: str) -> str:
    """Collapse any run of 2+ identical collapsible characters down to one, and collapse a run
    of *mixed* whitespace characters (space/tab/newline) down to a single space."""
    out: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char not in _COLLAPSIBLE:
            out.append(char)
            i += 1
            continue

        run_end = i
        while run_end < len(text) and text[run_end] in _COLLAPSIBLE:
            run_end += 1
        run = text[i:run_end]

        # A run can mix punctuation and whitespace ("! \n\n"). Emit at most one mark (the
        # first one the speaker "led with") followed by at most one space - never collapse a
        # mixed run down to bare punctuation, or the space separating it from the next word
        # disappears and words run together ("Great! Next" -> "Great!Next").
        punctuation_mark = next((c for c in run if c in "!?"), "")
        if punctuation_mark:
            out.append(punctuation_mark)
        if any(c.isspace() for c in run):
            out.append(" ")
        i = run_end
    return "".join(out)


def normalize_for_speech(text: AsyncIterable[str]) -> AsyncIterator[str]:
    """A `tts_text_transforms` entry (see agent.assistant.build_session): collapses repeated
    punctuation and whitespace runs in a streaming-safe way.

    Because chunk boundaries can fall mid-run (e.g. one chunk ending "..." and the next
    starting "!"), a trailing run of collapsible characters is always held back until either a
    non-collapsible character arrives (proving the run is complete) or the stream ends - the
    same holdback technique livekit-agents' own `replace()` text transform uses for a matching
    reason (see livekit.agents.voice.transcription.text_transforms).
    """

    async def _gen() -> AsyncIterator[str]:
        buffer = ""
        async for chunk in text:
            buffer += chunk
            # Find the trailing run of collapsible characters - it might still grow with the
            # next chunk, so hold it back; everything before it is safe to collapse and flush.
            end = len(buffer)
            while end > 0 and buffer[end - 1] in _COLLAPSIBLE:
                end -= 1
            if end > 0:
                yield _collapse_runs(buffer[:end])
                buffer = buffer[end:]

        if buffer:
            collapsed = _collapse_runs(buffer)
            # A purely-trailing whitespace/punctuation tail at the very end of the whole
            # utterance (e.g. the LLM's final newline) carries nothing worth speaking.
            if collapsed.strip():
                yield collapsed

    return _gen()
