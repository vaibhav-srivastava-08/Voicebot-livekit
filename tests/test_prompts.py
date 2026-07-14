"""Tests for agent.prompts: constant presence, shape, and voice-safe (non-markdown) content."""

from agent import prompts

_MARKDOWN_MARKERS = ("**", "##", "-", "*")

_ALL_PROMPT_STRINGS = [
    prompts.SYSTEM_PROMPT,
    prompts.GREETING_INSTRUCTION,
    prompts.FALLBACK_INSTRUCTION,
    prompts.HUMAN_TRANSFER_PHRASE,
    *prompts.TOOL_FILLER_PHRASES,
]


def test_system_prompt_is_a_non_empty_string():
    assert isinstance(prompts.SYSTEM_PROMPT, str)
    assert prompts.SYSTEM_PROMPT.strip() != ""


def test_greeting_instruction_is_a_non_empty_string():
    assert isinstance(prompts.GREETING_INSTRUCTION, str)
    assert prompts.GREETING_INSTRUCTION.strip() != ""


def test_fallback_instruction_is_a_non_empty_string():
    assert isinstance(prompts.FALLBACK_INSTRUCTION, str)
    assert prompts.FALLBACK_INSTRUCTION.strip() != ""


def test_human_transfer_phrase_is_a_non_empty_string():
    assert isinstance(prompts.HUMAN_TRANSFER_PHRASE, str)
    assert prompts.HUMAN_TRANSFER_PHRASE.strip() != ""


def test_tool_filler_phrases_is_a_non_empty_list_of_strings():
    assert isinstance(prompts.TOOL_FILLER_PHRASES, list)
    assert len(prompts.TOOL_FILLER_PHRASES) > 0
    assert all(isinstance(phrase, str) and phrase.strip() for phrase in prompts.TOOL_FILLER_PHRASES)


def test_tool_filler_phrases_has_at_least_five_items():
    assert len(prompts.TOOL_FILLER_PHRASES) >= 5


def test_no_prompt_contains_markdown_characters():
    for text in _ALL_PROMPT_STRINGS:
        for marker in _MARKDOWN_MARKERS:
            assert marker not in text, f"found markdown marker {marker!r} in: {text!r}"
