"""Tests for agent.prompts: constant presence, shape, and voice-safe (non-markdown) content."""

import re

from agent import prompts

# Real markdown syntax only - NOT a bare "-" or "*" anywhere, since SYSTEM_PROMPT legitimately
# contains hyphenated words ("twenty-eight"), spelled-out acronyms ("A-D-A-S"), and number
# ranges as ordinary prose, none of which are markdown formatting.
_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")
_HEADER_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*]\s", re.MULTILINE)

_ALL_PROMPT_STRINGS = [
    prompts.SYSTEM_PROMPT,
    prompts.GREETING_INSTRUCTION,
    prompts.FALLBACK_INSTRUCTION,
    *prompts.TOOL_FILLER_PHRASES["en"],
    *prompts.TOOL_FILLER_PHRASES["hi"],
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


def test_tool_filler_phrases_has_en_and_hi_non_empty_lists_of_strings():
    assert isinstance(prompts.TOOL_FILLER_PHRASES, dict)
    assert set(prompts.TOOL_FILLER_PHRASES) == {"en", "hi"}
    for phrases in prompts.TOOL_FILLER_PHRASES.values():
        assert isinstance(phrases, list)
        assert len(phrases) > 0
        assert all(isinstance(phrase, str) and phrase.strip() for phrase in phrases)


def test_tool_filler_phrases_has_at_least_five_items_per_language():
    for phrases in prompts.TOOL_FILLER_PHRASES.values():
        assert len(phrases) >= 5


def test_hindi_tool_filler_phrases_use_feminine_verb_forms():
    # Priya is a female persona - "sakti"/"karti"/"rahi"/"leti" are feminine forms; the
    # masculine counterparts ("sakta"/"karta"/"raha"/"leta") must never appear.
    masculine_endings = ("sakta", "karta", "raha", "leta", "gaya")
    for phrase in prompts.TOOL_FILLER_PHRASES["hi"]:
        lowered = phrase.lower()
        for word in masculine_endings:
            assert word not in lowered, f"found masculine form {word!r} in: {phrase!r}"


def test_no_prompt_contains_markdown_syntax():
    for text in _ALL_PROMPT_STRINGS:
        assert not _BOLD_RE.search(text), f"found **bold** markdown in: {text!r}"
        assert not _HEADER_RE.search(text), f"found a '#' header line in: {text!r}"
        assert not _BULLET_RE.search(text), f"found a '-'/'*' bullet list line in: {text!r}"


def test_build_system_prompt_fills_in_dynamic_variables():
    filled = prompts.build_system_prompt(
        objective="follow up on Brezza enquiry",
        lead_name="Rahul",
    )
    assert "{{objective}}" not in filled
    assert "{{lead_name}}" not in filled
    assert "follow up on Brezza enquiry" in filled
    assert "Rahul" in filled


def test_build_system_prompt_defaults_missing_variables_to_empty_string():
    filled = prompts.build_system_prompt()
    for name in prompts.DYNAMIC_VARIABLE_DEFAULTS:
        assert "{{" + name + "}}" not in filled
