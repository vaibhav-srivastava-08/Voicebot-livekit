"""Extracts and cross-checks numeric claims (prices, specs) between a generated reply and the
knowledge-base text actually retrieved for that turn - a safety net against a spoken price or
spec that doesn't come from what was retrieved.

Understands both spoken English number words - including the Indian lakh/crore numbering
Priya's system prompt requires ("eight lakh thirty thousand") - and raw digit-form numbers as
found in KB text ("8 lakh 30 thousand", "27.97"). Both parse to the same canonical numeric
value, so comparing a spoken reply against raw KB text works without converting either side.

This module only extracts/compares numbers; it has no opinion on what to do with the result -
see agent.assistant for how the agent acts on find_unverified_claims()'s output.
"""

import re

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_WORDS = {**_UNITS, **_TENS}
_MAGNITUDES = {"hundred": 100, "thousand": 1_000, "lakh": 100_000, "lac": 100_000, "crore": 10_000_000}

# Words that mark a nearby small number (< 100, with no magnitude word of its own) as a genuine
# spec claim worth verifying, rather than an ordinary conversational count ("let me ask one
# thing", "give me a second"). Checked in a small window before/after the number.
_SPEC_UNIT_WORDS = {
    "rupee",
    "rupees",
    "percent",
    "kilometer",
    "kilometers",
    "km",
    "liter",
    "litre",
    "liters",
    "litres",
    "kilowatt",
    "kilowatts",
    "ps",
    "horsepower",
    "hp",
    "airbag",
    "airbags",
    "seat",
    "seats",
    "seater",
    "seaters",
    "seating",
    "degree",
    "degrees",
    "cc",
    "mm",
    "inch",
    "inches",
    "kg",
    "star",
    "stars",
}

_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_UNIT_WINDOW = 3  # tokens checked before/after a number run for a unit word


def _is_digit_token(token: str) -> bool:
    return token[:1].isdigit()


def _is_number_token(token: str) -> bool:
    return token in _NUMBER_WORDS or token in _MAGNITUDES or token == "point" or _is_digit_token(token)


def _combine(tokens: list[str]) -> float | None:
    """Combine a run of digit/number-word/magnitude tokens into one value via the standard
    "total += current * magnitude" algorithm. Handles Western (thousand) and Indian (lakh,
    crore) numbering identically, and digit/word-mixed text like "8 lakh 11 thousand 400"
    (each digit token is folded into `current` exactly like a small number word)."""
    total = 0.0
    current = 0.0
    seen = False
    for tok in tokens:
        if tok in _MAGNITUDES:
            total += (current or 1) * _MAGNITUDES[tok]
            current = 0.0
            seen = True
        elif tok in _NUMBER_WORDS:
            current += _NUMBER_WORDS[tok]
            seen = True
        elif _is_digit_token(tok):
            current += float(tok)
            seen = True
        else:
            return None
    return (total + current) if seen else None


def _rewrite_colloquial_hundreds(tokens: list[str]) -> list[str]:
    """Rewrite an adjacent "<units 1-9> <tens>" pair (e.g. "three", "sixty") into the single
    three-digit value it represents in natural colloquial speech ("three sixty" -> "360",
    as in "a three sixty camera" - straight from this persona's own system prompt example).
    No one says "three sixty" in English to mean the sum 3+60=63, so this is unambiguous."""
    result: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _UNITS and 1 <= _UNITS[tok] <= 9 and i + 1 < len(tokens) and tokens[i + 1] in _TENS:
            result.append(str(_UNITS[tok] * 100 + _TENS[tokens[i + 1]]))
            i += 2
        else:
            result.append(tok)
            i += 1
    return result


def _decimal_fraction(tokens: list[str]) -> float:
    """Reconstruct the fractional part after a spoken "point" from individually-spoken digit
    words ("five one" -> 0.51) or a bare digit token."""
    digits = ""
    for tok in tokens:
        if tok in _UNITS:
            digits += str(_UNITS[tok])
        elif _is_digit_token(tok):
            digits += tok.replace(".", "")
    return int(digits) / (10 ** len(digits)) if digits else 0.0


class NumberClaim:
    """One numeric value mentioned in text, tagged with enough context to judge whether it's a
    genuine spec/price claim worth verifying vs. conversational filler, a phone number, or a
    bare year mention."""

    __slots__ = ("value", "has_magnitude", "has_unit_word", "digit_length")

    def __init__(self, value: float, has_magnitude: bool, has_unit_word: bool, digit_length: int) -> None:
        self.value = value
        self.has_magnitude = has_magnitude
        self.has_unit_word = has_unit_word
        self.digit_length = digit_length

    @property
    def is_verifiable(self) -> bool:
        if self.digit_length >= 8:
            return False  # phone-number/ID-shaped - not a spec/price claim
        if 1900 <= self.value <= 2099 and not self.has_magnitude and not self.has_unit_word:
            return False  # a bare year mention, e.g. "discontinued in twenty twenty-five"
        if self.has_magnitude or self.has_unit_word:
            return True
        return self.value >= 100  # a bare number this large is unlikely to be conversational

    def __repr__(self) -> str:
        return f"NumberClaim({self.value!r}, verifiable={self.is_verifiable})"


def extract_number_claims(text: str) -> list[NumberClaim]:
    """Extract every numeric claim in `text`. Consecutive number words/digits/magnitude words
    are combined into one value (so "eight lakh thirty thousand" -> 830000.0, matching
    "8 lakh 30 thousand" from raw KB text), and "point" + spoken digit words become a decimal
    ("twenty five point five one" -> 25.51).
    """
    normalized = text.lower().replace("-", " ")
    tokens = _TOKEN_RE.findall(normalized)

    claims: list[NumberClaim] = []
    i = 0
    while i < len(tokens):
        if not _is_number_token(tokens[i]):
            i += 1
            continue

        start = i
        while i < len(tokens) and _is_number_token(tokens[i]):
            i += 1
        run = _rewrite_colloquial_hundreds(tokens[start:i])

        if "point" in run:
            point_idx = run.index("point")
            integer_tokens, decimal_tokens = run[:point_idx], run[point_idx + 1 :]
            integer_value = _combine(integer_tokens) if integer_tokens else 0.0
            value = (integer_value or 0.0) + _decimal_fraction(decimal_tokens)
        else:
            combined = _combine(run)
            if combined is None:
                continue
            value = combined

        has_magnitude = any(tok in _MAGNITUDES for tok in run)
        digit_lengths = [len(tok.split(".")[0]) for tok in run if _is_digit_token(tok)]
        digit_length = max(digit_lengths) if digit_lengths else 0

        context_words = tokens[max(0, start - _UNIT_WINDOW) : start] + tokens[i : i + _UNIT_WINDOW]
        has_unit_word = any(word in _SPEC_UNIT_WORDS for word in context_words)

        claims.append(NumberClaim(value, has_magnitude, has_unit_word, digit_length))

    return claims


def _tolerance_for(value: float) -> float:
    """Verification tolerance scaled to magnitude, matching the prompt's own rounding rules:
    prices round to the nearest thousand, specs to the nearest whole unit - so a stated value
    within that same rounding step of a KB value counts as a match, not a hallucination."""
    if value >= 1000:
        return 1000.0
    if value >= 100:
        return 5.0
    return 1.0


def find_unverified_claims(reply_text: str, context_texts: list[str]) -> list[float]:
    """Return the value of every verifiable numeric claim in `reply_text` that has no
    approximate match among the numbers found in `context_texts` (the KB text actually
    retrieved this turn). An empty list means every claim checks out (or there was nothing
    worth checking)."""
    context_values = [claim.value for text in context_texts for claim in extract_number_claims(text)]

    unverified = []
    for claim in extract_number_claims(reply_text):
        if not claim.is_verifiable:
            continue
        tolerance = _tolerance_for(claim.value)
        if not any(abs(claim.value - ctx_value) <= tolerance for ctx_value in context_values):
            unverified.append(claim.value)
    return unverified
