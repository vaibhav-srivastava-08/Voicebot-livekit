"""Tests for shared.number_verification: number extraction (spoken English words, including
Indian lakh/crore numbering, and raw digit-form KB text) and cross-checking claims against
retrieved context."""

from shared.number_verification import extract_number_claims, find_unverified_claims


def _values(text: str) -> list[float]:
    return [claim.value for claim in extract_number_claims(text)]


class TestExtractNumberClaims:
    def test_spoken_lakh_thousand_hundred_combination(self):
        assert _values("eight lakh eleven thousand four hundred rupees") == [811400.0]

    def test_digit_form_matches_the_same_spoken_value(self):
        assert _values("8 lakh 11 thousand 400") == [811400.0]

    def test_bare_word_price_without_hundreds(self):
        assert _values("approximately eight lakh thirty thousand") == [830000.0]

    def test_bare_digit_with_unit_word_is_verifiable(self):
        claims = extract_number_claims("six airbags")
        assert claims[0].value == 6.0
        assert claims[0].is_verifiable is True

    def test_colloquial_three_sixty_means_360_not_63(self):
        assert _values("a three sixty camera") == [360.0]
        assert _values("360-degree camera") == [360.0]

    def test_spoken_decimal_point(self):
        assert _values("twenty five point five one km per liter") == [25.51]

    def test_range_extracts_both_bounds_separately(self):
        assert _values("8 lakh 11 thousand 400 to 13 lakh 1 thousand 300") == [811400.0, 1301300.0]

    def test_bare_small_number_is_not_verifiable(self):
        claims = extract_number_claims("let me ask you one question")
        assert claims[0].value == 1.0
        assert claims[0].is_verifiable is False

    def test_year_mention_without_unit_is_not_verifiable(self):
        claims = extract_number_claims("discontinued in the year 2025")
        assert any(c.value == 2025.0 and not c.is_verifiable for c in claims)

    def test_phone_number_shaped_digit_run_is_not_verifiable(self):
        claims = extract_number_claims("my number is 9876543210")
        assert claims[0].is_verifiable is False

    def test_no_numbers_returns_empty_list(self):
        assert extract_number_claims("How can I help you today?") == []


class TestFindUnverifiedClaims:
    def test_verified_price_returns_empty_list(self):
        context = ["Ex-showroom price range: 8 lakh 11 thousand 400 to 13 lakh 1 thousand 300"]
        reply = "The Brezza starts at approximately eight lakh eleven thousand rupees."
        assert find_unverified_claims(reply, context) == []

    def test_hallucinated_price_is_flagged(self):
        context = ["Ex-showroom price range: 8 lakh 11 thousand 400 to 13 lakh 1 thousand 300"]
        reply = "The Brezza starts at approximately seven lakh fifty thousand rupees."
        unverified = find_unverified_claims(reply, context)
        assert unverified == [750000.0]

    def test_price_rounded_to_nearest_thousand_is_tolerated(self):
        # 811400 rounds to 811000 at the nearest-thousand precision the prompt requires.
        context = ["price: 8 lakh 11 thousand 400"]
        reply = "It costs approximately eight lakh eleven thousand rupees."
        assert find_unverified_claims(reply, context) == []

    def test_spec_number_verified_against_context(self):
        context = ["Key features: 6 airbags (standard on higher variants), 360-degree camera"]
        reply = "It comes with six airbags and a three sixty camera."
        assert find_unverified_claims(reply, context) == []

    def test_hallucinated_spec_number_is_flagged(self):
        context = ["Key features: 6 airbags (standard on higher variants)"]
        reply = "It comes with eight airbags for extra safety."
        assert find_unverified_claims(reply, context) == [8.0]

    def test_conversational_numbers_never_flagged_even_with_no_context(self):
        reply = "Sure, give me one moment to check that for you."
        assert find_unverified_claims(reply, []) == []

    def test_phone_number_never_flagged(self):
        reply = "Thank you, I have your number as nine eight seven six five four three two one zero noted."
        assert find_unverified_claims(reply, []) == []

    def test_multiple_context_chunks_are_all_considered(self):
        context = [
            "Alto K10 price starts at three lakh sixty-nine thousand nine hundred rupees.",
            "Brezza price starts at eight lakh eleven thousand four hundred rupees.",
        ]
        reply = "The Brezza starts at approximately eight lakh eleven thousand rupees."
        assert find_unverified_claims(reply, context) == []

    def test_empty_reply_has_nothing_to_verify(self):
        assert find_unverified_claims("", ["some context"]) == []
