"""Function tools exposed to the LiveKit agent for tool-calling."""

import random

from livekit.agents import RunContext, function_tool

from agent import knowledge_base
from agent.prompts import TOOL_FILLER_PHRASES
from shared.logger import get_logger

logger = get_logger(__name__)


def _pick_filler_phrase(language: str | None) -> str:
    """Pick a random filler phrase matching the detected language ("hi" -> Hindi/Hinglish
    variants, anything else -> English), mirroring Priya's own code-switching rules."""
    phrases = TOOL_FILLER_PHRASES["hi"] if (language or "").lower().startswith("hi") else TOOL_FILLER_PHRASES["en"]
    return random.choice(phrases)


@function_tool
async def search_knowledge_base(ctx: RunContext, query: str) -> str:
    """Search the Maruti Suzuki knowledge base for model prices, specs, features, variants,
    colours, or answers to car-buying questions. Always call this before stating any price,
    spec, feature, or comparison - never answer from memory or guess.

    Returns a "STRUCTURED FACTS" section (when the match includes one) followed by a
    "PASSAGES" section:
      - STRUCTURED FACTS lists one variant/spec per line, e.g.
        "Brezza LXi MT | specs: 1.5 liter, 5-speed MT, petrol | price: 8 lakh 11 thousand 400".
        Each line is a single, complete fact - never combine numbers from two different lines
        (e.g. don't quote one variant's price alongside another variant's specs).
      - PASSAGES is supporting prose for context/features/comparisons.
    All values are raw KB text (may contain digits, decimals, symbols, or bullets); convert
    them to spoken English words per Hardcoded Rule Two before speaking. If nothing relevant
    is found, say so honestly per Hardcoded Rule One rather than guessing.
    """
    logger.info("search_knowledge_base called with query=%s", query)

    language = getattr(ctx.userdata, "last_detected_language", None)
    # Speaks a filler ("let me check that for you" / "ji, main abhi check karti hoon") if the
    # session is idle for a moment while this runs, so the caller never hits dead air. The
    # framework's own dwell/idle scheduling means fast lookups (like this one) usually finish
    # before it would ever fire - it only speaks up if the call genuinely takes a moment.
    async with ctx.with_filler(lambda step: _pick_filler_phrase(language)):
        return knowledge_base.search(query)
