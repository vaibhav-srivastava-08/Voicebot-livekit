"""Standalone harness for testing the voice agent through your microphone, with no telephony.

Run with: python agent/console_test.py console
"""

# .env must be loaded before any other project module is imported, since agent.config builds
# its AgentSettings() singleton (which requires LIVEKIT_*, DEEPGRAM_API_KEY, etc.) at import time.
from dotenv import load_dotenv

load_dotenv()

from livekit import agents  # noqa: E402 - must follow load_dotenv()

from agent.assistant import Assistant, build_session  # noqa: E402
from agent.prompts import GREETING_INSTRUCTION  # noqa: E402
from shared.logger import get_logger  # noqa: E402


async def entrypoint(ctx: agents.JobContext) -> None:
    """Console entrypoint: same wiring as agent.main.entrypoint, run against mic/speaker instead of a room."""
    logger = get_logger("console_test", room_name=ctx.room.name)
    logger.info("console test starting")

    await ctx.connect()

    # STT check: speak into your mic. Your words should show up as a transcript in the terminal
    # within a second or so of you finishing a sentence. If nothing appears, check DEEPGRAM_API_KEY
    # and that your OS is actually feeding audio from the correct input device.
    #
    # LLM check: the printed reply text should be short, conversational, spoken-style sentences
    # (per SYSTEM_PROMPT) — not a wall of text, not bullet points, not markdown.
    #
    # TTS check: you should hear the reply spoken back through your speakers in the configured
    # Cartesia voice almost immediately after the reply text is printed. If you see text but hear
    # nothing, check CARTESIA_API_KEY / CARTESIA_VOICE_ID and your output device.
    session = build_session()
    await session.start(agent=Assistant(), room=ctx.room)
    logger.info("console test ready")

    await session.generate_reply(instructions=GREETING_INSTRUCTION)
    logger.info("greeting sent")

    # Tool-call check: ask something like "can you look up my account, my number is 555 123 4567".
    # Watch the terminal for a tool-call event for get_customer_info firing, and the spoken reply
    # should reflect the mock data it returns ("Test Customer", account status "active").
    #
    # Interruption check: start talking while the agent is still mid-sentence. Playback should stop
    # almost immediately (barge-in). If the bot keeps talking over you or finishes its full sentence
    # regardless, VAD/interruption handling is not working.


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))


# ---------------------------------------------------------------------------------------------
# README: manual console test
# ---------------------------------------------------------------------------------------------
"""
COMMAND TO RUN
    python agent/console_test.py console

    The `console` subcommand is what tells LiveKit Agents' CLI to use its built-in console
    transport (local mic in / speaker out) instead of connecting to a real LiveKit room over
    telephony or WebRTC. Everything else in this script is identical to the production worker.

WHAT A PASSING TEST LOOKS LIKE
    1. On startup, you see "console test starting" then "console test ready" logged as JSON lines.
    2. You immediately hear the agent speak a short greeting that names the company and asks how
       it can help (GREETING_INSTRUCTION).
    3. When you speak, a transcript of your speech appears in the terminal within about a second.
    4. The agent's text reply appears as short, spoken-style sentences, with no lists or markdown,
       followed almost immediately by the agent's voice speaking that same reply out loud.
    5. Asking to look up an account triggers a visible tool-call log/event for get_customer_info,
       and the agent's next reply references the mock data (Test Customer, active).
    6. Speaking while the agent is mid-sentence cuts the agent off within a fraction of a second
       (barge-in), and it starts listening to your new input.

WHAT A FAILING TEST LOOKS LIKE, AND THE LIKELY CAUSE
    - Script crashes immediately with a pydantic ValidationError on AgentSettings:
        .env is missing or missing required keys (LIVEKIT_*, DEEPGRAM_API_KEY, CARTESIA_API_KEY,
        GROQ_API_KEY, CARTESIA_VOICE_ID). Fill in .env from .env.example.
    - No transcript ever appears when you speak:
        Wrong/no microphone selected at the OS level, or DEEPGRAM_API_KEY is invalid/expired.
    - Transcript appears but the reply text is nonsensical, generic, or ignores your SYSTEM_PROMPT
      rules (long sentences, lists, markdown):
        GROQ_API_KEY is wrong/rate-limited (check for LLM error logs), or prompts.py was edited
        and no longer matches what SYSTEM_PROMPT is supposed to say.
    - Reply text is printed but you hear no audio:
        CARTESIA_API_KEY or CARTESIA_VOICE_ID is invalid, or your system's default output device
        is muted/wrong.
    - Asking for an account lookup never fires a tool call:
        The LLM didn't decide to call the tool — try a more explicit phrasing, or check that
        get_customer_info is still decorated with @function_tool on the Assistant class.
    - Interrupting the agent does not stop playback:
        VAD isn't detecting your speech over the agent's own output (check silero.VAD.load() wired
        correctly in build_session), or system audio echo/feedback is confusing turn detection.
"""
