"""Agent entrypoint: registers the LiveKit worker and dispatches the voice assistant per job."""

# .env must be loaded into the real environment before anything else: the LiveKit CLI reads
# LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET directly from os.environ, not from our
# pydantic-settings object, which only loads .env into its own fields.
from dotenv import load_dotenv

load_dotenv()

from livekit import agents  # noqa: E402 - must follow load_dotenv()

from agent.assistant import Assistant, build_session  # noqa: E402
from agent.prompts import GREETING_INSTRUCTION  # noqa: E402
from shared.logger import get_logger  # noqa: E402


async def entrypoint(ctx: agents.JobContext) -> None:
    """LiveKit job entrypoint invoked for each dispatched room."""
    logger = get_logger("agent", room_name=ctx.room.name)

    try:
        logger.info("agent starting")

        await ctx.connect()

        session = build_session()
        await session.start(agent=Assistant(), room=ctx.room)
        logger.info("agent ready")

        await session.generate_reply(instructions=GREETING_INSTRUCTION)
        logger.info("greeting sent")
    except Exception:
        logger.error("agent entrypoint failed", exc_info=True)
        raise


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
