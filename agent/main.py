"""Agent entrypoint: registers the LiveKit worker and dispatches the voice assistant per job."""

# .env must be loaded into the real environment before anything else: the LiveKit CLI reads
# LIVEKIT_URL/LIVEKIT_API_KEY/LIVEKIT_API_SECRET directly from os.environ, not from our
# pydantic-settings object, which only loads .env into its own fields.
from dotenv import load_dotenv

load_dotenv()

import json  # noqa: E402

from livekit import agents  # noqa: E402 - must follow load_dotenv()
from livekit.plugins import silero  # noqa: E402

from agent.assistant import Assistant, build_session  # noqa: E402
from agent.call_state import CallStateTracker  # noqa: E402
from agent.config import settings  # noqa: E402
from agent.prompts import build_system_prompt  # noqa: E402
from shared.logger import get_logger  # noqa: E402
from shared.timing import TurnLatencyTracker  # noqa: E402


def prewarm(proc: agents.JobProcess) -> None:
    """Load the VAD model once per idle worker process, before any job is assigned to it.

    silero.VAD.load() is a blocking ONNX model load; doing it here instead of inside
    build_session() keeps it off the critical path of every individual call.
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: agents.JobContext) -> None:
    """LiveKit job entrypoint invoked for each dispatched room."""
    logger = get_logger("agent", room_name=ctx.room.name)

    try:
        logger.info("agent starting")

        # Per-call dynamic variables (objective, lead_name, ...) arrive as a JSON string on the
        # job's dispatch metadata - e.g. the Exotel-triggered dispatch attaches this per call.
        # Missing/blank/invalid metadata falls back to an empty dict (every variable defaults
        # to "", matching the prompt's own "if a variable is missing, follow the normal flow" rule).
        try:
            dynamic_variables = json.loads(ctx.job.metadata) if ctx.job.metadata else {}
        except json.JSONDecodeError:
            logger.error("job metadata was not valid JSON, ignoring it: %r", ctx.job.metadata)
            dynamic_variables = {}

        # Call-state tracking in Redis: create/resume the record before connecting, so even a
        # call that fails before the session starts leaves a trace. See agent/call_state.py for
        # the full key design (shareable across worker replicas, idempotent resume), checkpoint
        # definitions, and graceful degradation if Redis is unavailable.
        call_state_tracker = CallStateTracker(logger=get_logger("call_state", room_name=ctx.room.name))
        await call_state_tracker.start(
            room_name=ctx.room.name, worker_id=ctx.worker_id, dynamic_variables=dynamic_variables
        )

        await ctx.connect()

        instructions = build_system_prompt(**dynamic_variables)
        session = build_session(vad=ctx.proc.userdata["vad"])

        # Per-turn latency instrumentation: logs one "turn_latency" JSON line per turn and a
        # "turn_latency_summary" (rolling p50/p95) line when the session closes. Built entirely
        # from AgentSession's own events - see shared/timing.py.
        TurnLatencyTracker(logger=get_logger("turn_latency", room_name=ctx.room.name)).attach(session)
        call_state_tracker.attach(session)

        await session.start(agent=Assistant(instructions=instructions), room=ctx.room)
        logger.info("agent ready")

        # No agent-generated greeting: this is an outbound call where a pre-recorded greeting
        # has already played before the agent joins. The agent's first turn is generated in
        # reaction to the caller's reply to that greeting, per the system prompt's own rules.
    except Exception:
        logger.error("agent entrypoint failed", exc_info=True)
        raise


def _build_worker_options() -> agents.WorkerOptions:
    """Build WorkerOptions from agent.config.settings. See DEPLOY.md's "Scaling knobs" section
    for what each of these controls and recommended production values; defaults here reproduce
    this project's previous hardcoded behavior unchanged."""
    kwargs = dict(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        # dev mode defaults to 0 idle processes (cold-start per job); keep at least one warm
        # so a call never waits on process spin-up + VAD load. This is a standing pool size,
        # not a hard concurrency cap - see NUM_IDLE_PROCESSES in DEPLOY.md.
        num_idle_processes=settings.num_idle_processes,
        job_memory_limit_mb=settings.job_memory_limit_mb,
        drain_timeout=settings.drain_timeout,
    )
    # Only pass load_threshold when explicitly set - omitting it lets livekit-agents apply its
    # own mode-aware default (0.7 under `start`, unlimited under `dev`) instead of us silently
    # overriding it with a Python-level default of our own.
    if settings.worker_load_threshold is not None:
        kwargs["load_threshold"] = settings.worker_load_threshold
    return agents.WorkerOptions(**kwargs)


if __name__ == "__main__":
    agents.cli.run_app(_build_worker_options())
