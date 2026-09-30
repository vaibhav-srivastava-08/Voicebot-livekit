"""Tracks one call's lifecycle in Redis (see shared.redis_client) so any worker replica can see
a call's progress, and a redial/retry for the same lead can resume rather than starting blind.

KEY DESIGN (shareable across replicas + idempotent resume)
Call state is keyed on the customer's phone number (the dispatch's lead_phone dynamic
variable) when available, falling back to the LiveKit room name otherwise (e.g. ad hoc test
calls via the browser debug page, which have no lead_phone - see call_identity()). Redis
itself - not any one worker's memory - is the source of truth, so:
  - any worker replica computing the same key sees the same record (shareable across replicas)
  - the record outlives any single worker process (survives a worker restart)
  - a redial for the SAME lead (a new LiveKit room - rooms aren't reused across attempts)
    resolves to the SAME key, so start() finds the prior record and resumes it (preserves
    created_at and any already-set checkpoints, increments resume_count) instead of starting
    blind - "idempotent resume where feasible". Full mid-conversation resume isn't possible (a
    dropped call can't un-drop), but the call's *history* survives to inform the next attempt.

FAILURE HANDLING
Every Redis operation here is wrapped in _safe() - a Redis outage is logged and the call
continues normally. State tracking is an observability layer, never a reason to fail a call.

WHY EACH FIELD IS STORED
  - room_name: which LiveKit room this attempt used - correlates with agent logs, which are
    keyed by room_name (see get_logger usage in agent.main).
  - lead_name / lead_phone / objective / information_to_collect / prior_interaction_summary:
    the dynamic variables this call started with (see agent.prompts.build_system_prompt) -
    stored so a CRM/dashboard can see what the agent was told to do without re-reading LiveKit
    dispatch metadata, and so a resumed attempt can be compared against the previous one.
  - created_at: when this identity's call state was FIRST created - preserved across resumes,
    so it reflects the true start of the (possibly multi-attempt) engagement with this lead.
  - updated_at: last time any field changed - lets a dashboard flag "stale" calls (e.g. a
    worker crashed mid-conversation and never reached a terminal checkpoint).
  - worker_id: which worker process is/was handling this call - useful when running multiple
    replicas, to know which one's logs to check.
  - resume_count: how many times this identity has (re)joined; 0 on a first-ever join. A
    non-zero value is itself a signal worth surfacing (repeated redials to the same lead).
  - status: "in_progress" | "completed" | "interrupted" - the coarse outcome.
  - checkpoints.qualified_at: the first successful search_knowledge_base call - a proxy for
    "the customer engaged with a specific product question". Not a guaranteed BANT-qualified
    lead: there is no dedicated qualification tool/signal in this system, so this is
    deliberately approximate - see _on_function_tools_executed.
  - checkpoints.details_captured_at: reaching a minimum assistant-turn count - another
    approximate proxy. Name/model/variant/colour capture happens conversationally inside the
    LLM's own reasoning (per SYSTEM_PROMPT's STATE PERSISTENCE rules), not via a dedicated tool
    call, so there is no exact structured signal for "all fields captured" to hook into; see
    _DETAILS_CAPTURED_TURN_THRESHOLD.
  - checkpoints.handed_off_at: when the call ended cleanly. Under this persona's design, "call
    ended" and "handed off to a sales advisor" are the same moment: EndCallTool is only ever
    invoked after the confirmation summary + warm closing, per the system prompt's own
    TERMINATION RULE - there is no separate "handoff" event to hook beyond the session closing.

KNOWN LIMITATION: updates are read-modify-write (get the full record, mutate, set the full
record back), not atomic field updates. Safe for this system's actual concurrency (one worker
owns one active call's writes at a time; the only reader of an in-progress record is external
tooling), but would need real locking/transactions if multiple workers ever wrote the same
identity concurrently.
"""

import asyncio
import time
from typing import Any

from shared import redis_client
from shared.logger import get_logger

# Assistant-turn count after which we consider "details captured" reached, absent a dedicated
# structured-capture signal (see module docstring). Picked from the system prompt's own SALES
# FLOW shape: greeting-reply, product discovery, variant question, colour question, and the
# confirmation summary are each roughly one assistant turn - by the 5th, the core fields are
# typically in hand. Deliberately approximate; tune if real call transcripts suggest otherwise.
_DETAILS_CAPTURED_TURN_THRESHOLD = 5

_DYNAMIC_VARIABLE_FIELDS = (
    "lead_name",
    "lead_phone",
    "objective",
    "information_to_collect",
    "prior_interaction_summary",
)

# CloseReason values (livekit.agents.voice.events.CloseReason) that represent an intentional,
# completed call rather than a drop/crash/error - see _finalize.
_CLEAN_CLOSE_REASONS = {"user_initiated", "task_completed"}


def call_identity(room_name: str, dynamic_variables: dict[str, Any]) -> str:
    """The stable key a call's Redis state is stored under: lead_phone when the dialer
    provided one (so a redial for the same lead resumes the same record - see module
    docstring), else the room name (e.g. ad hoc test calls with no lead_phone)."""
    lead_phone = str(dynamic_variables.get("lead_phone") or "").strip()
    return lead_phone or room_name


class CallStateTracker:
    """Creates/resumes a call's Redis state on join, updates it at observable checkpoints via
    AgentSession's own events, and finalizes it when the session closes. See the module
    docstring for the full field-by-field rationale and the resume/failure-handling design.
    """

    def __init__(self, logger: Any = None) -> None:
        self._logger = logger or get_logger(__name__)
        self._identity: str | None = None
        self._state: dict[str, Any] | None = None
        self._assistant_turn_count = 0

    async def start(self, room_name: str, worker_id: str, dynamic_variables: dict[str, Any]) -> None:
        """Create or resume this call's state. Call once, before attach(), as early in the
        job as possible (see agent.main.entrypoint) - even a call that fails before the
        session starts is worth having a record of."""
        self._identity = call_identity(room_name, dynamic_variables)
        now = time.time()

        existing = await self._safe(redis_client.get_call_state(self._identity))

        if existing:
            resume_count = existing.get("resume_count", 0) + 1
            self._logger.info(
                "resuming call state for identity=%s (resume_count -> %d)",
                self._identity,
                resume_count,
            )
            state = existing
            state["resume_count"] = resume_count
        else:
            state = {
                "created_at": now,
                "resume_count": 0,
                "checkpoints": {
                    "qualified_at": None,
                    "details_captured_at": None,
                    "handed_off_at": None,
                },
            }

        state["room_name"] = room_name
        state["worker_id"] = worker_id
        state["updated_at"] = now
        state["status"] = "in_progress"
        for field_name in _DYNAMIC_VARIABLE_FIELDS:
            value = dynamic_variables.get(field_name)
            if value:
                state[field_name] = value

        self._state = state
        await self._safe(redis_client.set_call_state(self._identity, state))

    def attach(self, session: Any) -> None:
        """Register this tracker's listeners on `session`. Call once per AgentSession, after
        start()."""
        session.on("function_tools_executed", self._on_function_tools_executed)
        session.on("conversation_item_added", self._on_conversation_item_added)
        session.on("close", self._on_close)

    # -- event hooks: AgentSession's EventEmitter calls these synchronously, so each schedules
    # its actual (async, Redis-touching) work as a background task rather than awaiting inline.

    def _on_function_tools_executed(self, ev: Any) -> None:
        if any(call.name == "search_knowledge_base" for call in ev.function_calls):
            asyncio.create_task(self._mark_checkpoint("qualified_at"))

    def _on_conversation_item_added(self, ev: Any) -> None:
        item = ev.item
        if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "assistant":
            self._assistant_turn_count += 1
            if self._assistant_turn_count >= _DETAILS_CAPTURED_TURN_THRESHOLD:
                asyncio.create_task(self._mark_checkpoint("details_captured_at"))

    def _on_close(self, ev: Any) -> None:
        asyncio.create_task(self._finalize(ev))

    # -- the actual async work --

    async def _mark_checkpoint(self, name: str) -> None:
        if self._state is None or self._state["checkpoints"].get(name):
            return  # not started yet, or already recorded - checkpoints are set-once
        self._state["checkpoints"][name] = time.time()
        self._state["updated_at"] = time.time()
        self._logger.info(
            "call state checkpoint reached: %s (identity=%s)", name, self._identity
        )
        await self._safe(redis_client.set_call_state(self._identity, self._state))

    async def _finalize(self, ev: Any) -> None:
        if self._state is None or self._identity is None:
            return

        reason = getattr(ev, "reason", None)
        reason_value = getattr(reason, "value", str(reason))
        clean_end = reason_value in _CLEAN_CLOSE_REASONS

        if clean_end and not self._state["checkpoints"].get("handed_off_at"):
            self._state["checkpoints"]["handed_off_at"] = time.time()
        self._state["status"] = "completed" if clean_end else "interrupted"
        self._state["updated_at"] = time.time()

        if clean_end:
            # The call finished normally - clear its state immediately rather than waiting out
            # the 2 hour TTL, since there's nothing left to resume.
            self._logger.info(
                "call ended cleanly (%s), clearing call state for identity=%s",
                reason_value,
                self._identity,
            )
            await self._safe(redis_client.delete_call_state(self._identity))
        else:
            # Leave the record (relying on its TTL to eventually expire it): a redial for this
            # same lead should find it and resume rather than starting blind.
            self._logger.warning(
                "call ended abnormally (%s), leaving call state for identity=%s "
                "for possible resume on retry",
                reason_value,
                self._identity,
            )
            await self._safe(redis_client.set_call_state(self._identity, self._state))

    async def _safe(self, coro: Any) -> Any:
        """Run a shared.redis_client coroutine, logging and returning None on ANY failure
        instead of propagating - a Redis outage must never crash a live call."""
        try:
            return await coro
        except Exception:
            self._logger.warning(
                "call state Redis operation failed, continuing without it", exc_info=True
            )
            return None
