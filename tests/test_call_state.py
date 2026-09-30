"""Tests for agent.call_state: CallStateTracker's Redis-backed call lifecycle tracking.

Uses fakeredis's async client as a drop-in for shared.redis_client's module-level singleton
(monkeypatched directly - the client is a simple lazily-initialized module global, so patching
it is the natural test seam) - no real Redis server needed. Event hooks are exercised against a
plain `rtc.EventEmitter` standing in for an AgentSession, matching tests/test_timing.py's
approach for the same reason (AgentSession itself extends EventEmitter).

CallStateTracker's event handlers schedule their real (Redis-touching) work via
asyncio.create_task rather than running it inline (see agent/call_state.py's module docstring
for why: AgentSession's EventEmitter.emit() is synchronous and never awaits a callback), so
tests await _flush() after emitting an event to let that scheduled task actually run before
asserting on Redis state.
"""

import asyncio
from unittest.mock import MagicMock

import fakeredis
import pytest
from livekit import rtc
from livekit.agents.llm.chat_context import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.voice.events import (
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
)
from redis.exceptions import RedisError

from agent.call_state import CallStateTracker, call_identity
from shared import redis_client


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch):
    """Point shared.redis_client's lazily-initialized singleton at a fresh in-memory fake for
    every test, instead of connecting to a real Redis server."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "_client", client)
    return client


async def _flush():
    """Give a background task scheduled by asyncio.create_task(...) a chance to run to
    completion before asserting on its effects."""
    await asyncio.sleep(0.01)


def _kb_call_event(name="search_knowledge_base"):
    return FunctionToolsExecutedEvent(
        function_calls=[FunctionCall(call_id="c", arguments="{}", name=name)],
        function_call_outputs=[FunctionCallOutput(call_id="c", output="ok", is_error=False)],
    )


def _assistant_turn():
    return ConversationItemAddedEvent(item=ChatMessage(role="assistant", content=["ok"]))


def _user_turn():
    return ConversationItemAddedEvent(item=ChatMessage(role="user", content=["hi"]))


class TestCallIdentity:
    def test_prefers_lead_phone_when_present(self):
        assert call_identity("room-1", {"lead_phone": "+919876543210"}) == "+919876543210"

    def test_falls_back_to_room_name_without_lead_phone(self):
        assert call_identity("room-1", {}) == "room-1"

    def test_blank_lead_phone_falls_back_to_room_name(self):
        assert call_identity("room-1", {"lead_phone": "   "}) == "room-1"


class TestStart:
    @pytest.mark.asyncio
    async def test_creates_new_state_with_expected_shape(self):
        tracker = CallStateTracker()
        await tracker.start(
            room_name="room-1",
            worker_id="worker-1",
            dynamic_variables={
                "lead_name": "Asha",
                "lead_phone": "+91999",
                "objective": "sell a Brezza",
            },
        )

        stored = await redis_client.get_call_state("+91999")
        assert stored["room_name"] == "room-1"
        assert stored["worker_id"] == "worker-1"
        assert stored["lead_name"] == "Asha"
        assert stored["objective"] == "sell a Brezza"
        assert stored["resume_count"] == 0
        assert stored["status"] == "in_progress"
        assert stored["checkpoints"] == {
            "qualified_at": None,
            "details_captured_at": None,
            "handed_off_at": None,
        }

    @pytest.mark.asyncio
    async def test_falls_back_to_room_name_identity_when_no_lead_phone(self):
        tracker = CallStateTracker()
        await tracker.start(room_name="room-debug-1", worker_id="worker-1", dynamic_variables={})

        assert await redis_client.get_call_state("room-debug-1") is not None

    @pytest.mark.asyncio
    async def test_resume_preserves_created_at_and_checkpoints_and_bumps_resume_count(self):
        tracker1 = CallStateTracker()
        session1 = rtc.EventEmitter()
        await tracker1.start(
            room_name="room-1", worker_id="worker-1", dynamic_variables={"lead_phone": "+91999"}
        )
        tracker1.attach(session1)
        session1.emit("function_tools_executed", _kb_call_event())
        await _flush()

        first = await redis_client.get_call_state("+91999")
        assert first["checkpoints"]["qualified_at"] is not None

        tracker2 = CallStateTracker()
        await tracker2.start(
            room_name="room-2", worker_id="worker-2", dynamic_variables={"lead_phone": "+91999"}
        )

        second = await redis_client.get_call_state("+91999")
        assert second["created_at"] == first["created_at"]
        assert second["checkpoints"]["qualified_at"] == first["checkpoints"]["qualified_at"]
        assert second["resume_count"] == 1
        assert second["room_name"] == "room-2"  # latest attempt's room, not the stale one
        assert second["worker_id"] == "worker-2"


class TestCheckpoints:
    @pytest.mark.asyncio
    async def test_qualified_checkpoint_set_on_first_kb_search(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91111"}
        )
        tracker.attach(session)

        session.emit("function_tools_executed", _kb_call_event())
        await _flush()

        state = await redis_client.get_call_state("+91111")
        assert state["checkpoints"]["qualified_at"] is not None

    @pytest.mark.asyncio
    async def test_qualified_checkpoint_ignores_unrelated_tool_calls(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91112"}
        )
        tracker.attach(session)

        session.emit("function_tools_executed", _kb_call_event(name="end_call"))
        await _flush()

        state = await redis_client.get_call_state("+91112")
        assert state["checkpoints"]["qualified_at"] is None

    @pytest.mark.asyncio
    async def test_qualified_checkpoint_is_set_once(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91113"}
        )
        tracker.attach(session)

        session.emit("function_tools_executed", _kb_call_event())
        await _flush()
        first_value = (await redis_client.get_call_state("+91113"))["checkpoints"]["qualified_at"]

        session.emit("function_tools_executed", _kb_call_event())
        await _flush()
        second_value = (await redis_client.get_call_state("+91113"))["checkpoints"]["qualified_at"]

        assert first_value == second_value

    @pytest.mark.asyncio
    async def test_details_captured_checkpoint_set_after_turn_threshold(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91114"}
        )
        tracker.attach(session)

        for _ in range(4):
            session.emit("conversation_item_added", _assistant_turn())
        await _flush()
        state = await redis_client.get_call_state("+91114")
        assert state["checkpoints"]["details_captured_at"] is None

        session.emit("conversation_item_added", _assistant_turn())
        await _flush()
        state = await redis_client.get_call_state("+91114")
        assert state["checkpoints"]["details_captured_at"] is not None

    @pytest.mark.asyncio
    async def test_details_captured_ignores_user_turns(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91115"}
        )
        tracker.attach(session)

        for _ in range(10):
            session.emit("conversation_item_added", _user_turn())
        await _flush()

        state = await redis_client.get_call_state("+91115")
        assert state["checkpoints"]["details_captured_at"] is None


class TestFinalize:
    @pytest.mark.asyncio
    async def test_clean_close_deletes_state_and_sets_handed_off(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91201"}
        )
        tracker.attach(session)

        session.emit("close", CloseEvent(reason=CloseReason.TASK_COMPLETED))
        await _flush()

        assert await redis_client.get_call_state("+91201") is None
        assert tracker._state["checkpoints"]["handed_off_at"] is not None
        assert tracker._state["status"] == "completed"

    @pytest.mark.asyncio
    async def test_user_initiated_close_also_counts_as_clean(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91205"}
        )
        tracker.attach(session)

        session.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))
        await _flush()

        assert await redis_client.get_call_state("+91205") is None

    @pytest.mark.asyncio
    async def test_abnormal_close_preserves_state_for_resume(self):
        tracker = CallStateTracker()
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91202"}
        )
        tracker.attach(session)

        session.emit("close", CloseEvent(reason=CloseReason.PARTICIPANT_DISCONNECTED))
        await _flush()

        state = await redis_client.get_call_state("+91202")
        assert state is not None
        assert state["status"] == "interrupted"
        assert state["checkpoints"]["handed_off_at"] is None

    @pytest.mark.asyncio
    async def test_abnormal_close_allows_later_resume(self):
        tracker1 = CallStateTracker()
        session1 = rtc.EventEmitter()
        await tracker1.start(
            room_name="room-1", worker_id="w1", dynamic_variables={"lead_phone": "+91203"}
        )
        tracker1.attach(session1)
        session1.emit("close", CloseEvent(reason=CloseReason.ERROR))
        await _flush()

        tracker2 = CallStateTracker()
        await tracker2.start(
            room_name="room-2", worker_id="w2", dynamic_variables={"lead_phone": "+91203"}
        )

        state = await redis_client.get_call_state("+91203")
        assert state["resume_count"] == 1
        assert state["status"] == "in_progress"


class TestRedisUnavailable:
    """Redis raising RedisError must never crash the call - see shared/redis_client.py (which
    deliberately raises on failure) vs. agent/call_state.py's CallStateTracker._safe (which
    catches everything and logs a warning instead)."""

    @pytest.mark.asyncio
    async def test_start_does_not_raise_when_redis_get_fails(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RedisError("boom")

        monkeypatch.setattr(redis_client, "get_call_state", _boom)
        logger = MagicMock()
        tracker = CallStateTracker(logger=logger)

        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91301"}
        )

        assert tracker._state is not None  # in-memory state still built despite Redis being down
        logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_start_does_not_raise_when_redis_set_fails(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RedisError("boom")

        monkeypatch.setattr(redis_client, "set_call_state", _boom)
        tracker = CallStateTracker(logger=MagicMock())

        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91302"}
        )

        assert tracker._state["room_name"] == "room-1"

    @pytest.mark.asyncio
    async def test_checkpoint_update_does_not_raise_when_redis_fails(self, monkeypatch):
        tracker = CallStateTracker(logger=MagicMock())
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91303"}
        )
        tracker.attach(session)

        async def _boom(*args, **kwargs):
            raise RedisError("boom")

        monkeypatch.setattr(redis_client, "set_call_state", _boom)

        session.emit("function_tools_executed", _kb_call_event())
        await _flush()  # must not raise

        # in-memory checkpoint still recorded even though the Redis write failed
        assert tracker._state["checkpoints"]["qualified_at"] is not None

    @pytest.mark.asyncio
    async def test_finalize_does_not_raise_when_redis_delete_fails(self, monkeypatch):
        tracker = CallStateTracker(logger=MagicMock())
        session = rtc.EventEmitter()
        await tracker.start(
            room_name="room-1", worker_id="w", dynamic_variables={"lead_phone": "+91304"}
        )
        tracker.attach(session)

        async def _boom(*args, **kwargs):
            raise RedisError("boom")

        monkeypatch.setattr(redis_client, "delete_call_state", _boom)

        session.emit("close", CloseEvent(reason=CloseReason.TASK_COMPLETED))
        await _flush()  # must not raise
