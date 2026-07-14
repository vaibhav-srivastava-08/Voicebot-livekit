"""Redis connection management and helper functions shared across services."""

import json
import os

import redis.asyncio as redis
from redis.exceptions import RedisError

from shared.logger import get_logger

logger = get_logger(__name__)

_REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_CALL_STATE_KEY_PREFIX = "call_state:"
_CALL_STATE_TTL_SECONDS = 2 * 60 * 60

_client: redis.Redis | None = None


async def get_client() -> redis.Redis:
    """Return a connected, lazily-initialized async Redis client shared across the process."""
    global _client
    if _client is None:
        candidate = redis.from_url(_REDIS_URL, decode_responses=True)
        try:
            await candidate.ping()
        except RedisError:
            logger.error("Failed to connect to Redis at %s", _REDIS_URL)
            raise
        _client = candidate
    return _client


def _call_state_key(stream_id: str) -> str:
    return f"{_CALL_STATE_KEY_PREFIX}{stream_id}"


async def set_call_state(stream_id: str, data: dict) -> None:
    """Persist call state for `stream_id` as JSON with a 2 hour TTL."""
    client = await get_client()
    try:
        await client.set(_call_state_key(stream_id), json.dumps(data), ex=_CALL_STATE_TTL_SECONDS)
    except RedisError:
        logger.error("Failed to set call state for stream_id=%s", stream_id)
        raise


async def get_call_state(stream_id: str) -> dict | None:
    """Fetch and deserialize call state for `stream_id`, or None if no state is stored."""
    client = await get_client()
    try:
        raw = await client.get(_call_state_key(stream_id))
    except RedisError:
        logger.error("Failed to get call state for stream_id=%s", stream_id)
        raise
    return json.loads(raw) if raw is not None else None


async def delete_call_state(stream_id: str) -> None:
    """Delete call state for `stream_id`."""
    client = await get_client()
    try:
        await client.delete(_call_state_key(stream_id))
    except RedisError:
        logger.error("Failed to delete call state for stream_id=%s", stream_id)
        raise
