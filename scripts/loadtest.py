"""Load test the browser debug flow: fetch a token from the bridge, join a fresh LiveKit room
(exactly what bridge/static/debug.html's Connect button does), hold the connection briefly, then
disconnect - repeated across many concurrent "calls".

This exercises the SAME dispatch trigger a real call does (a new room being joined causes
LiveKit to dispatch an agent job to it), so it's the right tool for validating horizontal
scaling: how many concurrent rooms this deployment can actually pick up, and how job dispatch
spreads across your agent replicas (see DEPLOY.md's "How job dispatch distributes across
replicas" section) - a `docker compose -f docker-compose.prod.yml logs agent` grepped for
"registered worker"/"agent starting" during a run will show multiple worker IDs picking up jobs.

It does NOT publish real microphone audio or assert anything about conversation quality/latency
- see shared/timing.py's per-turn latency logging for that. This script only proves rooms get
joined and agents get dispatched under concurrency; it isn't a substitute for a real call.

Usage (run from the project's own venv - it reuses aiohttp and the `livekit` rtc client, both
already required by livekit-agents/this project, no extra install needed):

    python scripts/loadtest.py --bridge-url http://localhost:8080 --total 50 --concurrency 10

Or against a running docker-compose.prod.yml deployment's exposed bridge port:

    python scripts/loadtest.py --bridge-url http://<host>:8080 --total 200 --concurrency 40 \\
        --hold-seconds 30
"""

import argparse
import asyncio
import random
import string
import time
from dataclasses import dataclass

import aiohttp
from livekit import rtc


@dataclass
class CallResult:
    index: int
    outcome: str  # "ok" | "token_error" | "connect_error"
    connect_seconds: float | None
    error: str | None = None


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


async def _run_one_call(
    session: aiohttp.ClientSession, bridge_url: str, index: int, hold_seconds: float
) -> CallResult:
    identity = f"loadtest-{index}-{_random_suffix()}"
    # A fresh room name per call, same as debug.html - LiveKit only dispatches a new agent job
    # when a room is newly created, so reusing a room name would just silently skip dispatch.
    room_name = f"loadtest-room-{index}-{_random_suffix()}"

    start = time.monotonic()
    try:
        async with session.get(
            f"{bridge_url}/token", params={"room": room_name, "identity": identity}
        ) as resp:
            resp.raise_for_status()
            token_data = await resp.json()
    except Exception as exc:
        return CallResult(index, "token_error", None, str(exc))

    room = rtc.Room()
    try:
        await room.connect(token_data["url"], token_data["token"])
    except Exception as exc:
        return CallResult(index, "connect_error", None, str(exc))

    connect_seconds = time.monotonic() - start
    try:
        await asyncio.sleep(hold_seconds)
    finally:
        await room.disconnect()

    return CallResult(index, "ok", connect_seconds)


async def _run_load_test(
    bridge_url: str, total: int, concurrency: int, hold_seconds: float
) -> list[CallResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bound_call(session: aiohttp.ClientSession, index: int) -> CallResult:
        async with semaphore:
            return await _run_one_call(session, bridge_url, index, hold_seconds)

    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*(bound_call(session, i) for i in range(total)))


def _print_summary(results: list[CallResult]) -> None:
    ok = [r for r in results if r.outcome == "ok"]
    failed = [r for r in results if r.outcome != "ok"]

    print(f"\n{len(ok)}/{len(results)} calls connected and held successfully")

    if ok:
        connect_times = sorted(r.connect_seconds for r in ok)
        p50 = connect_times[len(connect_times) // 2]
        p95 = connect_times[int(len(connect_times) * 0.95)]
        print(f"token+connect latency: p50={p50:.2f}s p95={p95:.2f}s max={connect_times[-1]:.2f}s")

    if failed:
        print(f"\n{len(failed)} failed:")
        by_outcome: dict[str, int] = {}
        for r in failed:
            by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
        for outcome, count in by_outcome.items():
            print(f"  {outcome}: {count}")
        # One example per failure kind, not all of them - enough to diagnose without a wall of
        # repeated stack traces when e.g. every call fails the same way.
        seen_outcomes = set()
        for r in failed:
            if r.outcome not in seen_outcomes:
                seen_outcomes.add(r.outcome)
                print(f"    example ({r.outcome}): {r.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bridge-url", default="http://localhost:8080", help="Bridge base URL")
    parser.add_argument("--total", type=int, default=20, help="Total calls to simulate")
    parser.add_argument("--concurrency", type=int, default=10, help="Max calls in flight at once")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=20.0,
        help="How long each simulated call stays connected before disconnecting",
    )
    args = parser.parse_args()

    print(
        f"running {args.total} calls (concurrency={args.concurrency}) against "
        f"{args.bridge_url}, holding each for {args.hold_seconds}s..."
    )
    results = asyncio.run(
        _run_load_test(args.bridge_url, args.total, args.concurrency, args.hold_seconds)
    )
    _print_summary(results)


if __name__ == "__main__":
    main()
