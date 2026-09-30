"""Load-test harness: simulates N concurrent "browser-style" callers against the running
stack (a LiveKit dev/prod server + this project's agent worker(s) + bridge), the same way
bridge/static/debug.html does - fetch a token from the bridge's /token endpoint, join a fresh
room, publish a microphone-style audio track - except the audio is synthesized speech (see
fixtures.py) instead of a real human talking into a browser mic, so it can run unattended at
whatever concurrency you point it at.

This is NOT the telephony/Exotel path (bridge/dispatcher.py isn't involved) - it's a stand-in
for real callers hitting the browser debug flow, useful for capacity/latency testing before
Exotel dispatch exists and after, since the agent's own pipeline doesn't know or care how a
room's participant got there.

WHAT IT MEASURES
Per-stage latency comes entirely from the agent's OWN logs - specifically the "turn_latency"
JSON lines shared/timing.py's TurnLatencyTracker already emits per completed turn (see
agent/main.py). This harness's job is to (a) generate real concurrent conversations so those
turns actually happen, and (b) tail the agent's logs (scripts/load/log_source.py) and correlate
them back to the rooms it created, reporting how percentiles and failure/timeout rates shift as
concurrency climbs - surfacing where Groq/Cartesia rate limits or worker saturation kick in
(see "provider-problem log lines" and "no agent activity" in the report).

WHY THIS DOESN'T IMPORT `livekit.agents` (only `livekit.rtc` + `livekit.api`-free `aiohttp` for
the token fetch): this project's dev machines have hit a case where importing livekit.agents
pulls in `av`, which Windows Smart App Control can block outright (see the codebase's own
Dockerfile.agent/DEPLOY.md notes on this being a dev-only nuisance). Keeping this harness a
plain-`livekit.rtc` consumer - like scripts/loadtest.py already is - means it runs anywhere the
dev venv already works, with no dependency on the agent's own import graph.

USAGE - see DEPLOY.md's "Load testing with synthetic audio" section for the full walkthrough.
Minimal example, against a docker-compose.yml stack running locally:

    python scripts/load/harness.py --concurrency-levels 1,3,5,10

Fixture audio is generated once (via Cartesia) and cached on first run - see fixtures.py.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import string
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from livekit import rtc

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import ensure_fixtures, sample_rate  # noqa: E402
from log_source import LogTail  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.timing import RollingStats  # noqa: E402

_FRAME_MS = 20
_STAGES = ("stt_transcription_delay_s", "llm_ttft_s", "llm_duration_s", "tts_ttfb_s", "total_turn_latency_s")


def _random_suffix(length: int = 6) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _load_wav(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16-bit PCM (fixtures.py always writes 16-bit)")
        return (
            wav_file.readframes(wav_file.getnframes()),
            wav_file.getframerate(),
            wav_file.getnchannels(),
        )


async def _publish_pcm(source: rtc.AudioSource, pcm_bytes: bytes, sample_rate_hz: int, num_channels: int) -> None:
    bytes_per_sample = 2 * num_channels
    samples_per_frame = int(sample_rate_hz * _FRAME_MS / 1000)
    frame_bytes = samples_per_frame * bytes_per_sample
    for offset in range(0, len(pcm_bytes), frame_bytes):
        chunk = pcm_bytes[offset : offset + frame_bytes]
        if len(chunk) < frame_bytes:
            chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
        frame = rtc.AudioFrame(chunk, sample_rate_hz, num_channels, samples_per_frame)
        await source.capture_frame(frame)


async def _publish_silence(source: rtc.AudioSource, duration_s: float, sample_rate_hz: int, num_channels: int) -> None:
    silence = b"\x00" * (int(sample_rate_hz * duration_s) * 2 * num_channels)
    await _publish_pcm(source, silence, sample_rate_hz, num_channels)


@dataclass
class CallResult:
    room_name: str
    ok: bool
    stage: str  # "token" | "connect" | "publish" | "speak" | "ok"
    connect_seconds: float | None = None
    error: str | None = None


async def run_simulated_call(
    *,
    http_session: aiohttp.ClientSession,
    bridge_url: str,
    room_name: str,
    fixture_paths: list[Path],
    turn_pause_s: float,
    final_wait_s: float,
    connect_timeout_s: float,
) -> CallResult:
    """One simulated browser-style caller: fetch a token, join `room_name`, publish a
    microphone-style track, speak each fixture line with a pause after it, then disconnect."""
    identity = f"loadtest-{room_name}"

    try:
        async with http_session.get(
            f"{bridge_url}/token", params={"room": room_name, "identity": identity}
        ) as resp:
            resp.raise_for_status()
            token_data = await resp.json()
    except Exception as exc:
        return CallResult(room_name, False, "token", error=str(exc))

    room = rtc.Room()
    connect_start = time.monotonic()
    try:
        await asyncio.wait_for(
            room.connect(token_data["url"], token_data["token"]), timeout=connect_timeout_s
        )
    except Exception as exc:
        return CallResult(room_name, False, "connect", error=str(exc))
    connect_seconds = time.monotonic() - connect_start

    try:
        sample_rate_hz = sample_rate()
        source = rtc.AudioSource(sample_rate_hz, 1)
        track = rtc.LocalAudioTrack.create_audio_track("loadtest-mic", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        for path in fixture_paths:
            pcm_bytes, wav_rate, wav_channels = _load_wav(path)
            await _publish_pcm(source, pcm_bytes, wav_rate, wav_channels)
            await _publish_silence(source, turn_pause_s, wav_rate, wav_channels)

        await asyncio.sleep(final_wait_s)
    except Exception as exc:
        return CallResult(room_name, False, "speak", connect_seconds=connect_seconds, error=str(exc))
    finally:
        try:
            await asyncio.wait_for(room.disconnect(), timeout=10)
        except Exception:
            pass  # best-effort cleanup; doesn't change this call's recorded outcome

    return CallResult(room_name, True, "ok", connect_seconds=connect_seconds)


@dataclass
class LevelReport:
    level: int
    results: list[CallResult]
    turn_records: list[dict]
    rooms_with_activity: set[str]
    room_names: set[str]
    provider_problem_count: int
    stage_stats: dict[str, dict | None] = field(default_factory=dict)


async def run_concurrency_level(
    *,
    level: int,
    bridge_url: str,
    fixture_paths: list[Path],
    turn_pause_s: float,
    final_wait_s: float,
    connect_timeout_s: float,
    settle_s: float,
    log_tail: LogTail,
) -> LevelReport:
    room_names = {f"loadtest-c{level}-{i}-{_random_suffix()}" for i in range(level)}
    start_monotonic = time.monotonic()

    async with aiohttp.ClientSession() as http_session:
        results = await asyncio.gather(
            *(
                run_simulated_call(
                    http_session=http_session,
                    bridge_url=bridge_url,
                    room_name=room_name,
                    fixture_paths=fixture_paths,
                    turn_pause_s=turn_pause_s,
                    final_wait_s=final_wait_s,
                    connect_timeout_s=connect_timeout_s,
                )
                for room_name in room_names
            )
        )

    await asyncio.sleep(settle_s)  # let trailing agent log lines (esp. turn_latency_summary) land
    end_monotonic = time.monotonic()

    turn_records = log_tail.turn_latency_records_for_rooms(room_names)
    rooms_with_activity = log_tail.rooms_with_any_activity(room_names)
    provider_problem_count = log_tail.provider_problem_count_in_window(start_monotonic, end_monotonic)

    stage_stats: dict[str, dict | None] = {}
    for stage in _STAGES:
        stats = RollingStats(maxlen=len(turn_records) or 1)
        for record in turn_records:
            value = record.get(stage)
            if value is not None:
                stats.add(value)
        stage_stats[stage] = stats.percentiles()

    return LevelReport(
        level=level,
        results=results,
        turn_records=turn_records,
        rooms_with_activity=rooms_with_activity,
        room_names=room_names,
        provider_problem_count=provider_problem_count,
        stage_stats=stage_stats,
    )


def _print_report(report: LevelReport) -> None:
    ok = [r for r in report.results if r.ok]
    failed = [r for r in report.results if not r.ok]
    no_agent_activity = report.room_names - report.rooms_with_activity

    print(f"\n=== concurrency={report.level} ===")
    print(f"connect: {len(ok)}/{len(report.results)} ok")
    if failed:
        by_stage: dict[str, int] = {}
        for r in failed:
            by_stage[r.stage] = by_stage.get(r.stage, 0) + 1
        print(f"failures by stage: {by_stage}")
        example = failed[0]
        print(f"  example ({example.stage}): {example.error}")

    if ok:
        connect_times = sorted(r.connect_seconds for r in ok if r.connect_seconds is not None)
        if connect_times:
            p50 = connect_times[len(connect_times) // 2]
            p95 = connect_times[int(len(connect_times) * 0.95)]
            print(f"connect latency: p50={p50:.2f}s p95={p95:.2f}s max={connect_times[-1]:.2f}s")

    print(
        f"rooms with zero agent log activity (possible dispatch failure / worker saturation): "
        f"{len(no_agent_activity)}/{len(report.room_names)}"
    )
    print(f"turns completed (turn_latency records observed): {len(report.turn_records)}")
    for stage in _STAGES:
        stats = report.stage_stats.get(stage)
        if stats:
            print(f"  {stage}: count={stats['count']} p50={stats['p50']:.2f}s p95={stats['p95']:.2f}s")
        else:
            print(f"  {stage}: no samples")
    print(f"provider-problem log lines (rate-limit/timeout/API errors) in this window: {report.provider_problem_count}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bridge-url", default="http://localhost:8080")
    parser.add_argument(
        "--concurrency-levels", default="1,3,5,10", help="Comma-separated concurrency levels to ramp through"
    )
    parser.add_argument(
        "--turn-pause-seconds", type=float, default=4.0,
        help="Silence held after each simulated customer line, giving the agent time to respond",
    )
    parser.add_argument(
        "--final-wait-seconds", type=float, default=6.0,
        help="Extra wait after the last line before disconnecting, so the agent's last turn/summary gets logged",
    )
    parser.add_argument("--connect-timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--settle-seconds", type=float, default=5.0,
        help="Wait after a level's calls all finish before reading its logs",
    )
    parser.add_argument(
        "--log-source", choices=["docker-compose", "file"], default="docker-compose",
        help="Where to tail the agent's JSON logs from",
    )
    parser.add_argument("--compose-file", default="docker-compose.yml")
    parser.add_argument("--compose-service", default="agent")
    parser.add_argument("--log-file", help="Required when --log-source=file: path the agent's stdout is redirected to")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency_levels.split(",") if x.strip()]

    print("Preparing fixture audio (generated once via Cartesia, then cached)...")
    fixture_paths = await ensure_fixtures()

    if args.log_source == "docker-compose":
        cmd = ["docker", "compose", "-f", args.compose_file, "logs", "-f", "--no-log-prefix", args.compose_service]
    else:
        if not args.log_file:
            parser.error("--log-file is required when --log-source=file")
        cmd = ["tail", "-F", "-n", "0", args.log_file]

    log_tail = LogTail(cmd=cmd)
    await log_tail.start()
    print(f"tailing agent logs via: {' '.join(cmd)}")
    await asyncio.sleep(2)  # let the tail subprocess actually attach before the first level starts

    try:
        reports = []
        for level in levels:
            print(f"\nRunning concurrency level {level}...")
            report = await run_concurrency_level(
                level=level,
                bridge_url=args.bridge_url,
                fixture_paths=fixture_paths,
                turn_pause_s=args.turn_pause_seconds,
                final_wait_s=args.final_wait_seconds,
                connect_timeout_s=args.connect_timeout_seconds,
                settle_s=args.settle_seconds,
                log_tail=log_tail,
            )
            _print_report(report)
            reports.append(report)
    finally:
        await log_tail.stop()

    print("\n=== summary across levels ===")
    for report in reports:
        ok_count = len([r for r in report.results if r.ok])
        total_stat = report.stage_stats.get("total_turn_latency_s")
        p95 = f"{total_stat['p95']:.2f}s" if total_stat else "n/a"
        print(
            f"concurrency={report.level}: connect_ok={ok_count}/{len(report.results)}, "
            f"turns={len(report.turn_records)}, total_turn_latency_p95={p95}, "
            f"provider_problems={report.provider_problem_count}"
        )


if __name__ == "__main__":
    asyncio.run(main())
