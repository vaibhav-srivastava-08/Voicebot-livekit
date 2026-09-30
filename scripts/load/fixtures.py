"""Generates (once, then caches) a handful of short spoken-line WAV fixtures used by
scripts/load/harness.py to simulate a "browser-style" customer talking to the agent.

WHY REAL SPEECH, NOT A TONE/NOISE: the harness's whole point is to exercise the actual
STT -> LLM -> TTS pipeline and report the per-turn latency shared/timing.py already logs
(see agent/main.py's TurnLatencyTracker). Deepgram can't produce a transcript worth
responding to from silence or a sine wave, so no turn would ever complete and there would be
nothing to measure. Fixtures are therefore synthesized speech, generated via Cartesia's own
"/tts/bytes" REST endpoint (the same provider/voice the agent itself uses for output) - a
raw `aiohttp` POST, not the `cartesia`/`livekit-plugins-cartesia` package, so this stays
importable in plain Python without pulling in `livekit.agents` (see harness.py's docstring for
why that matters on this project's dev machines).

Cached under scripts/load/fixtures/*.wav next to this file, keyed by a hash of the line text +
voice/model/sample rate, so changing CONVERSATION_LINES or the target voice regenerates only
what changed - mirrors the cache-once pattern agent/knowledge_base.py uses for its embeddings.
"""

from __future__ import annotations

import hashlib
import os
import wave
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv()

_CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"
_CARTESIA_API_VERSION = "2025-04-16"  # matches livekit-plugins-cartesia's own pinned version
_SAMPLE_RATE = 16000
_MODEL_ID = "sonic-3"

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# A short, representative multi-turn exchange in this project's expected Hindi/English
# code-switching style (see prompts.py's LANGUAGE DETECTION section) - a greeting reply with
# product interest, a follow-up product question (exercises search_knowledge_base), and a
# close. Each line is played as one simulated "customer turn" by the harness, with a pause
# after it for the agent to respond - see harness.py's SimulatedCall.
CONVERSATION_LINES: list[str] = [
    "Haan boliye, main hi bol raha hoon. Mujhe Brezza ke baare mein jaankari chahiye.",
    "What's the price and mileage on that?",
    "Okay, that sounds good. Thank you, bye.",
]


def _cache_key(text: str) -> str:
    digest = hashlib.sha1(f"{text}|{_MODEL_ID}|{_SAMPLE_RATE}".encode()).hexdigest()[:12]
    return digest


async def _synthesize(session: aiohttp.ClientSession, api_key: str, voice_id: str, text: str) -> bytes:
    """POST to Cartesia's non-streaming bytes endpoint, returning raw pcm_s16le audio bytes."""
    body = {
        "model_id": _MODEL_ID,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": _SAMPLE_RATE},
        "language": "en",
        "transcript": text,
    }
    async with session.post(
        _CARTESIA_TTS_URL,
        headers={"X-API-Key": api_key, "Cartesia-Version": _CARTESIA_API_VERSION},
        json=body,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        resp.raise_for_status()
        return await resp.read()


def _write_wav(path: Path, pcm_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # pcm_s16le = 16-bit samples
        wav_file.setframerate(_SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)


async def ensure_fixtures(lines: list[str] | None = None) -> list[Path]:
    """Return the WAV file path for each of `lines` (default CONVERSATION_LINES), generating
    and caching via Cartesia any that don't already exist on disk. Raises a clear error if
    Cartesia credentials aren't configured and a fixture is actually missing."""
    lines = lines if lines is not None else CONVERSATION_LINES
    paths = [FIXTURES_DIR / f"line_{i:02d}_{_cache_key(text)}.wav" for i, text in enumerate(lines)]
    missing = [(path, text) for path, text in zip(paths, lines, strict=True) if not path.exists()]
    if not missing:
        return paths

    api_key = os.environ.get("CARTESIA_API_KEY")
    voice_id = os.environ.get("CARTESIA_VOICE_ID")
    if not api_key or not voice_id:
        raise RuntimeError(
            f"{len(missing)} load-test audio fixture(s) are missing and CARTESIA_API_KEY/"
            "CARTESIA_VOICE_ID aren't set (checked the environment and .env) - these fixtures "
            "are synthesized once via Cartesia's TTS API and then cached on disk. Set both env "
            "vars (the same ones the agent itself uses) and re-run, or run "
            "`python scripts/load/fixtures.py` once ahead of time."
        )

    async with aiohttp.ClientSession() as session:
        for path, text in missing:
            print(f"generating fixture: {path.name} ({text[:40]!r}...)")
            pcm_bytes = await _synthesize(session, api_key, voice_id, text)
            _write_wav(path, pcm_bytes)

    return paths


def sample_rate() -> int:
    return _SAMPLE_RATE


if __name__ == "__main__":
    import asyncio

    generated = asyncio.run(ensure_fixtures())
    print(f"\n{len(generated)} fixture(s) ready under {FIXTURES_DIR}:")
    for path in generated:
        print(f"  {path.name}")
