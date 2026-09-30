# Deploying the voice agent

This covers running the agent worker as multiple horizontally-scaled replicas in production
(Linux), on top of a shared LiveKit server and Redis. It assumes you've already got the agent
working locally via `docker-compose.yml` (see [README.md](README.md)); this document only
covers what changes for production.

Two things stay separate from the local dev setup on purpose, so nothing here can break it:

| Local dev | Production |
|---|---|
| `Dockerfile.agent` | `Dockerfile.agent.prod` |
| `docker-compose.yml` | `docker-compose.prod.yml` |

`Dockerfile.agent.prod` bakes the same VAD/turn-detector/knowledge-base artifacts as the dev
image (see its comments), and additionally runs as a non-root user and adds a `HEALTHCHECK`.
`docker-compose.prod.yml` is a standalone file (not an override) that builds from it and runs
multiple `agent` replicas plus the bridge and Redis.

## Environment variables

All of these are read by `agent/config.py` (`AgentSettings`) via `.env` / the process
environment - see `.env.example` for the full list with defaults. The ones relevant to scaling
are broken out below; everything else (LiveKit/Deepgram/Cartesia/Groq credentials, LLM tiering,
response guard, etc.) is unchanged by this document.

Required, no default (the agent will fail to start without these):

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `GROQ_API_KEY`, `CARTESIA_VOICE_ID`

Infrastructure, with defaults meant for local dev:

- `REDIS_URL` - **must** point at the same Redis instance for every replica; this is what
  makes call state shareable across replicas and resumable across worker restarts (see
  `agent/call_state.py`). Point this at a managed Redis in production if you're not running the
  `redis` service from `docker-compose.prod.yml` (see that file's comments).

## Scaling knobs

These map onto `livekit.agents.WorkerOptions`/`ServerOptions` fields (see `agent/main.py`'s
`_build_worker_options`) and control how much concurrent call load *one worker process*
(one replica) will take on before deferring to another replica.

| Env var | Default | What it does |
|---|---|---|
| `NUM_IDLE_PROCESSES` | `1` | How many pre-warmed idle processes (VAD model already loaded, see `agent/main.py`'s `prewarm`) this replica keeps on standby so a call never waits on process spin-up. This is a *standing pool size*, not a hard cap - livekit-agents spins up additional processes on demand for extra concurrent calls beyond this, gated by `WORKER_LOAD_THRESHOLD`. livekit-agents' own production default is `min(cpu_count, 4)`; the default here (`1`) matches this project's previous hardcoded behavior, not a production recommendation - see "How many replicas / what values" below. |
| `WORKER_LOAD_THRESHOLD` | unset (framework default: `0.7` under `start`, the mode this project always runs in - see below) | Fraction (0.0-1.0) of measured CPU load past which this replica reports itself unavailable for *new* job dispatch (in-flight calls are unaffected). This is the real per-replica concurrency gate. Leave unset unless you've measured your own CPU-per-call cost and want to tune it. |
| `JOB_MEMORY_LIMIT_MB` | `0` (disabled) | Per-call memory ceiling; that call's process is killed if exceeded. Bounds a single runaway call (e.g. a leak) to itself instead of it degrading every other concurrent call on the same replica. |
| `DRAIN_TIMEOUT` | `3600` (1 hour) | Seconds a replica waits for in-flight calls to finish before exiting on `SIGTERM`/`SIGINT`. Keep this in sync with your orchestrator's own graceful-shutdown grace period (e.g. Docker's `stop_grace_period`, Kubernetes' `terminationGracePeriodSeconds`) - if the orchestrator's timeout is shorter, it will hard-kill replicas with calls still in progress regardless of this setting. |

Not exposed as env vars, but worth knowing about:

- **`start` vs `dev` mode.** `Dockerfile.agent`/`Dockerfile.agent.prod` both run
  `python -m agent.main start`, livekit-agents' production mode (as opposed to `dev`, which adds
  auto-reload and disables load-based throttling entirely). This project has always run in
  `start` mode, including for local Docker dev - so `WORKER_LOAD_THRESHOLD`'s framework default
  (`0.7`) already applies today, it just wasn't previously documented.
- **`job_executor_type`.** Each call runs in its own OS process on Linux (`PROCESS` mode,
  livekit-agents' own Linux default) - one call crashing or leaking memory can't take down
  other concurrent calls on the same replica. Windows dev machines fall back to `THREAD` mode
  automatically (per livekit-agents, to dodge a `BrokenPipeError` some Python-on-Windows builds
  hit spawning processes) - not something this project overrides either way.

### How many replicas / what values

There's no formula that fits every deployment here - it depends on your KB size, expected
concurrent call volume, and available CPU. As a starting point for a first production
deployment: `NUM_IDLE_PROCESSES=2`, leave `WORKER_LOAD_THRESHOLD` unset (i.e. use the framework
default of `0.7`), and pick a replica *count* (not just per-replica idle processes) based on
peak expected concurrent calls - see resource usage below to size CPU/memory limits per
replica, then use the load test to validate before trusting a number.

## How LiveKit job dispatch distributes across replicas

There's no external load balancer or queue in front of the agent replicas - LiveKit's own
server does the job assignment:

1. Every replica connects to the same `LIVEKIT_URL` and **registers itself as a worker**
   (visible in logs as `"registered worker"`, with a unique `id` like `AW_...`).
2. When a new room is created (a caller/browser joins - see `bridge/server.py`'s `/token`
   endpoint and `bridge/static/debug.html`), LiveKit's server picks **one currently-available,
   registered worker** and dispatches that room to it as a job. "Available" means the worker's
   self-reported load hasn't crossed `WORKER_LOAD_THRESHOLD` (see above).
3. That worker spins up (or reuses an idle) process, running `agent/main.py`'s `entrypoint` for
   that room only. Every other replica is completely uninvolved in that call.

This means scaling out is just running more replicas of the same image against the same
`LIVEKIT_URL` - there's no replica identity or partitioning to configure. It also means an
individual replica has no way to "pull" extra work; it can only accept or defer what LiveKit's
server offers it based on its reported load.

Two ways to see this distribution happening on a running deployment:

- Logs: `docker compose -f docker-compose.prod.yml logs agent | grep "registered worker"` shows
  one line per replica with a distinct `id`. Every subsequent `"agent starting"` log line for a
  given call is tagged with that call's `room_name` (not directly with a worker id) - correlate
  a call to a replica via `docker compose ... logs agent-N` per-container, or by watching which
  container's logs a given `room_name` appears in.
- Redis: `agent/call_state.py`'s stored call state includes a `worker_id` field (from
  `ctx.worker_id`) precisely so you can see, after the fact, which replica handled which call -
  see that module's docstring for the full schema.
- The built-in `/worker` HTTP endpoint (see "Health/readiness expectations" below) reports a
  given replica's current `active_jobs` count and `worker_load` in real time.

## Expected resource usage per worker

Take these as a starting point to validate with `docker stats` (or your orchestrator's
per-container metrics) against your own KB size and traffic, not as a guarantee - the honest
answer is "measure it in your environment," and the numbers below exist to give you a
reasonable place to start rather than guessing from zero.

- **Per-replica baseline (idle, no active calls).** Dominated by the Python interpreter plus
  loaded libraries (`livekit-agents`, `onnxruntime` for the VAD and KB embedding models - no
  GPU or PyTorch involved, everything here runs on CPU) and the VAD model loaded by
  `agent/main.py`'s `prewarm`. Expect roughly **150-350 MB RSS** per idle process kept warm
  (i.e. multiply by `NUM_IDLE_PROCESSES` for one replica's idle floor), plus whatever the
  knowledge base's embedding model and ~1200 cached chunk vectors add wherever
  `agent.knowledge_base` gets imported (baked into the image at build time - see
  `Dockerfile.agent.prod` - so this is a one-time load cost per process, not per call).
- **Per active call, marginal.** STT (Deepgram), TTS (Cartesia), and the LLM (Groq) are all
  remote API calls - the replica's own CPU/memory cost per call is mostly audio
  buffering/streaming and VAD/turn-detection inference, not local model inference for the
  "smart" parts of the pipeline. Expect a modest marginal cost on top of the idle baseline per
  concurrent call on a given process - **low tens of MB, not hundreds** - but validate this
  under your own concurrency with the load test below before sizing memory limits tightly.
- **CPU.** VAD + turn-detection run continuously on any call with live audio; this is the main
  per-call CPU cost and is what `WORKER_LOAD_THRESHOLD` is actually measuring (see
  `_DefaultLoadCalc` in livekit-agents: a rolling average of the process's own CPU%).

`docker-compose.prod.yml`'s `deploy.resources` limits (`1.0` CPU / `768M` memory per replica,
`0.5` CPU / `384M` reserved) are a reasonable starting ceiling for a handful of concurrent calls
per replica with `NUM_IDLE_PROCESSES=2` - raise them (and re-run the load test) if you see
`JOB_MEMORY_LIMIT_MB` kills or throttling under real traffic.

## Health/readiness expectations

livekit-agents' own built-in HTTP server (started automatically in `start` mode, which this
project always uses) listens on **port 8081** and exposes:

- `GET /` - liveness/readiness check. Returns `200 OK` when the replica is healthy; `503` if
  its inference subprocess died or it's failed to connect to `LIVEKIT_URL`. This is what
  `Dockerfile.agent.prod`'s `HEALTHCHECK` polls, and what you should point a Kubernetes
  liveness/readiness probe (or equivalent) at.
- `GET /worker` - a JSON status document: `active_jobs` (calls this replica currently holds),
  `worker_load` (the same 0.0-1.0 figure compared against `WORKER_LOAD_THRESHOLD`),
  `agent_name`, `sdk_version`, `protocol_version`. Useful for a dashboard or for manually
  confirming load is spreading across replicas as expected, not something to gate a health
  probe on (a busy-but-healthy replica should still pass liveness).

Practical implications:

- A replica reporting `503` on `/` should be restarted by your orchestrator - it's not
  self-healing. Docker's own `HEALTHCHECK` in `Dockerfile.agent.prod` will mark the container
  `unhealthy`, but **won't restart it by itself** under plain `docker compose` - pair it with
  `restart: unless-stopped`/`on-failure` (already set in `docker-compose.prod.yml`) or a
  container orchestrator that acts on health status.
- A replica at `WORKER_LOAD_THRESHOLD` reports itself unavailable for *new* dispatch but is
  still healthy (`/` still returns `200`) and still finishing its in-flight calls - don't
  conflate "unavailable for new jobs" with "unhealthy."
- On `SIGTERM`, a replica stops accepting new jobs and waits up to `DRAIN_TIMEOUT` for
  in-flight calls to finish before exiting - make sure your orchestrator's own shutdown grace
  period is at least that long, or calls in progress during a deploy/scale-down get cut off.

## Running N replicas

```bash
# build + start (3 replicas by default, per docker-compose.prod.yml's deploy.replicas)
docker compose -f docker-compose.prod.yml up --build -d

# override the replica count for one run
docker compose -f docker-compose.prod.yml up -d --scale agent=8

# watch all replicas register and pick up calls
docker compose -f docker-compose.prod.yml logs -f agent

# scale down again
docker compose -f docker-compose.prod.yml up -d --scale agent=3
```

All replicas share the same `.env` (same `LIVEKIT_URL`, same `REDIS_URL`) - that's the entire
mechanism that makes this work; see "How LiveKit job dispatch distributes across replicas"
above. There's nothing else to configure per-replica (no replica IDs, no partitioning, no
sticky routing to worry about).

## Load testing against the browser debug flow

`scripts/loadtest.py` automates exactly what `bridge/static/debug.html`'s Connect button does -
fetch a token from the bridge's `/token` endpoint, join a fresh LiveKit room, hold the
connection, disconnect - repeated concurrently. This is enough to trigger real agent dispatch
under load (a new room joining is what causes LiveKit to dispatch a job to a worker), without
needing a browser or real microphone audio per simulated call.

```bash
# from the project's own venv (reuses aiohttp + the `livekit` rtc client, both already
# installed as part of livekit-agents - no extra install needed)
python scripts/loadtest.py --bridge-url http://localhost:8080 --total 50 --concurrency 10

# heavier run against a deployed environment
python scripts/loadtest.py --bridge-url http://<host>:8080 --total 200 --concurrency 40 \
    --hold-seconds 30
```

It reports how many simulated calls connected successfully and token+connect latency
percentiles. To confirm the load actually spread across replicas rather than piling onto one,
watch `docker compose -f docker-compose.prod.yml logs -f agent` (or the `/worker` endpoint's
`active_jobs` per replica) during the run.

This script does **not** validate conversation quality, STT/TTS correctness, or per-turn
latency - it only proves rooms get joined and jobs get dispatched under concurrency. For that,
use the full-pipeline harness below.

## Load testing with synthetic audio (`scripts/load/`)

`scripts/load/harness.py` goes further than `scripts/loadtest.py`: it publishes **synthesized
speech** as a browser-style participant (the same join path as `bridge/static/debug.html` and
`scripts/loadtest.py` - fetch a token from the bridge, join a fresh room, publish a
microphone-style track - just with real audio instead of an empty connection), so the agent's
actual STT -> LLM -> TTS pipeline runs for real, and reports the **per-stage latency
shared/timing.py's `TurnLatencyTracker` already logs** (see "Prompt 1" of this project's own
history - the per-turn latency instrumentation in `agent/main.py`), plus failure/timeout rates,
at each concurrency level you ask it to ramp through.

It lives under `scripts/load/` (not `tests/`) deliberately: it drives a live stack (a real
LiveKit server + a running agent + bridge, and burns real Groq/Deepgram/Cartesia usage), which
is a fundamentally different thing from the pytest unit suite's hermetic, mocked tests - it
should never run as part of `pytest tests/` or CI.

### What it measures and how

1. **Synthetic audio, not silence.** `scripts/load/fixtures.py` synthesizes a short, three-line
   Hindi/English code-switching exchange (a greeting reply with product interest, a follow-up
   product question, a close - see `CONVERSATION_LINES`) via Cartesia's own TTS REST endpoint,
   the first time it's needed, then caches the WAVs on disk (`scripts/load/fixtures/*.wav`,
   git-ignored). Real speech is required, not a tone or silence: Deepgram can't produce a
   transcript worth responding to from either, so no turn would ever complete and there'd be
   nothing to measure.
2. **One simulated call = one `rtc.Room`, one published microphone-style track.** Each
   simulated caller fetches a token from the bridge (exactly like a browser would), joins a
   freshly-named room (triggering LiveKit's automatic agent dispatch, same as any real caller),
   publishes an `AudioSource`-backed track tagged `SOURCE_MICROPHONE`, and plays each fixture
   line in turn with a configurable pause after it (`--turn-pause-seconds`) for the agent to
   respond, before disconnecting.
3. **Latency and errors come from the agent's own logs, not the harness.** The harness tails
   the agent's JSON log stream for the whole run (`scripts/load/log_source.py`) and, per
   concurrency level, correlates `event: turn_latency` records back to the room names it
   created - this is exactly the same `stt_transcription_delay_s` / `llm_ttft_s` /
   `llm_duration_s` / `tts_ttfb_s` / `total_turn_latency_s` fields `TurnLatencyTracker` logs
   per turn, aggregated into p50/p95 per stage as concurrency climbs.
4. **Two distinct failure signals, reported separately:**
   - **Connect failures** (token fetch, room join, or publish - a harness-side observation) and
     **"rooms with zero agent log activity"** (the room joined fine, but the agent never showed
     up for it at all in the log window - the sharpest available signal for **worker
     saturation / dispatch exhaustion**: every registered worker was over `WORKER_LOAD_THRESHOLD`
     when the room was created, so LiveKit had nowhere to dispatch the job).
   - **Provider-problem log lines**: a per-level count of ERROR/WARNING log lines matching
     rate-limit/timeout/API-error keywords (`scripts/load/log_source.py`'s
     `_PROVIDER_PROBLEM_KEYWORDS`) - this is how **Groq/Cartesia rate limits** surface, since
     those errors come from livekit-agents' own internal logger and aren't tagged with a
     room_name the way this project's own `turn_latency` records are.

### Running it

```bash
# from the project's own venv - reuses aiohttp + the `livekit` rtc client already required by
# livekit-agents (no extra install), and generates+caches fixture audio via Cartesia on first run
python scripts/load/harness.py --concurrency-levels 1,3,5,10,20
```

By default it tails `docker compose -f docker-compose.yml logs -f --no-log-prefix agent` as the
log source (matching a locally-run `docker compose up` stack); point `--compose-file` at
`docker-compose.prod.yml` to test a multi-replica deployment instead - correlation by
`room_name` works the same regardless of which replica logged a given turn, so this doubles as
a live demonstration of "How LiveKit job dispatch distributes across replicas" (above) under
real conversational load. If the agent isn't running under `docker compose` at all (e.g. a bare
`python -m agent.main dev` locally), use `--log-source file --log-file <path>` against wherever
its stdout is redirected instead.

```bash
# against a multi-replica prod-style deployment
python scripts/load/harness.py --bridge-url http://<host>:8080 \
    --compose-file docker-compose.prod.yml --concurrency-levels 5,10,25,50
```

Tune `--turn-pause-seconds`/`--final-wait-seconds` if your deployment's real per-turn latency
runs longer than the defaults (4s / 6s) - too short and the harness disconnects before the
agent's reply (and this run's `turn_latency` line) actually lands.

### A genuine finding from building this harness

Running it once against a local dev stack surfaced a real, pre-existing issue worth flagging
here rather than treating as a load-test artifact: with `ENABLE_TWO_TIER_LLM` off (the
default), every single turn - even at concurrency 1 - sent Groq's `openai/gpt-oss-120b` a
request exceeding this project's Groq account's on-demand-tier token-per-minute budget (`SYSTEM_PROMPT`
alone is close to it before any conversation history is added), failing with a `413
rate_limit_exceeded` on every turn. This isn't something concurrency caused - it reproduces at
the lowest possible load - but it's exactly the kind of provider-limit issue this harness exists
to surface. If you see `413`/`rate_limit_exceeded` in the "provider-problem log lines" count
even at concurrency 1, check your Groq tier's TPM limit against `SYSTEM_PROMPT`'s token count
before assuming it's a scaling problem.

This harness does **not** validate conversation *quality* (whether the reply was actually a
sensible answer) - only that turns complete, how fast, and how often they don't. Pair it with a
manual `bridge/static/debug.html` session for quality checks.
