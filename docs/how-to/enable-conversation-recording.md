# Record Conversations and Post-Process Them Offline

Session recording stores every conversation of the multilingual assistant: its
timeline, per-service metrics, transcripts, the exact LLM inputs (including
images), and audio. You can then compare pipeline variants, and a background
runner (the *dreamer*) post-processes past sessions while nobody is talking to
the agent.

Recording has two parts, a hot path and a cold path:

- **Hot path (inside the voice pipeline).** An observer and a few event hooks
  only put small records on a queue. A single writer task batches them to
  SQLite and the filesystem in a worker thread, so the pipeline never waits on
  I/O. If storage can't keep up, records are dropped and counted rather than
  adding latency.
- **Cold path (the `dreamer` container).** Jobs run only when no session is
  live. A job checks between steps whether a conversation has started. If one
  has, it stops at that point and resumes from its checkpoint later.

## Enable recording

Set these in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITORING_ENABLED` | `false` | Record sessions |
| `MONITORING_DATA_DIR` | `data` | Root for the database and artifacts (relative to project root, or absolute) |
| `MONITORING_DB_URL` | `sqlite:///<data dir>/voice_agent.db` | SQLAlchemy URL; `postgresql+psycopg://…` works once the driver is installed |
| `RECORD_AUDIO_TURNS` | `true` | Per-turn user WAVs (16 kHz, the raw ASR input) and bot WAVs |
| `RECORD_AUDIO_STEREO` | `true` | Whole-call `conversation.wav`, user on the left channel and bot on the right, written in 10 s chunks |
| `RECORD_VIDEO` | `off` | `off`, `keyframes` (JPEG snapshots of the input video) or `full` (chunked MP4, for development only). Only takes effect once the transport enables video input (`video_in_enabled`); the multilingual transport doesn't enable it yet. Images in the LLM context and images returned by tools are recorded regardless. |
| `RECORD_VIDEO_FPS` | `1.0` | Keyframe rate for `RECORD_VIDEO=keyframes` |

When recording is enabled, it replaces the legacy `ENABLE_*_AUDIO_DUMP` recorder
for the multilingual assistant.

Compose mounts `./data` into the app container. The container runs as root, so
files under `data/` are root-owned on the host. Run the tools below inside a
container (`docker compose exec dreamer …`), or `chown` the directory.

## What is stored

Everything is keyed by the server `session_id`. The same id is passed to
pipecat as the OpenTelemetry `conversation_id`, so Phoenix traces can be joined
to these rows.

| Table | Content |
|-------|---------|
| `sessions` | Start and end, `end_reason`, heartbeat, and a **config snapshot** (models, servers, language, voice, turn detection, client tools, git sha). The snapshot is the A/B key. |
| `turns` | One row per exchange: user and bot text, timestamps, interrupted. Turn 0 is the greeting. |
| `events` | Timeline: VAD and bot speaking, interruptions, final ASR, function calls and results, latency breakdowns, errors |
| `metrics` | Flat numeric samples: `ttfb`, `processing`, `llm_usage.*`, `tts_usage`, `user_bot_latency`, `first_bot_speech_latency`, `turn_duration`, … |
| `llm_calls` | The full LLM input, tools, output, function calls, TTFT, tokens, and `n_images` / `image_pixels` for each call of the conversation LLM |
| `media` | Audio clips, images and video, deduplicated by SHA-256. `source` is `context`, `tool:<name>`, `user_image` or `camera:<track>`. |
| `annotations` | Job outputs and human labels: `(target, source, kind, value)` |
| `jobs` | Post-processing queue with checkpointed `progress` |

Images are never stored inline. Any `data:` image found in the LLM context or in a
tool result is replaced by `{"type": "media_ref", "sha256": …}`, and the file is
written once to
`data/artifacts/sessions/<date>/<session_id>/media/<sha256>.<ext>`. The
`llm_calls` rows can therefore be replayed against another LLM or VLM.

```text
data/
├── voice_agent.db
└── artifacts/sessions/2026-09-26/<session_id>/
    ├── conversation.wav          # stereo, user L / bot R
    ├── turns/001_user_00.wav     # VAD segment 0 of user turn 1 (16 kHz)
    ├── turns/001_bot_00.wav
    ├── media/<sha256>.jpg        # images the LLM saw / tools returned
    └── video/0000.mp4            # RECORD_VIDEO=full only
```

## Compare pipeline variants

```bash
PYTHONPATH=src uv run python -m monitoring.report                      # grouped by llm/asr/tts model + language
PYTHONPATH=src uv run python -m monitoring.report --since-days 7 \
    --group-by llm.model,turn_detection.silero_vad_only --csv report.csv
```

Each group shows:

- user→bot latency (p50/p90) and first-speech latency;
- per-service TTFB, with LLM TTFT reported separately for text-only and image-bearing calls;
- prompt tokens and interruption rate;
- WER per ASR source, once `reasr` has run.

## Offline post-processing (dreamer)

```bash
docker compose --profile multilingual-assistant/single-gpu --profile dreamer up -d
docker compose --profile dreamer-models create   # once: on-demand model containers
docker compose exec dreamer uv run python -m monitoring.jobs status
```

How the runner behaves:

- **Scheduling.** Every ended session gets the jobs listed in
  `auto_enqueue`. A job runs only when no session has a recent heartbeat and
  the last activity is older than `idle_grace_secs`.
- **On-demand models.** A job can require a compose service from the
  `dreamer-models` profile, for example `dreamer-asr-en`. The runner starts it
  through the Docker socket only while idle, and stops it when the queue is
  empty or a conversation starts.
- **Crash recovery.** Sessions orphaned by a crash are closed after
  `orphan_after_secs`.

Configure the runner in
[`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml)
(or point `DREAMER_CONFIG` at your own copy).

### `reasr`: evaluate another ASR on real conversations

For each recorded user turn, `reasr` transcribes the audio with the configured
`reference` and `candidates` endpoints (any Riva-compatible gRPC server). It
then scores the live transcript and each candidate by word error rate (WER):

- **Reference.** The latest **human correction** is used when there is one.
  Otherwise the `reference` model serves as a pseudo ground truth.
- **Output.** Per-turn `kind="wer"` annotations, plus a session-level
  `kind="wer_summary"`: total errors divided by total reference words.

To add human corrections:

```bash
docker compose exec dreamer uv run python -m monitoring.transcripts_csv export /app/data/turns.csv
# fill the "corrected" column, then:
docker compose exec dreamer uv run python -m monitoring.transcripts_csv import /app/data/turns.csv --annotator alice
docker compose exec dreamer uv run python -m monitoring.jobs enqueue reasr <session_id> ...
```

### Add a job

```python
from monitoring.jobs.base import Job, JobContext, register

@register
class SafetyJob(Job):
    kind = "safety"

    def run(self, ctx: JobContext) -> None:
        for turn in ctx.store.rows("turns", ctx.job["target"]):
            if turn["idx"] < ctx.progress.get("next", 0):
                continue
            ctx.check_preempted()            # yields to live sessions
            verdict = ...                    # e.g. call the (idle) vLLM endpoint
            ctx.store.add_annotations([{ "session_id": ctx.job["target"], "target_type": "turn",
                "target_id": f"{ctx.job['target']}:{turn['idx']}", "source": "guard-model",
                "kind": "safety", "value": verdict }])
            ctx.save_progress(next=turn["idx"] + 1)
```

Import the module in `src/monitoring/jobs/__init__.py`, then add its `kind` to
`auto_enqueue`. Planned jobs follow the same pattern:

- `safety`: kid-safety policy checks on text and images;
- `dream`: long-term memory consolidation;
- `rl_export`: JSONL with implicit rewards such as interruptions, rephrasing,
  tool success and latency;
- `revlm`: replay of `llm_calls` against another model.

## Remote storage later

- `SessionStore` uses SQLAlchemy Core. Upserts are dialect-aware for SQLite and
  PostgreSQL, so moving the database only means changing `MONITORING_DB_URL`.
- `ArtifactStore` is a five-method protocol (`put`, `get`, `exists`,
  `local_path`, `commit`). An S3 or MinIO implementation can stage streamed
  files locally and upload them on `commit`.
- The dreamer decides idleness from the database alone, so it can run on
  another machine as soon as both stores are remote.
