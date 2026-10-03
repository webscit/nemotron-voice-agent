# Record Conversations and Post-Process Them Offline

Session recording stores every conversation of the multilingual assistant: its
timeline, per-service metrics, transcripts, the exact LLM inputs (including
images), and audio. You can then compare pipeline variants, and a background
runner (the *dreamer*) post-processes past sessions while nobody is talking to
the agent, for example to extract memories about the people it talks to.

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
| `people` | People the agent talks to: name and an `archived` flag |
| `speaker_assignments` | Who spoke: one row for the whole session (`turn_idx=-1`) or per-turn overrides. `source` is `live:picker`, `live:voice-id:<model key>` (suffixed with `:verified` when the speaker's face confirmed the voice, or with `:low` or `:unknown` for an uncertain attribution), or `human:<name>`. |
| `voice_embeddings` | Voice and face ID enrollment data: speaker and face embeddings per person and model key, written by the `enroll_speaker` tool. |
| `memories` | Facts about a person: text, category, confidence, `status` (`proposed`, `active`, `forgotten` or `superseded`), source, and review fields. Rows are never deleted. |
| `memory_evidence` | The session, turn and quote that support each memory |
| `memory_uses` | Which memories were injected into which live session |

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

## Review in the web UI

Click **Review** in the header of the web client, or open `#/review` directly.
It opens a separate tab, so a call in progress isn't interrupted. The review
API (`/api/review/*`) is only served when `MONITORING_ENABLED=true`.

- **Sessions.** Recorded conversations with their models, review progress and
  live WER. Opening a session shows each turn with:
  - the user audio, every transcript diffed against the reference, and the human reference;
  - images, and the LLM calls (the full stored input on click);
  - the assistant reply, plus the whole-call stereo recording;
  - a speaker selector for the whole session and for each turn. Changing who
    spoke re-queues the `dream` job for that session.
- **People.** Create, rename, and archive the people the agent talks to. Each
  person shows their session and memory counts.
- **Memories.** A queue of memories that nobody has reviewed yet. Each memory
  shows its evidence turns (audio and quote), its confidence, and how often it
  was used ("used in N sessions / N replies"). You can:
  - **approve** it, so that it becomes `active`;
  - **correct** its text or reassign it to another person. This creates a
    `human:<name>` memory that supersedes the original;
  - **forget** it, so that it is never used again.
- **ASR reference.** Every recorded user turn, with the ones where the live
  ASR and the reference model (Voxtral) disagree first.
  - Listen, then take the live (`1`) or reference (`2`) text, or edit it (`e`),
    and save (`Enter`). Saving rescores WER immediately, on the CPU only.
  - Turns where both agree can be accepted in bulk.
  - Your annotator name is kept in the browser and stored as
    `source="human:<name>"`.
- **Metrics.** Compares pipeline variants over a time range (24 h, 7 days,
  30 days, all). A variant is the combination of the config fields you
  compare by: LLM, ASR, TTS, voice, language, turn detection, transport. Each
  variant keeps its color across filter changes. The page shows:
  - user→bot latency: median with a p10–p90 whisker, plus a per-session trend
    (click a point to open the session);
  - time to first byte per service;
  - LLM time to first token, text-only vs with images;
  - ASR WER per transcript source;
  - a table with all the numbers.
- **Dreamer.**
  - **Status:** what the worker is doing and why (idle, waiting for a live
    session to end, running a job, paused), its heartbeat, and job counts.
  - **Controls:** Pause/Resume (a running job stops at its next checkpoint and
    resumes later), the state of the on-demand model containers, and the job
    queue filtered by status, with Cancel for pending jobs and Retry/Re-run for
    the others.
  - **Queue for all sessions:** runs a job kind on every ended session that has
    never had it.
  - **How it works:** the dreamer publishes its state and reads the pause flag
    through the database (`kv` table). The voice server therefore needs no
    Docker access.

## Offline post-processing (dreamer)

```bash
docker compose --profile multilingual-assistant/single-gpu --profile dreamer up -d
docker compose --profile dreamer-models build    # once: Voxtral image
docker compose --profile dreamer-models create   # once: on-demand model containers
docker compose exec dreamer uv run python -m monitoring.jobs status
```

How the runner behaves:

- **Scheduling.** Every ended session gets the jobs listed in
  `auto_enqueue` (default `[reasr, dream]`). A job runs only when no session has a recent heartbeat and
  the last activity is older than `idle_grace_secs`.
- **On-demand models.** A job can require a compose service from the
  `dreamer-models` profile, for example `dreamer-asr-en`. The runner starts it
  through the Docker socket only while idle, and stops it when the queue is
  empty or a conversation starts. After `docker compose down`, run
  `docker compose --profile dreamer-models create --force-recreate` again:
  containers created before keep a reference to the removed network and fail to
  start with `network ... not found`.
- **Crash recovery.** Sessions orphaned by a crash are closed after
  `orphan_after_secs`.

Configure the runner in
[`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml)
(or point `DREAMER_CONFIG` at your own copy).

### `reasr`: evaluate another ASR on real conversations

For each recorded user turn, `reasr` transcribes the audio with the configured
`reference` and `candidates` endpoints. It then scores the live transcript and
each candidate by word error rate (WER):

- **Reference.** The latest **human correction** is used when there is one.
  Otherwise the `reference` model serves as a pseudo ground truth.
- **Output.** Per-turn `kind="wer"` annotations, plus a session-level
  `kind="wer_summary"`: total errors divided by total reference words.

Endpoints can use either protocol:

- `protocol: riva` (default): a Riva-compatible gRPC server, such as Riva, a NIM
  or NeMo-Speech.cpp. Set `server: host:port`.
- `protocol: openai`: an OpenAI-compatible `POST /v1/audio/transcriptions`
  endpoint. Set `base_url` and `model`.

Choose which models run with `enabled` on each endpoint (default `true`). A
disabled endpoint stays configured, but `reasr` skips it and its container is
never started. By default only the Voxtral reference runs: the shipped
`nemotron-speech-streaming-en-0.6b` candidate has `enabled: false`. Set it to
`true` to compare that model on English sessions; it starts `dreamer-asr-en`.

The default reference is **Voxtral Mini 4B**
(`mistralai/Voxtral-Mini-4B-Realtime-2602`). It is served by vLLM in the
on-demand `dreamer-asr-voxtral` container and covers French, English, German,
Spanish and more. Its image (`docker/voxtral.Dockerfile`) adds the audio
decoders missing from the stock vLLM image. Build it once with
`docker compose --profile dreamer-models build dreamer-asr-voxtral`. You can
tune it with `VOXTRAL_MODEL`, `VOXTRAL_GPU_MEMORY_UTILIZATION` (default `0.15`,
since it shares unified memory with the idle live stack) and
`VOXTRAL_MAX_MODEL_LEN`.

To add human corrections:

```bash
docker compose exec dreamer uv run python -m monitoring.transcripts_csv export /app/data/turns.csv
# fill the "corrected" column, then:
docker compose exec dreamer uv run python -m monitoring.transcripts_csv import /app/data/turns.csv --annotator alice
docker compose exec dreamer uv run python -m monitoring.jobs enqueue reasr <session_id> ...
```

### `dream`: extract memories about people

For each ended session attributed to a person, `dream` extracts durable facts
about that person, such as their name, preferences, relationships, routines,
events, and health information they shared. The job works as follows:

- **Skipped sessions.** A session without an attributed person is marked done
  with `skipped` progress. Attribute it on the session page to run the job again.
- **Transcript.** For each user turn, the job uses the human reference, then
  the `reasr` reference model transcript, then the live ASR text.
- **LLM.** By default, the job calls the session's own LLM (the `base_url` and
  `model` stored in the session config), for example the always-on vLLM of a
  `*/single-gpu` recipe. Set `dream.llm` to use another OpenAI-compatible
  endpoint, or `dream.service` to start an on-demand `dreamer-models`
  container.
- **Status.** A memory with a confidence at or above `auto_use_threshold`
  (default `0.8`) becomes `active` and is used in live prompts right away. You
  review it afterwards. A memory below the threshold stays `proposed` until you
  approve it. A memory that would replace a human-reviewed memory is always
  `proposed`.
- **Re-runs.** The job never changes a reviewed memory. Before extracting
  again, it supersedes the unreviewed memories whose only evidence is that
  session.

The `dream` section of
[`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml)
also sets `reasoning` and `temperature`.

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
- `rl_export`: JSONL with implicit rewards such as interruptions, rephrasing,
  tool success and latency;
- `revlm`: replay of `llm_calls` against another model.

## Remember people between conversations

The multilingual assistant can remember facts about the people who talk to it.
This feature requires `MONITORING_ENABLED=true`. A client that supports voice
identification attributes each turn automatically. Refer to
[Identify Speakers by Voice](#identify-speakers-by-voice). With the browser
client, you attribute sessions manually:

1. Create people on the **People** page of the review UI.
2. Before you connect, pick the person in the **Who's talking** menu in the
   header of the live client. The browser remembers your choice and sends it
   as `person_id` in the session config.
3. At session start, the system prompt names the person and lists up to 30 of
   their `active` memories, most confident first. The prompt block is
   `person_memory_addon` in
   [`src/examples/multilingual/prompts.yaml`](../../src/examples/multilingual/prompts.yaml).
   The session is attributed to the person (source `live:picker`), and the
   memories used are logged in `memory_uses`.
4. After the session ends, the dreamer runs `dream` to extract new memories.
   Review them in the **Memories** activity.

An unknown or archived person, or a database error, never blocks a session.
The session then starts without memories.

Memories can contain personal and health information. They stay in the local
monitoring database. The `dream` job sends the session transcript to the LLM it
calls. With a cloud LLM, either as the session LLM or through `dream.llm`, the
transcript leaves your machine.

## Identify Speakers by Voice

A client that computes speaker embeddings, such as the Reachy Mini conversation
app, can tell the multilingual assistant who speaks in each turn. The server
stores the embedding gallery and applies the result. The browser client does
not send speaker updates, so its sessions behave as before. The wire format is
specified in [Voice ID protocol](../voice-id-protocol.md).

Voice identification requires `MONITORING_ENABLED=true`, because people and
their embeddings live in the monitoring database. It works as follows:

1. The client fetches `GET /api/voice-id/gallery?model=<model key>`. The
   response lists one centroid per non-archived person who has embeddings for
   that model key. The endpoint returns `503` when monitoring is disabled.
2. During a session, the client sends `speaker-update` messages with its
   running estimate. The first valid update activates voice identification for
   the session: the `voice_id_addon` prompt block is added to the system
   prompt and the `enroll_speaker` tool is offered to the LLM.
3. When a user turn is committed, the pipeline attributes it by using the
   latest update received for that turn. Without one, it reuses the previous
   speaker at tier `low`.
4. The user message that the LLM receives starts with a tag such as
   `[speaker: Alice]`, `[speaker: Alice (uncertain)]`, or
   `[speaker: unknown guest]`. The recorded transcript is not tagged.
5. The turn is attributed in `speaker_assignments` with the source
   `live:voice-id:<model key>`, so the `dream` job extracts memories for the
   right person.
6. The `person_memory_addon` block follows the speaker. A confident match
   (tier `high`) loads that person's memories. A different or unknown speaker
   removes them. An uncertain turn of the same person keeps them.

To enroll a new voice, the person tells the assistant their name and agrees
that it remembers their voice. The assistant then calls `enroll_speaker`,
which runs on the server. The tool binds the embeddings of the current speaker
to the person with that name, or creates the person, and sends
`speaker-enrolled` to the client. To enroll a person you created in the review
UI, have them give the same name. The tool refuses to rename a voice that is
confidently recognized as another person.

Per-person policies must use `VoiceIdSession.current_speaker()` in
[`src/examples/shared/voice_id.py`](../../src/examples/shared/voice_id.py).
Its `trusted_person_id` is set only for a confident match. For every other
speaker it is `None`, and guest policies apply. A voice match is not
authentication: a recording of a voice can pass it.

Speaker embeddings are biometric data. They stay in the monitoring database
and are never sent to the LLM. To delete the embeddings of a person, call
`DELETE /api/voice-id/people/<person id>/embeddings`. Archiving a person
removes them from the gallery but keeps the embeddings.

Voice identification has the following limits:

- A turn of an unknown speaker has no row in `speaker_assignments`. If you
  also picked a person in **Who's talking**, the review UI and the `dream` job
  attribute that turn to the picked person.
- The assistant does not enroll voices on its own. Embeddings are stored only
  through `enroll_speaker`.
- Only the multilingual assistant handles `speaker-update` messages.

### Confirm Speakers by Face

A client with a camera can also recognize faces. Face identification never
replaces the voice: it confirms it and reports who is in view. No image leaves
the client. The client sends only face embeddings, which are stored next to the
voice embeddings under their own model key and served by the same gallery
endpoint.

- **Tier `verified`.** A `speaker-update` can carry the face that the client
  linked to the speaker. When the voice and that face name the same person, the
  tier is `verified`. The server treats `verified` like `high` for memories and
  trust, and records the turn with the source
  `live:voice-id:<model key>:verified`. A policy that needs both factors must
  check `current_speaker().verified`. The server downgrades a `verified` update
  whose face does not name the same person to `high`.
- **Who is in view.** The client sends `presence-update` messages with the
  people it sees. The first face message adds the `face_id_addon` prompt block.
  The speaker tag then lists the other people in view, for example
  `[speaker: Alice; also in view: Bob, unknown guest]`.
- **Greeting.** When a person recognized at face tier `high` comes into view
  for the first time in a session, and nobody speaks and no reply is pending,
  the server adds `[presence: Alice came into view]` to the context and runs
  the LLM once so that the assistant can greet them by name. A person is
  greeted at most once per session. A person who already spoke, or who appeared
  while the conversation stayed busy for 10 seconds, is not greeted.
- **Enrollment.** `enroll_speaker` also binds the face embeddings that the
  client linked to the current speaker. The assistant asks for consent to
  remember both voice and face. Enrollment still works when no face was seen.

Presence alone never loads the memories of a person, never changes
`current_speaker()`, and never grants a policy. There is no liveness check: a
photo of an enrolled person produces a face match. For this reason a face
without a matching voice is never trusted.

## Remote storage later

- `SessionStore` uses SQLAlchemy Core. Upserts are dialect-aware for SQLite and
  PostgreSQL, so moving the database only means changing `MONITORING_DB_URL`.
- `ArtifactStore` is a five-method protocol (`put`, `get`, `exists`,
  `local_path`, `commit`). An S3 or MinIO implementation can stage streamed
  files locally and upload them on `commit`.
- The dreamer decides idleness from the database alone, so it can run on
  another machine as soon as both stores are remote.
