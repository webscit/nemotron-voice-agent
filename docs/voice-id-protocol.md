# Voice ID protocol (client ↔ voice agent)

Contract between the Reachy Mini client (`reachy_mini_pipecat`) and the voice agent server
(`nemotron-voice-agent`, `multilingual` pipeline) for per-turn speaker identification.
The client implements it in `audio/voice_id.py` and `pipecat_webrtc.py`; this file is the single copy.

## Roles

- **Client** computes speaker embeddings from the microphone, matches them against a cached
  gallery, and streams a running estimate of who is speaking.
- **Server** owns people ids and the embedding gallery, attributes each user turn to a person,
  swaps per-person memory, enforces per-person policies, and enrolls unknown speakers.

Voice ID is best effort on both sides: if the gallery endpoint is unavailable (monitoring
disabled, older server) the client keeps running and reports every speaker as unknown; if the
client never sends updates the server behaves as it does today (`person_id` from the session
body, or nobody).

## Embeddings

- A vector is L2-normalised `float32`, little-endian, base64-encoded (standard alphabet).
- Vectors are only comparable within one **model key**: an opaque string `<name>@<version>`
  chosen by the client (e.g. `wespeaker-voxblink2-samresnet34-ft@1`). The server stores and
  filters by model key and never compares vectors across keys.
- Scores are cosine similarities in `[-1, 1]`.

## Gallery HTTP API (server)

Mounted on the same origin as the WebRTC offer endpoint.

### `GET /api/voice-id/gallery?model=<model key>`

```json
{
  "model": "wespeaker-voxblink2-samresnet34-ft@1",
  "people": [
    {"person_id": "a1b2", "name": "Alice", "centroid": "<base64>", "count": 12}
  ]
}
```

- Only non-archived people with at least one embedding for that model are listed.
- `centroid` is the L2-normalised mean of the person's stored embeddings.
- `404`/`503` when the people store is unavailable; the client treats that as an empty gallery.

## Client → server: `speaker-update`

Sent as an RTVI `client-message` on the data channel:

```json
{
  "label": "rtvi-ai",
  "type": "client-message",
  "id": "<uuid>",
  "data": {
    "t": "speaker-update",
    "d": {
      "model": "wespeaker-voxblink2-samresnet34-ft@1",
      "utterance_id": 17,
      "person_id": "a1b2",
      "provisional_id": null,
      "score": 0.71,
      "tier": "high",
      "speech_ms": 2300,
      "final": false,
      "embedding": "<base64, only when final>"
    }
  }
}
```

- `utterance_id`: client-side counter, incremented for each detected utterance.
- `person_id`: matched gallery person, or `null`.
- `provisional_id`: `unk-<n>` when the speaker matches nobody in the gallery; stable for the
  session as long as the client keeps clustering that voice together. `null` when `person_id`
  is set or when the client has no opinion yet.
- `tier`: `high` (confident match), `low` (plausible match, do not grant elevated policies),
  `unknown` (no match → `provisional_id` set), `none` (too little speech to say; the server
  should fall back to the previous speaker at reduced confidence).
- `speech_ms`: voiced audio accumulated in this utterance so far.
- `final`: `false` for running estimates (sent roughly every 500 ms once ~1 s of speech is
  available), `true` once when the client's VAD closes the utterance. Final updates carry
  `embedding` whenever enough speech was available to compute one.
- The robot's own voice is rejected on the client and never reported.

The server keeps the most recent update. When a user turn is committed it attributes the turn
using the latest update received since that turn started (falling back to the previous turn's
speaker, at tier `low`, when there is none).

## Server → client: `speaker-enrolled`

Sent as an RTVI `server-message` after the `enroll_speaker` tool succeeds:

```json
{
  "label": "rtvi-ai",
  "type": "server-message",
  "data": {
    "type": "speaker-enrolled",
    "model": "wespeaker-voxblink2-samresnet34-ft@1",
    "provisional_id": "unk-1",
    "person_id": "c3d4",
    "name": "Bob",
    "centroid": "<base64>",
    "count": 4
  }
}
```

The client adds/updates that person in its cached gallery and relabels the provisional cluster.

## Enrollment (`enroll_speaker` tool, server side)

- The server registers an LLM tool `enroll_speaker(name)` executed **on the server** (not
  delegated to the client).
- It binds the buffered final embeddings of the current speaker (the provisional id, or the
  matched person when re-enrolling) to the person named `name`: an existing non-archived person
  with that name (case-insensitive) or a new one.
- The prompt instructs the agent to ask for the person's name and their consent to remember
  their voice before calling the tool.
- If no usable embedding is buffered, the tool returns an error asking the person to say a
  full sentence first.

## Face identification

The client can also recognise faces with its camera. Face identification never replaces the
voice: it confirms it, and reports who is in view when nobody speaks. No image leaves the
client, only embeddings.

### Face embeddings and gallery

Face vectors use the same wire format as voice vectors and their own model key (e.g.
`insightface-buffalo-s-mbf@1`). The client fetches the face gallery with the same endpoint,
`GET /api/voice-id/gallery?model=<face model key>`; the server stores face vectors next to
voice vectors, told apart by model key only.

### Tiers

`tier` in `speaker-update` gains a top value:

- `verified`: the voice matched a person at `high` **and** the face linked to the speaker
  matched the same person. The server treats it as at least `high`; policies may require it.
- A voice matched at `high` with no linked face stays `high`.
- A face alone (speech too short to embed, or a voice that matches nobody) never yields more
  than `low`.
- A linked face that confidently matches a different person than the voice yields `low` with
  the voice's `person_id`.

There is no liveness check: a photo of an enrolled person can produce a face match. This is
why a face alone never exceeds `low` and never grants memory or elevated policies.

### `speaker-update` additions

```json
{
  "tier": "verified",
  "face": {"model": "insightface-buffalo-s-mbf@1", "person_id": "a1b2", "score": 0.62, "link": "doa"},
  "face_embedding": "<base64, only when final and a face was linked>"
}
```

- `face` is `null`/absent when no face could be linked to the speaker. `face.person_id` is
  `null` for an unrecognised face. `link` says how the face was tied to the speaker: `doa`
  (sound direction matched the face's bearing) or `single` (the only face in view, no usable
  sound direction).
- `provisional_id` is shared between modalities: an unknown person has one `unk-<n>` for both
  face and voice.

### Client → server: `presence-update`

Sent as an RTVI `client-message` whenever the set of people in view changes (the client
debounces it), with an empty list when nobody is visible:

```json
{
  "t": "presence-update",
  "d": {
    "model": "insightface-buffalo-s-mbf@1",
    "people": [
      {"person_id": "a1b2", "provisional_id": null, "score": 0.64, "tier": "high"},
      {"person_id": null, "provisional_id": "unk-2", "score": 0.12, "tier": "unknown"}
    ]
  }
}
```

- `tier` here is face-only: `high`, `low` or `unknown`.
- The server keeps the latest presence for the session and makes it visible to the LLM (who is
  in view) on the next turn. When a person recognised at `high` comes into view for the first
  time in a session while the conversation is idle, the server may trigger one turn so the
  agent can greet them by name. Presence alone never loads a person's memories and never
  grants policies.

### Enrollment additions

- `enroll_speaker` also binds the buffered `face_embedding` vectors of the current speaker,
  stored under the face model key. The consent question must cover both voice and face.
- `speaker-enrolled` gains an optional `face` object when face vectors were bound:

```json
{"face": {"model": "insightface-buffalo-s-mbf@1", "centroid": "<base64>", "count": 3}}
```
