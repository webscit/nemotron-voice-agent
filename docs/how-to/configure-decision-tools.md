# Configure Decision Tools

Decision tools let a small decision model call a client tool after each answer of the large language model (LLM). The first use is body language: a Reachy Mini robot plays an emotion that fits the exchange on every turn, while the agent speaks its answer.

The feature is available in the Multilingual Assistant example and is off by default. It uses the [Ollama decision API](https://docs.ollama.com/capabilities/decision) (System One, `POST /v1/systemone`), which answers multiple-choice questions with a choice, a confidence, and a probability for each option.

## How It Works

The decision model handles each LLM answer as follows:

1. After the LLM finishes an answer, the pipeline sends the user's last message and the answer to the decision model. It asks one choice question for each active decision tool.
2. The decision model picks one value for each tool, or the `none` option when no value fits.
3. When the choice is a value and its confidence reaches the tool's `min_confidence`, the server calls the client tool with that value. The call uses the same RTVI `llm-function-call-in-progress` message as an LLM tool call, so the client runs it like any other tool call.
4. Text-to-speech (TTS) speaks the answer while the decision runs, so the decision does not delay the reply.

The decision call and its result stay out of the chat history. The LLM never sees them.

While the feature is configured, the LLM does not see the decision tools. The following behavior applies:

- Explicit spoken requests, such as "Montre-toi joyeux," still reach the tool through the [intent engine](configure-intent-engine.md#map-intents-to-client-tools) when it is enabled.
- Replies that the intent engine speaks do not trigger a decision.
- When the decision model fails or does not answer within `DECISION_MODEL_TIMEOUT_SECS`, the pipeline skips the decision for that answer and logs a warning. The conversation continues normally.
- An interruption cancels the pending decision.
- At session start, the pipeline sends one warm-up decision so that Ollama loads the model before the first answer. After an Ollama restart, loading the model takes 4 to 7 seconds on Jetson Thor, and up to 14 seconds when the model file is not in the page cache. The warm-up waits up to 120 seconds, and the log shows `Decision tools: decision model ready`.

When `DECISION_MODEL_URL` is unset, the decision tools are ordinary client tools that the LLM can call.

## Before You Begin

- Deploy the Multilingual Assistant example. Refer to the [Multilingual Assistant README](../../src/examples/multilingual/README.md).
- Use a client that declares the tool at connect time, for example the Reachy Mini conversation app, which declares `play_emotion`. A decision tool is active only when the client declares it.
- Provide a decision model through one of the following options:
  - A local [Ollama](https://ollama.com/) v0.35.0 or later. Pull the decision model first, for example `ollama pull d3:2b`.
  - Ollama Cloud at `https://ollama.com/v1/systemone` with an Ollama API key.

## Enable Decision Tools

1. Add the following settings to `.env`. Replace `<ollama-host>` with the address of the machine that runs Ollama.

   ```bash
   DECISION_MODEL_URL=http://<ollama-host>:11434/v1/systemone
   DECISION_MODEL=d3:2b
   ```

   With Docker Compose, the agent runs in a container, so `localhost` does not reach an Ollama server on the host. The application services map `host.docker.internal` to the host, so you can use `http://host.docker.internal:11434/v1/systemone`. Ollama must then listen on an address that the container can reach, for example with `OLLAMA_HOST=0.0.0.0`.

   For Ollama Cloud, set the URL and the API key instead.

   ```bash
   DECISION_MODEL_URL=https://ollama.com/v1/systemone
   DECISION_MODEL_API_KEY=<ollama-api-key>
   ```

2. Re-apply Compose so the container reads the new environment.

   ```bash
   docker compose --profile multilingual-assistant/single-gpu up -d
   ```

3. Connect the client and start a session. The log reports the active decision tools and the model.

   ```bash
   docker compose logs multilingual-assistant-single-gpu | grep "Decision tools"
   ```

   The log shows a line such as `Decision tools: play_emotion (d3:2b)` at session start, and a line such as `Decision tools: play_emotion -> happy (confidence 0.82, 240 ms, called)` after each answer.

The API key stays on the server. The pipeline sends it only as a Bearer token to the decision model, and does not log it or store it with recorded sessions.

## Settings

The following table lists the `.env` settings.

| Setting | Default | Description |
| --- | --- | --- |
| `DECISION_MODEL_URL` | unset | Full URL of the decision endpoint, for example `http://<ollama-host>:11434/v1/systemone`. The feature is on only when this is set. |
| `DECISION_MODEL` | `d3:2b` | Decision model name. Refer to [Choose a Decision Model](#choose-a-decision-model). |
| `DECISION_MODEL_API_KEY` | unset | Optional Bearer token. Required for Ollama Cloud. |
| `DECISION_MODEL_TIMEOUT_SECS` | `3.0` | Time to wait for the decision model before the pipeline skips the decision. The minimum is `0.1`. With `d3:2b` on Jetson Thor, a decision takes about 0.15 seconds on an idle GPU. Larger models take up to about 1.3 seconds, and every model takes longer while the LLM and TTS run. |

## Choose a Decision Model

The default model, `d3:2b`, is the best trade-off between speed and accuracy among the models measured on Jetson Thor. The measurements used the shipped `play_emotion` catalog on 32 exchanges, mostly in French: 23 that call for an emotion and 9 neutral ones. Each model ran alone on an idle GPU through a local Ollama 0.40.11.

| Model | Size | Correct | Latency | Notes |
| --- | --- | --- | --- | --- |
| `d3:2b` (default) | 3.2 GB | 30 of 32 | 0.14 s | Misses only sarcasm and boredom. Confidence is low, so keep `min_confidence` at `0.0`. |
| `d1` | 3.7 GB | 29 of 32 | 0.15 s | Can answer `none` to obvious cases, such as a greeting. |
| `d3:4b` | 5.8 GB | 26 of 32 | 0.27 s | Often picks `helpful` instead of the emotion. |
| `d3:9b` | 10.4 GB | 31 of 32 | 0.41 s | Most accurate `d3` model. |
| `nimble` | 9.5 GB | 32 of 32 | 0.67 s | Most accurate, about five times slower than `d3:2b`. |
| `nimble-v3:9b` | 9.5 GB | 28 of 32 | 0.68 s | Picks `helpful` on some neutral and emotional exchanges. |
| `nimble-v3:9b-q4_K_M` | 5.6 GB | 29 of 32 | 0.77 s | Smaller than `nimble-v3:9b`, but slower. |

The `d3` models need about half as many prompt tokens as `nimble` for the same question (462 compared with 852), which explains most of their speed. When accuracy matters more than latency, use `d3:9b` or `nimble`.

## Configure the Tool Catalog

The catalog [`src/examples/shared/decision_tools/decision_tools.yaml`](../../src/examples/shared/decision_tools/decision_tools.yaml) lists the client tools that the decision model controls. The repository ships one entry, `play_emotion`. It follows the tool schema of the Reachy Mini conversation app and keeps 25 of the 42 emotions that the app declares, the ones that read best from a conversation.

Each entry under `tools` is keyed by the client tool name and contains the following fields.

| Field | Required | Description |
| --- | --- | --- |
| `argument` | Yes | Tool argument that the decision model chooses, for example `emotion`. |
| `instructions` | Yes | Question asked to the decision model. |
| `none` | Yes | Description of the option for "no value fits." When the model picks it, the server does not call the tool. |
| `criteria` | Yes | Map from each value to a short description. Use at most 25 values, because the decision API accepts at most 26 options, including `none`. The value `none` is reserved. |
| `min_confidence` | No | Minimum confidence, from `0` to `1`, to call the tool. The default is `0.0`. |
| `arguments` | No | Fixed arguments sent with every call, in addition to the chosen value. |

The following example shows a shortened `play_emotion` entry.

```yaml
tools:
  play_emotion:
    argument: emotion
    instructions: >-
      A robot reacts to a conversation with a short body-language emotion. Which emotion
      best expresses the user's last message or the robot's answer?
    none: "None of these emotions fits the exchange."
    min_confidence: 0.0
    criteria:
      happy: "Joy or pleasure, good news, something fun"
      thinking: "Pondering a question, reflecting before answering"
      greeting: "Saying hello, someone arriving"
```

The catalog is checked as follows:

- The pipeline skips a malformed entry and logs a warning that starts with `Decision tools: ignoring`.
- At session start, values that are not in the `enum` that the client declared for the argument are dropped, and the log lists them. When no value is left, the tool stays an ordinary LLM tool.
- A tool that the client does not declare is inactive for that session.

Restart the example service after you change the catalog. With Compose, the `src` directory is mounted into the container, so you do not need to rebuild the image.

With `min_confidence` set to `0.0`, the robot plays an emotion on every answer where one fits. Raise the value to play only the clearer choices. A choice below the threshold is logged and recorded as `low_confidence`.

## Review Decisions

When [conversation recording](enable-conversation-recording.md) is enabled, the recorder stores the following data:

- A `decision_tool` event for each decision with the `outcome` (`called`, `none`, `low_confidence`, or `error`), the `tool`, the `choice`, the `confidence`, the `probabilities` for each option, the `latency_ms` of the decision call, and the `state` sent to the decision model. A called decision also records the `tool_call_id` of the call. A failed call records only the `error`, the `latency_ms`, and the `state`.
- A `decision_secs` metric with the latency of each decision request. One request answers every active tool.
- A `function_call_in_progress` event when the server sends a decision call, and a `function_result` event with the client's reply. The `tool_call_id` of a decision call starts with `decision-`.
- A `decision_tools` section in the session configuration snapshot with `enabled`, `model`, `timeout_secs`, the active `tools`, and the `questions` asked to the decision model. The snapshot does not contain the URL or the API key.

### Decision Metrics Per Turn

Each decision call becomes a `tool_calls` row with the trigger `decision`. The call counts for the response latency when the client marks the tool as perceivable. A decision call does not make the turn a `tool` turn, and `n_tool_calls` does not count it, because the decision model runs after every answer.

The decision is not a latency stage, because it runs alongside TTS. The following `turn_metrics` columns describe the decision for the first answer of each turn instead.

| Column | Description |
| --- | --- |
| `decision_secs` | Latency of the decision model |
| `decision_lag_secs` | Time when the decision call was sent, minus the time of the first bot audio. A negative value means that the robot moved before it spoke. |
| `decision_outcome` | `called`, `none`, `low_confidence`, or `error` |
| `decision_choice` | The value that the decision model chose |
| `decision_confidence` | Confidence of the decision that `decision_choice` reports |

The recorder adds these columns to an existing database automatically. To fill them for sessions recorded earlier, run the backfill in [Per-Turn Metrics](enable-conversation-recording.md#per-turn-metrics).

### Decisions in the Review UI

The [review UI](enable-conversation-recording.md#review-in-the-web-ui) shows the decisions in the following places:

- **Metrics** page: the **Decision model** section shows, for each variant, the number of turns, the decision latency p50 and p90, the **Sent after audio** p50, the **Confidence p50**, the called, none, low confidence, and error rates, the top choices, and the agreement of each source with the reference, after the `redecide` job has run. Select **Decision model** in **Compare by** to compare decision models. The tools table lists decision calls with the trigger **decision model**. **Confidence p50** covers the live decision model only. It reads `decision_confidence`, so a variant with only earlier sessions shows `–` until you run the backfill in [Per-Turn Metrics](enable-conversation-recording.md#per-turn-metrics). The **Sessions** page does not need this backfill.
- **Sessions** page, turn timeline: the stage legend below the latency bar lists the decision model with a hollow swatch, for example `Decision model 0.15 s · happy 41%`. The item shows the latency, and then the choice of each tool with its confidence. The decision model is not a segment of the bar, because it runs alongside TTS. Next to the item, the legend shows when the call was sent relative to the first bot audio, and a badge when the outcome is `none`, `low_confidence`, or `error`.
- **Sessions** page, **Decision tools** table: each row shows the source name on the left and the picked choice on the right, followed by the confidence of that source as a percentage, like the ASR transcripts table. The sources are the live model as `live:<model>`, each `redecide` model, and each human reviewer. A choice that differs from the live choice is highlighted in amber. Human reference choices have no confidence. The tool name appears only when the session has several decision tools. Sessions that are already recorded show the confidence too, so you do not need a backfill.
- **Sessions** page header: a session that ran decision tools shows one **agrees** badge per source, with the share of turns that agree with the reference, and a **Re-run redecide** button.
- **Decision reference** activity: a queue of the turns where the live decision model ran, where you set the reference choice. Refer to [Set the Reference Choice](#set-the-reference-choice).

### Set the Reference Choice

A human reference choice is the ground truth of the agreement scores, like the human reference transcript for the [ASR reference](enable-conversation-recording.md#review-in-the-web-ui). The review UI lists the **Decision reference** activity in the sidebar after a recorded session ran decision tools. Its counter shows the number of open turns, that is, turns without a human choice.

The activity lists one item per turn where the live decision model ran, for the first answer of the turn. Filter the turns by **open**, **reviewed**, or **all**, and order them by **Most recent first** or **Disagreement first**. **Disagreement first** lists the turns where the live choice differs from the reference model first, then the turns that still wait for the reference model, and then the turns that agree.

Each card shows the following information:

- The judged exchange, with the **User** and **Assistant** lines that the decision model received.
- The live choice (key `1`), the reference model choice (key `2`), and the choice of each candidate model.
- One chip per option of each tool, `none` included. Hover over a chip to read the option description.

The selected chips start from the human choice of the turn, else from the reference model choice, else from the live choice. Use the following shortcuts to review the turns:

- Press `1` or `2` to take the live or the reference model choice.
- Press `Enter` to save the reference.
- Press `←` for the previous turn, and `→` or `s` for the next turn.

In the **open** list, **Accept N turns where live = reference** saves the live choice for every listed turn where it agrees with the reference model.

You can also set the choice on the **Sessions** page. Under the **Decision tools** table of a turn, click **Set reference choice**, or **Edit reference choice** for a reviewed turn, pick a value for each tool, and click **Save**. The source row of the human choice shows a **reference** badge.

Saving works as follows:

- Set your annotator name first. The review UI stores the choice as a `kind="decision"` annotation with the source `human:<annotator>`.
- The review server rescores the `decision_summary` of the session immediately, so you do not need to run the dreamer. It uses the `name` of the `redecide` reference and candidates in [`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml), so the review server does not need their URL.
- A choice that the session's question does not offer, or an unknown tool, is refused with HTTP `422`, and nothing is saved.
- The choices of a reviewer for a turn merge across saves, tool by tool. The most recent human annotation of a turn is the reference.

The API provides the same operations:

- `GET /api/review/decisions/queue` lists the turns. It accepts `status=open|reviewed|all`, `sort=recent|disagreement`, `limit`, `offset`, and `session_id`.
- `POST /api/review/decisions/references` saves choices with `{"annotator", "items": [{"session_id", "turn_idx", "tool", "choice"}]}`.

## Re-Evaluate Decisions With Another Model

The `redecide` job of the [offline post-processing runner](enable-conversation-recording.md#offline-post-processing-dreamer) asks other decision models the questions of the live session. It then measures how often the live model agrees with a slower, more accurate reference model. Use it to check the live model on real conversations before you change `DECISION_MODEL`.

### How Redecide Works

For each turn where the live decision model ran, the job works as follows:

1. The job replays the first decision of the turn, with the same `state` and the same `questions` from the session snapshot, on the `reference` model and on each enabled `candidates` model. It stores each answer as a `kind="decision"` annotation, with the endpoint `name` as the source.
2. The job picks the reference choice of the turn. A human choice, stored as a `kind="decision"` annotation with the source `human:<name>`, takes precedence over the reference model. Set it in the review UI, as described in [Set the Reference Choice](#set-the-reference-choice).
3. The job compares the live choice, with the source `live:<decision model>`, and each candidate with the reference choice. A turn agrees when every compared tool has the reference choice.
4. The job stores one `kind="decision_summary"` annotation per session and source, with `agreement`, `agreed`, `turns`, and `reference_sources`.

Sessions without decision tools, or recorded before the snapshot included the `questions`, finish immediately.

### Configure Redecide

The `redecide` section of [`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml) defines the endpoints. The following example shows the shipped configuration.

```yaml
redecide:
  reference:
    name: nimble
    model: nimble
  candidates:
    - name: d3-9b
      model: d3:9b
      enabled: false
```

Each endpoint accepts the following fields.

| Field | Default | Description |
| --- | --- | --- |
| `name` | `model` | Annotation source name of the endpoint |
| `model` | Required | Decision model name |
| `url` | `DECISION_MODEL_URL` | Ollama System One URL, for example `http://host.docker.internal:11434/v1/systemone` |
| `api_key_env` | `DECISION_MODEL_API_KEY` | Name of the environment variable that holds the Bearer key |
| `service` | unset | On-demand `dreamer-models` container that the runner starts for the job, as for `reasr` |
| `enabled` | `true` | Set to `false` to keep the endpoint configured without running it |

The default reference is `nimble`, which chose correctly on all 32 exchanges in [Choose a Decision Model](#choose-a-decision-model). The `d3:9b` candidate is disabled. Pull each model on the Ollama server that the endpoint uses, for example `ollama pull nimble`.

The `dreamer` container reads `.env` when it is created, so an endpoint without `url` uses the same `DECISION_MODEL_URL` as the live decision model. The container maps `host.docker.internal` to the host, so it can reach an Ollama server on the host.

After you add `DECISION_MODEL_URL` to `.env`, recreate the container so it reads the new value.

```bash
docker compose --profile dreamer up -d dreamer
```

The job skips an enabled endpoint that has no URL and logs a warning that names the endpoint, `skipping <endpoint>: no URL`. Sessions that already ended do not run again on their own, so re-run the job for them as described in [Run Redecide](#run-redecide).

### Run Redecide

The default `auto_enqueue` list contains `redecide`, so the dreamer runs the job for every ended session. To run it again for a session, click **Re-run redecide** on the session page, or enqueue the job from the command line.

```bash
docker compose exec dreamer uv run python -m monitoring.jobs enqueue redecide <session_id>
```

A re-run asks only the turns and models that have no `decision` annotation yet, for example after you enable a candidate. It then computes the agreement again.

## Troubleshooting

The following table lists common symptoms and resolutions.

| Symptom | Resolution |
| --- | --- |
| The log has no `Decision tools:` line at session start | Verify that `DECISION_MODEL_URL` is set in `.env`, that you re-applied Compose, and that the client declared the tool. |
| The log shows `decision model failed (ConnectError ...)` | The container cannot reach the decision endpoint. With a local Ollama server, use `host.docker.internal` instead of `localhost` and make Ollama listen on `0.0.0.0`. |
| The log shows `decision model failed (ReadTimeout ...)` | The decision model answers slower than `DECISION_MODEL_TIMEOUT_SECS`. Use a faster model, or raise the timeout. If only the first answers time out, check for `decision model warm-up failed` in the log. |
| The log shows `decision model failed (HTTPStatusError ...)` | Verify the model name, that Ollama is v0.35.0 or later, and, for Ollama Cloud, the API key. |
| The robot never plays an emotion, and the log shows `-> none` | The decision model found no fitting value. Review the `criteria` descriptions and the `instructions`. |
| The log shows `low_confidence` | The choice is below `min_confidence`. Lower the threshold in the catalog. |
| The `redecide` job fails with `ConnectError` or `HTTPStatusError` | Verify that the `dreamer` container can reach the endpoint URL, or `DECISION_MODEL_URL` when the endpoint has no `url`, and that the model is pulled on that Ollama server. |
| The `dreamer` log shows the warning `skipping <endpoint>: no URL`, usually with `no reference decision available` | The endpoint has no `url`, and `DECISION_MODEL_URL` is missing from the `dreamer` environment. Add it to `.env`, recreate the container with `docker compose --profile dreamer up -d dreamer`, and then [re-run the job](#run-redecide) for the sessions that already ended. |
| The `dreamer` log shows `no reference decision available` without that warning | The reference model has no answer for the session, and the session has no human choice. Verify that `redecide.reference` is enabled in [`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml), or [set the reference choice](#set-the-reference-choice). |
| The session shows no **agrees** badge | The `redecide` job has not run, the session was recorded before the snapshot included the `questions`, or no reference choice is available. Check the job on the **Dreamer** page. |
| The review UI has no **Decision reference** activity | No ended recorded session ran decision tools, or the session was recorded before the snapshot included the `questions`. |
| Saving a reference choice returns HTTP `422` | The choice is not an option of the session's question, or the tool is unknown for that session. Pick one of the listed chips. |
| The LLM still calls `play_emotion` | `DECISION_MODEL_URL` is unset, or the client's `enum` accepts none of the catalog values. Check the `does not accept` and `left to the LLM` log lines. |
