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
- At session start, the pipeline sends one warm-up decision so that Ollama loads the model before the first answer. Loading `nimble` after an Ollama restart takes about 14 seconds on Jetson Thor. The warm-up waits up to 120 seconds, and the log shows `Decision tools: decision model ready`.

When `DECISION_MODEL_URL` is unset, the decision tools are ordinary client tools that the LLM can call.

## Before You Begin

- Deploy the Multilingual Assistant example. Refer to the [Multilingual Assistant README](../../src/examples/multilingual/README.md).
- Use a client that declares the tool at connect time, for example the Reachy Mini conversation app, which declares `play_emotion`. A decision tool is active only when the client declares it.
- Provide a decision model through one of the following options:
  - A local [Ollama](https://ollama.com/) v0.35.0 or later. The decision API supports the `nimble`, `tev1`, `clef`, and `clef-flash` models.
  - Ollama Cloud at `https://ollama.com/v1/systemone` with an Ollama API key.

## Enable Decision Tools

1. Add the following settings to `.env`. Replace `<ollama-host>` with the address of the machine that runs Ollama.

   ```bash
   DECISION_MODEL_URL=http://<ollama-host>:11434/v1/systemone
   DECISION_MODEL=nimble
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

   The log shows a line such as `Decision tools: play_emotion (nimble)` at session start, and a line such as `Decision tools: play_emotion -> happy (confidence 0.82, 240 ms, called)` after each answer.

The API key stays on the server. The pipeline sends it only as a Bearer token to the decision model, and does not log it or store it with recorded sessions.

## Settings

The following table lists the `.env` settings.

| Setting | Default | Description |
| --- | --- | --- |
| `DECISION_MODEL_URL` | unset | Full URL of the decision endpoint, for example `http://<ollama-host>:11434/v1/systemone`. The feature is on only when this is set. |
| `DECISION_MODEL` | `nimble` | Decision model name, for example `nimble`, `tev1`, `clef`, or `clef-flash`. |
| `DECISION_MODEL_API_KEY` | unset | Optional Bearer token. Required for Ollama Cloud. |
| `DECISION_MODEL_TIMEOUT_SECS` | `3.0` | Time to wait for the decision model before the pipeline skips the decision. The minimum is `0.1`. With `nimble` on Jetson Thor, a decision takes about 1.3 seconds on an idle GPU and longer while the LLM and TTS run. |

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

- A `decision_tool` event for each decision with the `outcome` (`called`, `none`, `low_confidence`, or `error`), the `tool`, the `choice`, the `confidence`, the `probabilities` for each option, and the `latency_ms` of the decision call. A failed call records only the `error` and `latency_ms`.
- A `function_result` event with the client's reply to each call. The `tool_call_id` of a decision call starts with `decision-`.
- A `decision_tools` section in the session configuration snapshot with `enabled`, `model`, `timeout_secs`, and the active `tools`. The snapshot does not contain the URL or the API key.

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
| The LLM still calls `play_emotion` | `DECISION_MODEL_URL` is unset, or the client's `enum` accepts none of the catalog values. Check the `does not accept` and `left to the LLM` log lines. |
