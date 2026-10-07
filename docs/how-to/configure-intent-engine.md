# Configure the Intent Engine

The intent engine answers device commands without the large language model (LLM). It matches the final user transcript against sentence templates with [hassil](https://github.com/OHF-Voice/hassil), executes a match directly, and speaks a templated reply. A matched turn skips both LLM calls that a tool-calling turn needs. Every other turn reaches the LLM unchanged.

The engine is available in the Multilingual Assistant example and is off by default. It supports the following two targets:

- **Home Assistant**: executes built-in Home Assistant intents, such as turning a light on, through the Home Assistant REST API.
- **Client tools**: calls a tool that the client declared at connect time, such as the head and volume tools of a Reachy Mini robot.

## Before You Begin

- Deploy the Multilingual Assistant example. Refer to the [Multilingual Assistant README](../../src/examples/multilingual/README.md).
- To control Home Assistant, create a long-lived access token under your user profile, in the **Security** tab. Expose to Assist only the entities that the agent can control. Home Assistant enforces this setting for every request from the engine.
- To use only client tools, skip the Home Assistant settings. Refer to [Use Client Tools Without Home Assistant](#use-client-tools-without-home-assistant).
- Rebuild the application image after you update the repository, because the engine adds the `hassil` and `home-assistant-intents` Python dependencies.

## Enable the Engine

1. Add the following settings to `.env`.

   ```bash
   INTENT_ENGINE_ENABLED=true
   HOME_ASSISTANT_URL=http://homeassistant.local:8123
   HOME_ASSISTANT_TOKEN=<long-lived-access-token>
   INTENT_ENGINE_AREA=Salon
   INTENT_ENGINE_DRY_RUN=true
   ```

2. Re-apply Compose so the container reads the new environment.

   ```bash
   docker compose --profile multilingual-assistant/single-gpu up -d --build
   ```

3. Start a session and say a command. With `INTENT_ENGINE_DRY_RUN=true`, the engine logs the request that it would send, sends nothing, and lets the LLM answer.

   ```bash
   docker compose logs multilingual-assistant-single-gpu | grep "Intent engine"
   ```

4. Set `INTENT_ENGINE_DRY_RUN=false` and re-apply Compose after the logged requests look correct.

The engine stays off when `INTENT_ENGINE_ENABLED` is not `true`. The token stays on the server. The pipeline does not send it to the client, log it, or store it with recorded sessions.

### Use Client Tools Without Home Assistant

Set only `INTENT_ENGINE_ENABLED=true` and leave `HOME_ASSISTANT_URL` or `HOME_ASSISTANT_TOKEN` unset. The engine then loads only the custom sentences that map to tools the client declared. It does not load the stock Home Assistant sentences and sends no request to Home Assistant. When the client declares none of the mapped tools, the engine stays off for that session.

## Settings

The following table lists the `.env` settings.

| Setting | Default | Description |
| --- | --- | --- |
| `INTENT_ENGINE_ENABLED` | `false` | Turns the engine on. |
| `HOME_ASSISTANT_URL` | unset | Base URL of Home Assistant, for example `http://homeassistant.local:8123`. Required for the Home Assistant target. |
| `HOME_ASSISTANT_TOKEN` | unset | Long-lived access token. Required for the Home Assistant target. |
| `INTENT_ENGINE_AREA` | unset | Home Assistant area name where the agent is located. Commands without a name or an area, such as "Allume les lumières," only match when this is set. |
| `INTENT_ENGINE_DRY_RUN` | `false` | Logs the request for each match without sending it. The LLM answers the turn. |
| `INTENT_ENGINE_TIMEOUT_SECS` | `3.0` | Time to wait for Home Assistant before the turn falls through to the LLM. |
| `INTENT_ENGINE_WAKE_WORDS` | unset | Comma-separated names to strip from the start of a sentence before matching. Include the spellings that speech recognition produces for the agent name. |
| `INTENT_ENGINE_ALLOWED_INTENTS` | refer to [Allowed Intents](#allowed-intents) | Comma-separated Home Assistant intent names that replace the default allow-list. |
| `INTENT_ENGINE_BLOCKED_DOMAINS` | `lock,alarm_control_panel` | Comma-separated entity domains that the engine never controls. Set `none` to lift the block. Refer to [Blocked Domains](#blocked-domains). |
| `INTENT_ENGINE_SENTENCES_DIR` | `src/examples/shared/intent_engine/sentences` | Directory with custom sentences and the client-tool mapping. Relative paths resolve from the repository root. |
| `INTENT_ENGINE_ENTITY_CACHE_TTL_SECS` | `60` | Time that synced entity, area, and floor names are reused across sessions. |

## How a Turn Is Handled

The engine sits between the user context aggregator and the LLM and handles each finished user turn as follows:

1. It strips a leading greeting and wake word, for example "Bonjour Reachy," from the transcript.
2. It matches the text against the custom sentences first, and then against the stock Home Assistant sentences for the session language. A session language such as `fr-FR` uses the `fr` sentences. A language without sentences never matches.
3. On a miss, the turn goes to the LLM.
4. On a hit, the engine executes the intent, waits for the result, and then speaks the reply. The reply is stored in the chat history as an assistant message.

The following table describes how the engine handles failures.

| Situation | Behavior |
| --- | --- |
| Home Assistant is unreachable, times out, or returns an HTTP error | The turn falls through to the LLM. |
| Home Assistant reports an error, for example no matching or exposed entity | The engine speaks the packaged error sentence for the session language. It never speaks the error text that Home Assistant returns. |
| The client does not answer a tool call within `CLIENT_FUNCTION_TIMEOUT_SECS` | The LLM receives the timeout as the tool result and answers. |
| The user interrupts while the engine waits | The engine drops the pending reply. |

At session start, the engine loads entity, area, and floor names from Home Assistant with one template request. Entity aliases are not available through this request, so only entity names match.

## Allowed Intents

The engine only executes the Home Assistant intents on its allow-list. The default list contains the following intents:

- `HassTurnOn` and `HassTurnOff`
- `HassLightSet` and `HassSetPosition`
- `HassClimateSetTemperature` and `HassClimateGetTemperature`
- `HassSetVolume`, `HassSetVolumeRelative`, `HassMediaPlayerMute`, and `HassMediaPlayerUnmute`
- `HassMediaPause`, `HassMediaUnpause`, `HassMediaNext`, and `HassMediaPrevious`
- `HassGetState`, `HassGetCurrentTime`, `HassGetCurrentDate`, and `HassGetWeather`

Intents with free-text slots, such as `HassMediaSearchAndPlay`, `HassBroadcast`, and the list intents, are excluded because they match almost any sentence. Timer intents are excluded because they need a satellite device. For `HassGetState`, the engine answers only questions about the state of one named entity. Yes-or-no, "which," and "how many" questions go to the LLM.

## Blocked Domains

The engine never controls locks and alarm control panels, even when Home Assistant exposes them to Assist. A misheard sentence must not unlock a door. The block works as follows:

- Entities in a blocked domain are left out of the names that the engine can match, so "Déverrouille la porte d'entrée" does not match. Another entity that shares its name with a blocked entity is left out too.
- A match that targets a blocked domain as a whole is not sent.
- A `HassTurnOn` or `HassTurnOff` match without an entity name or a domain, such as a custom "turn everything off here" sentence, is sent with a `domain` list that contains every synced domain except the blocked ones. Without this list, Home Assistant also unlocks the locks in the area.

In the first two cases the turn goes to the LLM, which has no Home Assistant access. Change the list with `INTENT_ENGINE_BLOCKED_DOMAINS`.

## Add Custom Sentences

Custom sentence files use the same YAML format as the [Home Assistant intents repository](https://github.com/OHF-Voice/intents) and take priority over the stock sentences. Add one or more files under `<sentences-dir>/<language>/`, for example `src/examples/shared/intent_engine/sentences/fr/`.

A custom intent is active in one of the following cases:

- Its name is on the allow-list and Home Assistant is configured. The engine sends it to Home Assistant. Use this to add phrasings for a built-in intent.
- `client_tools.yaml` maps it to a tool that the client declared for the session.

The following example adds a phrasing for `HassTurnOn`.

```yaml
language: fr
intents:
  HassTurnOn:
    data:
      - sentences:
          - "que la lumière soit"
        slots:
          domain: light
```

Restart the example service after you change a sentence file.

## Map Intents to Client Tools

`client_tools.yaml` in the sentences directory maps intents to client tools. Each entry names the tool and its arguments. An argument is a literal value or a reference to a matched slot. An argument whose slot did not match is left out of the call.

```yaml
intents:
  ReachyMoveHead:
    tool: move_head
    arguments:
      direction: {slot: direction}
  ReachySetVolume:
    tool: volume_control
    arguments:
      device: speaker
      level: {slot: level}
```

An entry can chain a second call to the same tool under `then`. In the second call, an argument can also be computed from the result of the first call with `{result: <key>, add: <value or slot>, min: <n>, max: <n>}`. The following entry reads the current volume and then sets it 10 or 20 points higher or lower, depending on the matched sentence.

```yaml
intents:
  ReachyVolumeRelative:
    tool: volume_control
    arguments:
      device: speaker
    then:
      arguments:
        device: speaker
        level: {result: volume, add: {slot: step}, min: 0, max: 100}
```

An entry with `defer_to_llm: true` and no arguments reserves its sentences for the LLM. Use it to keep a sentence away from a stock Home Assistant intent.

The repository ships French sentences in `sentences/fr/reachy_mini.yaml` for the following tools of the Reachy Mini conversation app: `volume_control`, `move_head`, `dance`, `stop_dance`, `play_emotion`, `go_to_sleep`, and `sweep_look`. The mapping follows the tool schemas of that app. If your client differs, adjust the mapping.

Note the following behavior:

- The engine calls the tool over the same channel that the LLM uses. Each call and its result are stored in the chat history.
- When the client declares `volume_control`, "Mets le volume à 70" and "Parle plus fort" change the robot volume instead of a Home Assistant media player.
- "Parle plus fort" and "Parle moins fort" change the volume by 20 points, and by 10 points with "un peu." The request takes two tool calls.
- The `dance` and `play_emotion` mappings omit the move or the emotion when the sentence names none, and the client then picks one at random. To ask for a dance move by name, use other words so that the LLM handles the request.
- The engine waits for the tool result before it speaks. A tool that answers after `CLIENT_FUNCTION_TIMEOUT_SECS` makes the turn fall through to the LLM.

## Review Engine Decisions

When [conversation recording](enable-conversation-recording.md) is enabled, the engine stores the following data for each user turn:

- An `intent_engine` event with `handled_by` set to `intent` or `llm`, the intent name, the target, and the reason for a fall-through, such as `no_match`, `blocked_domain`, or `target_unavailable`.
- The `intent_match_secs` metric with the matching time.
- The `intent_target_secs` metric with the round-trip time of the target.

The session configuration snapshot records the engine settings without the Home Assistant URL and token.

## Troubleshooting

The following table lists common symptoms and resolutions.

| Symptom | Resolution |
| --- | --- |
| The log shows `HOME_ASSISTANT_URL or HOME_ASSISTANT_TOKEN is unset; client tools only` | Home Assistant commands are inactive. Set both values in `.env` and re-apply Compose. |
| A lock or alarm command reaches the LLM instead of Home Assistant | This is the default. Refer to [Blocked Domains](#blocked-domains). |
| The log shows `Home Assistant sync failed` | Verify that the URL is reachable from the container and that the token is valid. `HTTP 401` indicates a rejected token. The engine retries the sync on later turns. |
| A command with a wake word does not match | Add the spelling from the transcript to `INTENT_ENGINE_WAKE_WORDS`. |
| "Allume les lumières" does not match | Set `INTENT_ENGINE_AREA` to the exact Home Assistant area name. |
| The agent says the packaged error sentence | Home Assistant found no matching entity. Verify that the entity is exposed to Assist and, for area commands, assigned to the area. |
| A Reachy Mini sentence does not match | Verify that the client declared the tool. An intent whose tool is not declared is inactive. |
