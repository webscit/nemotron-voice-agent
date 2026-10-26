# Configure the Intent Engine

The intent engine answers device commands without the large language model (LLM). It matches the final user transcript against sentence templates with [hassil](https://github.com/OHF-Voice/hassil), executes a match directly, and speaks a templated reply. A matched turn skips both LLM calls that a tool-calling turn needs. Every other turn reaches the LLM unchanged.

The engine is available in the Multilingual Assistant example and is off by default. It supports the following three targets:

- **Home Assistant**: executes built-in Home Assistant intents, such as turning a light on, through the Home Assistant REST API.
- **Client tools**: calls a tool that the client declared at connect time, such as the head and volume tools of a Reachy Mini robot.
- **Server features**: executes intents of a feature that runs in the server process. Matrix messaging uses this target to check, read, play, and send text and voice messages through a multi-turn dialog. A server feature can play audio inside its reply, such as a received voice message. Refer to [Use the Voice Dialog](enable-matrix-messaging.md#use-the-voice-dialog).

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

Set only `INTENT_ENGINE_ENABLED=true` and leave `HOME_ASSISTANT_URL` or `HOME_ASSISTANT_TOKEN` unset. The engine then loads only the custom sentences that map to tools the client declared. It does not load the stock Home Assistant sentences and sends no request to Home Assistant. When the client declares none of the mapped tools, and no server feature such as [Matrix messaging](enable-matrix-messaging.md#use-the-voice-dialog) is configured, the engine stays off for that session.

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

A reply from a server feature can ask a question that the next turn answers, such as "Je l'envoie ?" Refer to [Multi-Turn Dialogs](#multi-turn-dialogs).

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
- A configured server feature executes it, for example the `Messaging*` intents of `fr/messaging.yaml` and `en/messaging.yaml` when Matrix messaging is configured. Refer to [Customize the Sentences](enable-matrix-messaging.md#customize-the-sentences).

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

### Multi-Turn Dialogs

A server feature can ask a follow-up question in its reply. Only the next user turn answers the question, and it is matched as follows:

- The custom sentences are matched with the hassil intent context `dialog: <context>`. A sentence block with `requires_context: {dialog: <context>}` matches only while such a question is pending. Messaging uses `dialog: messaging` for its yes, no, and change sentences, so a plain "oui" never matches outside the dialog.
- A question can capture the next turn's own words as a slot, for example a dictated message. The turn is then taken word for word, even when it resembles another command, unless it matches one of the intents that the question accepts, such as "annule."
- A question can let a short answer open a longer turn. When the turn starts with one of these answers, such as "Oui. Lis mes messages.", the leading answer counts and the rest is ignored. A turn with conflicting answers, such as "oui" and "non", matches nothing.
- A question can take the next turn as more of the user's words when the user starts talking before the agent finishes asking it. Messaging uses this to add the rest of a dictated message that turn detection cut off.
- When the next turn matches nothing, the engine asks the question again once, and it stays open for one more turn. After a second miss, the turn goes to the LLM.

A `{message}`-style wildcard slot returns the user's original words with their punctuation, even though hassil matches without punctuation. The following example from `fr/messaging.yaml` matches only during the messaging dialog.

```yaml
language: fr
intents:
  MessagingYes:
    data:
      - sentences:
          - "(oui|ouais|ok|d'accord|vas-y)"
        requires_context:
          dialog: messaging
```

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

- An `intent_engine` event with `handled_by` set to `intent` or `llm`, the intent name, the target, and the reason for a fall-through, such as `no_match`, `blocked_domain`, or `target_unavailable`. A repeated follow-up question is recorded with `handled_by` set to `intent` and the reason `reprompt`. When a learned sentence matched, the event also carries `learned` with the suggestion ID.
- The `intent_match_secs` metric with the matching time.
- The `intent_target_secs` metric with the round-trip time of the target.

The session configuration snapshot records the engine settings without the Home Assistant URL and token.

## Learn Sentences From Conversations

The `dream` job of the [offline post-processing runner](enable-conversation-recording.md#offline-post-processing-dreamer) learns from the turns that the engine missed. After you approve its suggestions, the engine understands the way your users actually phrase requests, and recurring requests skip the LLM. The job proposes the following two kinds of suggestions:

- **New sentences** for an intent that the engine already has, for example another phrasing of `HassTurnOn` or `ReachyMoveHead`.
- **New intents** for a recurring request that the LLM answered with one client tool call whose arguments are the same every time. A new intent maps its sentences to that client tool and adds a short spoken response.

The job does not learn sentences for server features such as Matrix messaging yet.

### Requirements

The job learns from a session only when the following conditions are met:

- Recording is enabled with `MONITORING_ENABLED=true`, and the dreamer runs. Refer to [Record Conversations and Post-Process Them Offline](enable-conversation-recording.md).
- The session ran with `INTENT_ENGINE_ENABLED=true` and had sentences for its language.
- At least one user turn was a miss, that is, the engine found no match and the LLM answered. A turn that matched but went to the LLM on purpose, for example in dry-run mode or for a blocked domain, is not a miss.

The intent part of the job does not need a person attributed to the session. It calls the same LLM as the memory extraction of the `dream` job.

### How Suggestions Are Checked

The job rebuilds the catalog that the session used: the custom sentence files, the stock Home Assistant sentences when the session used Home Assistant, and the learned suggestions that were active. The catalog contains only the intents that the session could run, that is, the declared client tools and the allow-list. The job shows the LLM this catalog, the conversation with the missed turns marked, and the tool calls that the LLM made.

The job then checks every proposal with hassil, the way the live engine matches:

- Each sentence must parse with the lists and expansion rules of the catalog. Unknown lists and rules, including the ones that a referenced expansion rule uses, and wildcard lists, are refused.
- The words of the person in an evidence turn must match the proposal and must not already match the catalog.
- A sentence for a stock Home Assistant intent must have the shape of that intent's stock sentence blocks: the same target slots (`{name}`, `{area}`, or `{floor}`) and the slots that the block fixes, for example `domain: light` for an area command. When a sentence has optional target slots, such as `{name} [dans {area}]`, each variant must match a block.
- For a client tool, the matched slots must produce the same tool call that the LLM made.
- A new intent must use a tool that the client declared, a CamelCase name, and a response template that only uses slots. The name must not clash with an intent of the repository sentence files, `client_tools.yaml`, or the stock Home Assistant sentences, even an intent that was inactive in the session.

The job drops a proposal that no evidence turn matches. A proposed new intent whose literal tool call equals the call of an existing intent becomes a sentence suggestion for that intent. A proposal that reuses the name of a learned intent with the same tool call merges into that intent.

The engine merges a learned sentence for a stock Home Assistant intent as one copy per stock block that matches one of its variants. Each copy inherits the block's fixed slots, `requires_context` (entity domains), `excludes_context`, and response.

The job can only check a sentence that needs Home Assistant names, such as `{name}`, `{area}`, or `{floor}`, when the dreamer can sync Home Assistant. The dreamer reads `HOME_ASSISTANT_URL` and `HOME_ASSISTANT_TOKEN` from the same `.env` as the application, and contacts Home Assistant only when it checks proposals. Without these values, the job keeps the suggestion and marks it unchecked.

The job merges a proposal into an equivalent suggestion from an earlier session, so the evidence accumulates and recurring requests rise to the top of the review list. A suggestion that waits for review also gains the new sentences. A suggestion that is already in use, that is, approved or active through `auto_use_threshold`, gains the evidence but keeps its sentences. New sentences then become a separate unchecked suggestion that waits for review. The job does not propose a suggestion again after a reviewer rejected it.

When you re-run the job on a session, it replaces that session's unreviewed proposals after it parses the new LLM answer. If the LLM call or its answer fails, the earlier proposals stay. If the re-run finds no missed turn, or the session ran without the engine, the job retracts the earlier proposals.

### Configure Learning

The `dream.intents` section of [`src/monitoring/jobs/dreamer.yaml`](../../src/monitoring/jobs/dreamer.yaml) contains the following settings.

| Setting | Default | Description |
| --- | --- | --- |
| `enabled` | `true` | Learns intent sentences in the `dream` job. |
| `auto_use_threshold` | `null` | With `null`, every suggestion waits for approval. When set, a checked sentence suggestion for an existing intent with a confidence at or above this value is active right away. New intents always wait for approval. |
| `max_examples` | `4` | Number of sentence templates per intent that the prompt shows. |

### Review and Apply Suggestions

Open **Review** > **Intents** in the web client. The page lists the suggestions with the most occurrences first. Filter by **To review**, **In use**, **Rejected**, or **All**, and by kind. Each card shows the sentences, a **checked** or **unchecked** badge, the number of occurrences and live uses, and the evidence turns with their audio, text, and LLM tool calls.

You can take the following actions on a suggestion:

- **Approve** a proposed suggestion, or **Keep** one that is already active, to mark it reviewed and active.
- **Correct** its sentences, one per line. For a new intent, you can also correct the name and the response. When you rename a new intent, the sentence suggestions learned for it follow the new name. The corrected copy is active and supersedes the original. A correction whose sentences do not parse, or do not fit a stock sentence shape of a Home Assistant intent, is refused.
- **Reject** it, so that it is not used and not proposed again.
- **Restore** a rejected suggestion, so that it becomes active.

When two reviewers act on the same suggestion at the same time, the second action fails with an error that asks you to reload the suggestion.

Active suggestions apply to new sessions, not to sessions that are already running. At session start, the pipeline loads the active suggestions for the session language from the monitoring database and merges them into the sentence set that defines their intent, custom sentences first. A learned new intent is active only when the client declares its tool. The engine ignores an invalid learned sentence, including a Home Assistant sentence that fits no stock sentence shape, and logs a warning. The startup log line `Intent engine enabled` reports the number of loaded suggestions as `learned=<count>`.

To keep approved suggestions in the repository, click **Export** for a language, for example **Export fr**. The page shows one **Export** button for each language with active suggestions, whatever the filter and page. The download, also available at `GET /api/review/intents/export?language=fr`, contains up to three YAML documents:

- The first document is a sentence file with the learned sentences for intents that the custom sentence files define, and the learned new intents. Add it under `<language>/` in the sentence directory that `INTENT_ENGINE_SENTENCES_DIR` configures, by default `src/examples/shared/intent_engine/sentences/<language>/`. The comments of the export name this directory, relative to the repository root when it is inside the repository.
- The second document lists the learned client-tool intents. Merge it into `client_tools.yaml` in the same directory.
- A third document, present only when needed, lists the learned sentences for stock Home Assistant intents. Do not add them to a custom sentence file, because a custom file that defines a Home Assistant intent replaces its stock sentences. These sentences stay in use from the monitoring database. To keep them permanently, contribute them to the [Home Assistant intents project](https://github.com/home-assistant/intents).

After you deploy an exported new intent, the engine ignores its learned copy with a warning, because a sentence file now defines the intent.

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
| An approved learned sentence does not match | Start a new session. Running sessions keep the sentences they loaded at start. Verify that the log shows no `learned suggestion #<id> ignored` warning. |
| The **Intents** page stays empty | Verify that the session ran with `INTENT_ENGINE_ENABLED=true` and that the dreamer ran its `dream` job. The job only learns from turns that the engine missed. |
| A suggestion is marked unchecked | Read the notes on the card. A suggestion is unchecked when it needs Home Assistant names that the dreamer cannot sync, when only some evidence turns match it, or when a reviewer corrected it. For Home Assistant names, set `HOME_ASSISTANT_URL` and `HOME_ASSISTANT_TOKEN` in `.env`, and re-run the `dream` job for the session. |
