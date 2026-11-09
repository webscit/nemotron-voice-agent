# Enable Matrix Messaging

The multilingual assistant can check, read, and send text and voice messages
through [Matrix](https://matrix.org/), the open chat network behind the Element
app. The server logs in to one dedicated Matrix account for the agent. People
who talk with the agent can then ask it whether they have new messages, have
them read aloud or played, and dictate or record replies to the contacts you
configure.

Playing and recording voice messages, and the announcement of new messages,
work only in the [voice dialog](#use-the-voice-dialog) of the intent engine.
The LLM tools get a transcript of each received voice message. Refer to [Limitations](#limitations) before you enable
messaging.

## How It Works

Messaging has a server part and a set of LLM tools:

- **Gateway.** When the server starts, it logs in to the agent's Matrix account
  and syncs in the background. One gateway serves every session. The first
  sync only catches up, so the agent does not import older chat history. The
  first sync of each server process also loads the full state of every room.
- **Inbox.** The gateway stores messages only from configured contacts, and
  only from direct rooms with at most two members. It ignores messages from
  anybody else. It joins rooms automatically when a contact invites the agent
  and leaves invitations from other accounts pending.
- **Sending.** The first time the agent writes to a contact, it creates an
  end-to-end encrypted direct room and invites that contact. Later replies go to the room
  where the contact last wrote, as long as the contact is still in it. Rate-limited requests (`M_LIMIT_EXCEEDED`) are
  retried.
- **Read receipts.** When the agent reads a message aloud, it marks the message
  read in its inbox and sends a read receipt to the sender.
- **Voice messages.** The gateway downloads a received voice message through the
  homeserver's authenticated media API when the agent plays it. The agent plays
  the audio through its own voice output, in place of speech. A voice message
  that you record is encoded as Ogg/Opus, uploaded, and sent as an Element voice
  message, with its duration and waveform.
- **Transcripts.** The agent transcribes received voice messages with the ASR
  server, model, and language of the session. It sends the clip through the
  same Riva-compatible streaming API as the live pipeline, outside the
  pipeline. No setting controls it. Refer to
  [Transcripts of Voice Messages](#transcripts-of-voice-messages).
- **Encryption.** The agent reads and answers end-to-end encrypted rooms, and
  it keeps its device keys on disk. Unencrypted rooms keep working. Refer to
  [Use End-to-End Encryption](#use-end-to-end-encryption).

## Prerequisites

You need the following items:

- A dedicated Matrix account for the agent, for example `@my-agent:matrix.org`.
  Do not use your personal account. The agent reads and sends messages as this
  account.
- An access token for that account and, optionally, its device ID.
- The Matrix IDs of the contacts the agent can message.
- Optionally, session recording with `MONITORING_ENABLED=true`. Without it, the
  agent cannot tell who is talking, so it cannot sign messages or use
  per-person contacts. Refer to
  [Record Conversations](enable-conversation-recording.md).

## Get an Access Token

Request a token for a new device of the agent's account with the Matrix
[login API](https://spec.matrix.org/latest/client-server-api/#post_matrixclientv3login)
(`POST /_matrix/client/v3/login`):

```bash
read -rs "PW?Matrix password: " && echo
curl -s -X POST https://matrix-client.matrix.org/_matrix/client/v3/login \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg pw "$PW" '{type:"m.login.password", identifier:{type:"m.id.user",user:"reachy-fco"}, password:$pw, initial_device_display_name:"nemotron-voice-agent"}')" \
  | jq '{access_token, device_id, expires_in_ms}'
```

The response contains `access_token` and `device_id`. The agent keeps the
encryption keys of that device in `<MATRIX_DATA_DIR>/crypto`, so use the token
for the agent only.

Do not use the access token of an Element session (**Settings > Help & About >
Advanced**) while encryption is on. That token belongs to Element's own device,
which already has encryption keys that the agent does not hold. When the agent
finds keys on the server for its device but none in its `crypto` directory, it
does not replace them: it runs unencrypted and logs `already has encryption
keys`. With `MATRIX_ENCRYPTION=false`, an Element token works, as long as you
do not sign out of that Element session.

With encryption on, the agent uses the device of its token, whatever
`MATRIX_DEVICE_ID` says.

## Configure the Account

Set the following variables in `.env`. Messaging turns on when the three
required variables are set. They are read from the environment by
`src/messaging/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `MATRIX_HOMESERVER` | Not set (required) | Base URL of the homeserver client API, for example `https://matrix-client.matrix.org` |
| `MATRIX_USER_ID` | Not set (required) | The agent's Matrix ID, for example `@my-agent:matrix.org` |
| `MATRIX_ACCESS_TOKEN` | Not set (required) | Access token of the agent's account |
| `MATRIX_DEVICE_ID` | Empty | Device ID that the token was issued for. With encryption on, the server asks the homeserver for the token's device, logs a warning when this value differs, and ignores it. |
| `MATRIX_DATA_DIR` | `data/matrix` | Directory for `inbox.db`, which stores received messages, the room used with each contact, and the sync token, and for `crypto/`, which stores the encryption keys of the agent's device |
| `MATRIX_CONTACTS_FILE` | `<MATRIX_DATA_DIR>/contacts.yaml` | Address book of the contacts that the agent can message |
| `MATRIX_ENCRYPTION` | `true` | End-to-end encryption: read and answer encrypted rooms, and create encrypted rooms. Set `false` to ignore encrypted rooms. |
| `MATRIX_STORE_KEY` | Empty (a built-in passphrase) | Passphrase of the encryption key store. If you change it later, the stored keys become unreadable. The server never logs it. |

Relative paths resolve against the project root. The Compose app services
already mount `./data` read-write at `/app/data`, so the defaults persist
across container restarts without a Compose change.

Treat `MATRIX_ACCESS_TOKEN` and `MATRIX_STORE_KEY` as passwords. Keep them in
`.env` and do not commit them.

## Add Contacts

The agent only reads messages from, and only sends messages to, the contacts in
the contacts file. Create `data/matrix/contacts.yaml`, or the path in
`MATRIX_CONTACTS_FILE`, with entries like the following example:

```yaml
contacts:
  - name: Bob
    matrix_id: "@bob:matrix.org"
    aliases: [Robert]       # other names people use for Bob (optional)
    owner: Alice            # only Alice may message Bob and hear his replies (optional)
  - name: Grandma
    matrix_id: "@grandma:example.org"
```

Each entry uses the following fields:

| Field | Required | Description |
| --- | --- | --- |
| `name` | Yes | The name people say to the agent |
| `matrix_id` | Yes | The contact's Matrix ID, in the form `@user:server` |
| `aliases` | No | Other names for the same contact |
| `owner` | No | A person's name or ID from the **People** page. Only that person can message this contact or hear their messages. |

The server skips entries without a name or a valid Matrix ID and logs a
warning. It reloads the file when it changes, so you can edit contacts while the
server runs.

The agent matches a spoken name against names and aliases without regard to
case or accents. It tries an exact match on a name, alias, or Matrix ID, then a match on a
single word of the name, then a close spelling, so an ASR spelling such as "Jon" can still match
"John". When several contacts match, the agent asks which one you mean.

A contact without `owner` is shared by everyone who talks with the agent. A
contact with `owner` is available only when the agent knows for sure that the
owner is talking. That is, the person is selected in **Who's talking**, or the
voice ID recognizes them at a trusted tier. This is the same rule that the
`remember` and `forget` memory tools use. Refer to
[Identify Speakers by Voice](enable-conversation-recording.md#identify-speakers-by-voice).

## Start a Conversation From Element

A contact can start a chat in one of two ways:

- Wait for the first message from the agent. The contact accepts the invitation
  to the direct room that the agent created.
- Invite the agent's Matrix ID to a new direct chat. The agent joins
  automatically if the contact is in the contacts file.

Element encrypts new direct chats by default, and the agent reads them. With
`MATRIX_ENCRYPTION=false`, contacts must use a direct room with encryption
turned off.

## Use End-to-End Encryption

Encryption is on by default. It needs the `matrix-nio[e2e]` dependency, which
uses the vodozemac library. Its wheels include aarch64, so Jetson needs no
`libolm` build. The dependency changed with this feature, so rebuild the
example image of an existing deployment before you restart it.

Encryption works as follows:

- **Device keys.** The agent's device keys live in
  `<MATRIX_DATA_DIR>/crypto`. The Compose app services mount `./data`, so the
  keys persist across container restarts. The startup log line
  `Matrix: end-to-end encryption on (device <ID>)` names the device.
- **Rooms.** The agent reads and answers encrypted rooms. The rooms it creates
  use Megolm encryption. Existing unencrypted rooms keep working.
- **Voice messages.** A voice message to an encrypted room is uploaded as an
  encrypted attachment, so the homeserver stores only ciphertext. The agent
  decrypts the encrypted voice messages that it receives.
- **Trust on first use.** The agent encrypts messages for every device of a
  contact, verified or not. When a contact adds a device, the server logs
  `Matrix: Bob has a new device (<ID>); messages are encrypted for it too`.
- **Late keys.** When the agent cannot decrypt a message yet, it asks the
  sender's devices for the key and tries again after each sync, for up to
  24 hours. The agent stores and announces the message only after it
  decrypts it. This happens often right after you turn encryption on.
- **One process.** One server process owns the key store, through a file lock.
  With `UVICORN_WORKERS` greater than 1, only the first worker has messaging.
  The other workers log `Matrix messaging unavailable in this process`.

With `MATRIX_ENCRYPTION=false`, the agent ignores encrypted rooms and logs one
warning per room.

Keep the `crypto` directory with the access token. If you lose the directory
but keep the token, the agent finds its device's keys on the server without
the matching store, runs unencrypted, and logs `already has encryption keys`.
Log the agent in again to get a new token and device, and update `.env`.

## Use Messaging in a Conversation

After you restart the server with the new `.env`, the server log shows
`Matrix: syncing as <user id> on <homeserver>`, and each session logs
`Registered messaging tools`. The pipeline then appends the
`messaging_addon` prompt block from `src/examples/multilingual/prompts.yaml`
and adds the following server-side tools:

| Tool | Behavior |
| --- | --- |
| `check_messages` | Counts unread messages per contact, without reading them |
| `read_messages` | Returns up to five messages, oldest first, and marks them read. `contact` limits the result to one contact. `include_read` repeats the latest messages that you already heard. The LLM cannot play a voice message: it tells you to ask the voice dialog to play it. |
| `send_message` | Prepares a draft for `contact` with `text`, up to 2,000 characters. It does not send anything. |
| `confirm_send_message` | Sends the current draft |

The agent sends a message in two steps. It reads the recipient and the text
back and asks you to confirm. The server sends the draft only when
`confirm_send_message` runs in a later user turn than the draft, from the same
person. A draft expires after 5 minutes. To change the text, ask the agent to
change the draft before you confirm it.

The agent signs each message with the name of the confirmed speaker, in the
form `Alice: See you at six`. When the speaker is unknown, the message is sent
without a name.

For example, you can say "Do I have new messages?", "Read me Bob's message,"
or "Tell Grandma that I will call her tonight."

## Use the Voice Dialog

When the [intent engine](configure-intent-engine.md) is on, the engine runs
messaging as a fixed dialog without the LLM. It matches the usual phrasings,
asks one short question at a time, and answers from templated replies. These
turns are faster and more predictable than LLM turns, and a dictated message is
sent exactly as you said it.

### Turn On the Dialog

Set `INTENT_ENGINE_ENABLED=true` in `.env`, in addition to the `MATRIX_*`
variables, and restart the server. The dialog does not need Home Assistant or
client tools. The startup log line `Intent engine enabled` then shows
`server=messaging`.

The dialog ships with French and English sentences. Sessions in other
languages use the LLM tools only.

The LLM tools stay registered. The LLM still answers phrasings that the engine
does not match, and every messaging request when the engine is off. The engine
and the LLM tools share one draft, so you can start a message one way and
confirm it the other way.

When the LLM calls `send_message` or `confirm_send_message` while the engine is
on, the agent speaks the result from the dialog templates, and the LLM does not
reply to it. The agent reads back the draft exactly as stored, and it says
"C'est envoyé" ("Sent") only after the homeserver accepted the message. If the
LLM confirms a draft that does not exist, the agent says that no message is
waiting. After a read-back, the engine answers your yes or no as in the dialog
below.

### Example Exchanges

The following French exchange checks, reads, and answers messages:

```text
Vous : Est-ce que j'ai des messages ?
Agent : Tu as 2 nouveaux messages de Bob. Je te les lis ?
Vous : Oui.
Agent : À 18:05, Bob a écrit : Tu passes ce soir ? À 18:07, Bob a écrit : Il reste de la tarte. Tu veux répondre à Bob ?
Vous : Oui.
Agent : Qu'est-ce que je dis à Bob ?
Vous : J'arrive vers huit heures, garde-moi une part.
Agent : Je dis à Bob : J'arrive vers huit heures, garde-moi une part. Je l'envoie ?
Vous : Oui.
Agent : C'est envoyé à Bob.
```

The following English exchange sends a message in one sentence and changes it
before it is sent:

```text
You: Tell Grandma that I will call her tonight.
Agent: I'll tell Grandma: I will call her tonight. Shall I send it?
You: No, say that I will call her tomorrow instead.
Agent: I'll tell Grandma: I will call her tomorrow. Shall I send it?
You: Yes.
Agent: Sent to Grandma.
```

The dialog works as follows:

- **Check.** "Est-ce que j'ai des messages ?" or "Do I have new messages?"
  counts the unread messages per sender and offers to read them.
- **Read.** The agent reads up to five messages, oldest first, then asks
  "Je continue ?" ("Shall I go on?") when more are waiting. When all messages
  came from one sender, it offers to reply. A voice message plays where it
  appears in the list, after the agent says who sent it and how long it is.
- **Dictate.** After "Qu'est-ce que je dis à Bob ?" ("What should I tell
  Bob?"), the agent takes your next turn as the message, word for word, even
  when it sounds like a command. Say "annule" ("cancel") to stop instead.
- **Confirm.** The agent reads the message back and asks "Je l'envoie ?"
  ("Shall I send it?"). Answer "oui" ("yes") or "je confirme" ("I confirm")
  to send it, "non" ("no") to drop it, "change le message" ("change it") to
  dictate it again, or "non, plutôt que ..." ("say that ... instead") to
  replace the text.
- **Continue.** If you start talking again before the agent finishes the
  read-back, your words are added to the message. The agent then reads the
  whole message back and asks again.
- **One-shot sending.** "Dis à Bob que ...", "Envoie un message à Bob pour lui
  dire que ...", or "Tell Bob that ..." go straight to the confirmation. After
  the agent reads messages from one sender, "Réponds-lui que ..." ("Reply
  saying that ...") answers that sender.

An answer can open a longer turn. For example, "Oui. Lis mes messages." counts
as "oui", and the agent ignores the rest. This happens when speech recognition
joins a short answer to your next sentence. When the turn holds both "oui" and
"non", the agent asks again. A later "change le message" wins over a leading
"non". While the agent waits for a dictated message, every word is part of the
message, including a leading "non".

### Voice Messages

The voice dialog plays the voice messages that contacts send, and records the
ones you send. The following French exchange plays a voice message and answers
it with another one:

```text
Vous : Lis les messages de Bob.
Agent : À 18:05, Bob t'a envoyé un message vocal de 12 secondes : <the message plays> Tu veux répondre à Bob ?
Vous : Réponds-lui par un message vocal.
Agent : Je t'écoute : dis ton message pour Bob.
Vous : <your message, in one turn>
Agent : J'ai enregistré 9 secondes pour Bob. Je l'envoie, ou tu veux le réécouter ?
Vous : Réécoute.
Agent : Voici ton message : <your message plays> Je l'envoie à Bob ?
Vous : Oui.
Agent : C'est envoyé à Bob.
```

The following English exchange plays the latest voice message from a contact,
even one you already heard, and then records a new one:

```text
You: Play the voice message from Bob.
Agent: At 18:05, Bob sent you a 12-second voice message: <the message plays> Do you want to reply to Bob?
You: Send a voice message to Bob.
Agent: I'm listening: say your message for Bob.
You: <your message, in one turn>
Agent: I recorded 9 seconds for Bob. Shall I send it, or do you want to hear it first?
You: Yes.
Agent: Sent to Bob.
```

Voice messages follow these rules:

- **Playing.** "Lis mes messages" ("Read my messages") plays each voice message
  in the list. "Fais-moi écouter le message vocal de Bob" ("Play the voice
  message from Bob") plays the latest voice message from that contact. Without
  a name, it plays the latest voice message from any contact. Speak to
  interrupt a voice message, the same way you interrupt the agent.
- **Recording.** "Envoie un message vocal à Bob" ("Send a voice message to
  Bob") asks you to speak. The agent records your next turn, from just before
  you start speaking until the turn ends. When the recording is shorter than
  half a second, the agent asks you again. It does not send a recording longer
  than 2 minutes.
  Say "annule" ("cancel") to stop instead.
- **Confirming.** Answer "oui" ("yes") to send the recording, "non" ("no") to
  drop it, "réécoute" ("play it") to hear it first, or "recommence" ("start
  over") to record it again.
- **Caption.** The voice message carries what the speech recognition heard as
  its text, signed like a text message, for example `Alice: See you at six`.
  Element shows the voice message player. Clients without voice messages show
  the text.

#### Transcripts of Voice Messages

The agent transcribes the voice messages that it receives, so that the LLM can
work with their content. Transcripts follow these rules:

- **Dialog.** The voice dialog plays the recording and does not read the
  transcript aloud. When the agent cannot play a recording but already has its
  transcript, it says the transcript instead, for example "..., which I can't
  play. It says: See you at six." or "..., que je n'arrive pas à écouter. On y
  entend : À six heures."
- **Follow-up questions.** After the dialog plays voice messages, the agent
  transcribes them in the background. The LLM can then answer questions such as
  "What did Bob say?"
- **LLM tools.** `read_messages` returns a `transcript` field for each voice
  message, with a note that the transcript can contain recognition mistakes
  and that you can say "play the voice message from Bob" to hear it. Without
  the intent engine, the LLM tells you the content of voice messages from
  their transcripts.
- **Cache.** The inbox stores each transcript, so a message is transcribed
  once. When the ASR hears nothing, the agent stores an empty transcript. When
  the transcription fails, the server logs
  `Matrix: cannot transcribe voice message <event ID>` and tries again the next
  time the message is read. A transcription fails when the ASR is unreachable,
  takes more than 20 seconds, or the download or decoding fails.
- **Length.** The agent transcribes at most the first 5 minutes of a voice
  message.

A question lasts one turn. When your answer matches nothing, the agent asks
once more, for example "Je l'envoie à Bob ? Réponds par oui ou par non." If
the next answer does not match either, the question is dropped and the LLM
handles the turn. The recorder stores the repeated question as an
`intent_engine` event with the reason `reprompt`.

The contact names in the sentences come from the contacts file: every `name`
and alias. Accents and case do not matter in either direction: "Helene" in the
file matches "Hélène" from speech recognition, and the other way round. The dialog
follows the same rules as the LLM tools. A contact with an `owner` needs the
person known for sure, only the person who dictated a message can confirm it,
and a guest's message is sent without a name. A message that is being sent
when you interrupt the agent is still sent.

### New Message Announcements

The voice dialog announces unread messages when a session starts, and new
messages when a contact writes while the session runs. It speaks only after
nobody talks. A yes reads the messages, the same way as after a check. The
following English exchange starts when a message arrives during a
conversation:

```text
Agent: You just got a new message from Bob. Shall I read it?
You: Yes.
Agent: At 18:05, Bob wrote: Are you coming tonight? Do you want to reply to Bob?
```

In French, the agent says "Tu viens de recevoir un nouveau message de Bob. Je
te le lis ?" Announcements follow these rules:

- **Waiting messages.** When the session starts, the agent announces the
  unread messages that are already waiting, after the welcome message. When
  the welcome message is off, it announces them as soon as the conversation is
  idle. For example, it says "You have 2 unread messages from Bob. Shall I
  read them?" or "Tu as 2 messages non lus de Bob. Je te les lis ?"
- **Once per session.** The agent announces each message once per session. It
  skips a message that somebody read in the meantime, in this session or
  another one. A message that you leave unread is announced again in the next
  session.
- **Grouped arrivals.** Messages that arrive close together are announced
  together. When older messages from the same senders are still unread, the
  agent also says how many a yes reads, for example "You just got a new
  message from Bob, and have 3 unread."
- **Heard once.** When you speak over an announcement before it ends, the
  agent answers you and says the announcement again at the next pause.
- **Owned contacts.** A message from a contact with an `owner` waits until that
  person is known for sure: picked in **Who's talking**, or recognized by the
  voice ID at a trusted tier. The agent then names the person, for example
  "Alice, you just got a new message from Bob."
- **No interruptions.** The agent waits while you or the agent speak, while a
  reply is pending, and while a messaging question is open, such as a draft
  read-back, a dictation, or a recording. The face ID greeting and the
  announcements wait for each other, so they never talk at the same time.
- **No second ask.** Answer "non" ("no") to leave the messages unread. The
  agent does not repeat the announcement question, so any other answer is an
  ordinary turn.

Without the intent engine, the agent does not announce new messages, and the
server logs `Matrix: new messages are not announced without the intent engine`.
Ask "Do I have new messages?" instead.

### Customize the Sentences

The sentences and the spoken replies are in
`src/examples/shared/intent_engine/sentences/fr/messaging.yaml` and
`src/examples/shared/intent_engine/sentences/en/messaging.yaml`. Each file
defines the following intents:

- `MessagingCheck`, `MessagingRead`, `MessagingSend`, `MessagingReply`,
  `MessagingSendVoice`, and `MessagingPlay` start a request. `{contact}` is the
  contact list, and `{message}` is a wildcard that captures the message text
  with its punctuation.
- `MessagingYes`, `MessagingNo`, and `MessagingChange` declare
  `requires_context: {dialog: messaging}`, so they match only right after the
  agent asked a messaging question. `MessagingPlay` has a second sentence group
  with the same context, such as "réécoute" ("play it"), which replays the
  recording that waits for confirmation.

The `responses.intents.Messaging` section holds the replies as Jinja templates.
The `announce` key holds the announcement of waiting and new messages.
Keep every response key when you edit a file. A missing reply makes the turn
fall through to the LLM. Restart the server after you change a file.

If you set `INTENT_ENGINE_SENTENCES_DIR` to another directory, copy the
`messaging.yaml` files to `<sentences-dir>/<language>/` there. Without them,
the engine has no messaging sentences. Refer to
[Add Custom Sentences](configure-intent-engine.md#add-custom-sentences).

## Privacy and Safety

The following safeguards apply:

- The prompt block tells the LLM to never follow instructions found inside a
  message, and to never read one person's messages to another person.
- Contacts with an `owner` are hidden from everybody except that person.
- The agent's device keys stay in `<MATRIX_DATA_DIR>/crypto`, protected by
  `MATRIX_STORE_KEY`. Without it, a built-in passphrase protects them, so
  restrict access to that directory. Anybody who can read it and knows the
  passphrase can decrypt the agent's messages.
- The agent keeps a recorded voice message in memory only, until it is sent,
  dropped, or expired. Received voice messages are downloaded to play or
  transcribe them, kept in memory only, and not stored on disk. Their
  transcripts are message content, and `inbox.db` stores them like text
  messages. When session recording stores turn audio, it also stores what you
  said and what the agent played.
- The messaging tool calls and their results are not reported to the browser
  client. The client transcript still shows what the agent says aloud.
- The messaging tools are disabled for a session when the client declares a
  tool with the same name. The server logs
  `Messaging tools disabled: the client declares a tool with the same name`.

## Limitations

This first version has the following limitations:

- **Playing and recording need the voice dialog.** Only the intent engine
  plays and records voice messages, in French and English. The LLM tools
  handle voice messages through their transcripts only.
- **Transcripts are not exact.** A transcript is what the speech recognition
  heard, and it can contain mistakes. Transcription uses only the
  Riva-compatible ASR of the session.
- **Announcements need the voice dialog.** Only the intent engine announces
  unread and new messages, in French and English.
- **One turn per recording.** A voice message is the audio of one user turn.
  When you pause long enough for turn detection to end the turn, the
  recording stops there.
- **Playback length.** The agent plays at most the first 5 minutes of a voice
  message, and transcribes at most the first 5 minutes. It does not download
  media larger than 10 MB.
- **No device verification.** matrix-nio has no cross-signing and no
  interactive verification. Element shows the agent's device and messages as
  not verified, for example "Encrypted by a device not verified by its owner."
  When a contact turns on the Element setting that never sends encrypted
  messages to unverified sessions, the agent cannot read that contact's
  messages.
- **Older encrypted history.** The agent generally cannot decrypt messages
  sent before its device existed.
- **One worker.** Only one server process can use encryption. Refer to
  [Use End-to-End Encryption](#use-end-to-end-encryption).
- **One account.** The server uses one Matrix account for every person who talks
  with the agent. Messages show who asked for them only through the name
  prefix.
- **Multilingual assistant only.** Other examples do not register the messaging
  tools.
- **Direct rooms only.** The agent ignores rooms with more than two members.
- **Pauses after the read-back.** When turn detection splits a dictated
  message, the voice dialog adds the second part only if you start it before
  the read-back ends. If you wait until the agent has finished, it takes your
  words as an answer and asks the yes-or-no question again. Say "change le
  message" and dictate the whole message again.
- **LLM claims without a tool call.** The agent speaks the result of the
  messaging tools itself, but the LLM can still claim that it sent a message
  without calling any tool. The prompt forbids it. The voice dialog avoids the
  LLM for the usual phrasings.
- **Learned messaging sentences cover requests only.** The `dream` job learns
  new phrasings to check, read, send, and reply to messages, and keeps contact
  names and message text out of them. It does not learn yes-or-no answers,
  corrections, or voice message phrasings. Add those phrasings to the
  `messaging.yaml` files. Refer to
  [Learn Messaging Sentences](configure-intent-engine.md#learn-messaging-sentences).

## Troubleshooting

The following table lists common symptoms and checks:

| Symptom | What to check |
| --- | --- |
| The agent does not offer to check or send messages | Confirm that `MATRIX_HOMESERVER`, `MATRIX_USER_ID`, and `MATRIX_ACCESS_TOKEN` are all set, then restart the server and look for `Matrix: syncing as` in the logs. |
| Sync fails with an authentication error | The token is invalid. Get a new token and do not sign out of the session that issued it. |
| Messages from a contact never arrive | Confirm that the sender's Matrix ID matches `matrix_id` exactly, and that the room has at most two members. Messages sent before the server first synced are not imported. With `MATRIX_ENCRYPTION=false`, encrypted rooms are ignored. |
| The log shows `cannot decrypt ... yet; asking for its key` | The key did not arrive yet. The agent retries after each sync for up to 24 hours, and the message appears after it is decrypted. Messages sent before the agent's device existed generally stay unreadable. If a contact's Element never sends to unverified sessions, the key never arrives. |
| The log shows `already has encryption keys` and the agent runs unencrypted | The token is an Element session's (refer to [Get an Access Token](#get-an-access-token)), or the `crypto` directory was lost or replaced while the token stayed. Log the agent in again for a new token and device, and update `MATRIX_ACCESS_TOKEN`. |
| The log shows `the homeserver refused the device keys` | The homeserver rejected the keys of the agent's device. Log the agent in again for a new token and device, and update `MATRIX_ACCESS_TOKEN`. |
| The log shows `the access token has no device` | Get a token that belongs to a device, for example from the login API. |
| The log shows `Matrix messaging unavailable in this process` | Another server process holds the key store. Set `UVICORN_WORKERS=1`, or accept that only the first worker has messaging. |
| The log shows `encryption needs matrix-nio[e2e] (vodozemac); running unencrypted` | The image predates the dependency change. Rebuild the example image. |
| Element marks the agent's messages as not verified | Expected. The agent cannot verify its device. Refer to [Limitations](#limitations). |
| The agent does not join a contact's invitation | The inviter is not in the contacts file. Add them, then invite the agent again. |
| The agent says a contact belongs to someone else | The contact has an `owner`, and the agent does not know for sure who is talking. Pick the person in **Who's talking** or let the voice ID recognize them. |
| The agent never sends the message | You must confirm the draft in a later turn, within 5 minutes, as the same person who asked for it. |
| The agent says a message was sent, but it never arrives | Look for a `function_result` event of `confirm_send_message` with status `sent` in the turn. Without one, the LLM answered without sending. Turn on the voice dialog, and add the phrasing you used to the `messaging.yaml` files. |
| The LLM answers messaging requests instead of the voice dialog | Confirm that `INTENT_ENGINE_ENABLED=true` is set, that the session language is French or English, and that the `Intent engine enabled` log line shows `server=messaging`. With a custom `INTENT_ENGINE_SENTENCES_DIR`, copy the `messaging.yaml` files there. |
| A dictated message is cut off | Turn detection split the message into two turns, and you continued after the read-back ended. Refer to [Limitations](#limitations). |
| The agent does not announce new messages | Confirm the intent engine settings in the row about the LLM above. Look for `Matrix: announced new messages from` in the server log. A message from a contact with an `owner` waits until that person is known for sure. The agent waits while anybody speaks or a messaging question is open. |
| The agent says it cannot play a voice message | Look for `Matrix: cannot play voice message` in the server log. The download failed, the file is larger than 10 MB, or FFmpeg cannot decode it. |
| The LLM does not know what a voice message says | Look for `Matrix: cannot transcribe voice message` in the server log. The ASR server was unreachable or took more than 20 seconds, or the download or decoding failed. The agent tries again the next time the message is read. |
| A voice message request goes to the LLM | Playing and recording voice messages need the voice dialog. Confirm the intent engine settings in the row about the LLM above. |
| A recorded voice message is cut off | You paused long enough to end the turn. Record the message again without long pauses. |
