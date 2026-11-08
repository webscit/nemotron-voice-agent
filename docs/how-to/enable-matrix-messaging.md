# Enable Matrix Messaging

The multilingual assistant can check, read, and send short text messages
through [Matrix](https://matrix.org/), the open chat network behind the Element
app. The server logs in to one dedicated Matrix account for the agent. People
who talk with the agent can then ask it whether they have new messages, have
them read aloud, and dictate replies to the contacts you configure.

This first version supports text messages only. Refer to
[Limitations](#limitations) before you enable it.

## How It Works

Messaging has a server part and a set of LLM tools:

- **Gateway.** When the server starts, it logs in to the agent's Matrix account
  and syncs in the background. One gateway serves every session. The first
  sync only catches up, so the agent does not import older chat history.
- **Inbox.** The gateway stores messages only from configured contacts, and
  only from direct rooms with at most two members. It ignores messages from
  anybody else. It joins rooms automatically when a contact invites the agent
  and leaves invitations from other accounts pending.
- **Sending.** The first time the agent writes to a contact, it creates an
  unencrypted direct room and invites that contact. Later replies go to the room
  where the contact last wrote, as long as the contact is still in it. Rate-limited requests (`M_LIMIT_EXCEEDED`) are
  retried.
- **Read receipts.** When the agent reads a message aloud, it marks the message
  read in its inbox and sends a read receipt to the sender.

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

You can get an access token from Element:

1. Sign in to Element with the agent's account.
2. Open **Settings > Help & About** and copy the **Access Token** under
   **Advanced**.
3. Open **Settings > Sessions** and note the ID of the current session. This is
   the device ID.

Alternatively, request a token with the Matrix
[login API](https://spec.matrix.org/latest/client-server-api/#post_matrixclientv3login)
(`POST /_matrix/client/v3/login`). The response contains `access_token` and
`device_id`.

Do not sign out of the Element session that issued the token. Signing out
invalidates the token, and the agent can no longer sync.

## Configure the Account

Set the following variables in `.env`. Messaging turns on when the three
required variables are set. They are read from the environment by
`src/messaging/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `MATRIX_HOMESERVER` | Not set (required) | Base URL of the homeserver client API, for example `https://matrix-client.matrix.org` |
| `MATRIX_USER_ID` | Not set (required) | The agent's Matrix ID, for example `@my-agent:matrix.org` |
| `MATRIX_ACCESS_TOKEN` | Not set (required) | Access token of the agent's account |
| `MATRIX_DEVICE_ID` | Empty | Device ID that the token was issued for |
| `MATRIX_DATA_DIR` | `data/matrix` | Directory for `inbox.db`, which stores received messages, the room used with each contact, and the sync token |
| `MATRIX_CONTACTS_FILE` | `<MATRIX_DATA_DIR>/contacts.yaml` | Address book of the contacts that the agent can message |

Relative paths resolve against the project root. The Compose app services
already mount `./data` read-write at `/app/data`, so the defaults persist
across container restarts without a Compose change.

Treat `MATRIX_ACCESS_TOKEN` as a password. Keep it in `.env` and do not commit
it.

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
- Invite the agent's Matrix ID to a new unencrypted direct chat. The agent joins
  automatically if the contact is in the contacts file.

Element encrypts new direct chats by default. The agent cannot read encrypted
rooms yet, so contacts must write in the room that the agent created, or in a
direct room with encryption turned off.

## Use Messaging in a Conversation

After you restart the server with the new `.env`, the server log shows
`Matrix: syncing as <user id> on <homeserver>`, and each session logs
`Registered messaging tools`. The pipeline then appends the
`messaging_addon` prompt block from `src/examples/multilingual/prompts.yaml`
and adds the following server-side tools:

| Tool | Behavior |
| --- | --- |
| `check_messages` | Counts unread messages per contact, without reading them |
| `read_messages` | Returns up to five messages, oldest first, and marks them read. `contact` limits the result to one contact. `include_read` repeats the latest messages that you already heard. |
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
  came from one sender, it offers to reply. Voice messages are announced, not
  played.
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

### Customize the Sentences

The sentences and the spoken replies are in
`src/examples/shared/intent_engine/sentences/fr/messaging.yaml` and
`src/examples/shared/intent_engine/sentences/en/messaging.yaml`. Each file
defines the following intents:

- `MessagingCheck`, `MessagingRead`, `MessagingSend`, and `MessagingReply`
  start a request. `{contact}` is the contact list, and `{message}` is a
  wildcard that captures the message text with its punctuation.
- `MessagingYes`, `MessagingNo`, and `MessagingChange` declare
  `requires_context: {dialog: messaging}`, so they match only right after the
  agent asked a messaging question.

The `responses.intents.Messaging` section holds the replies as Jinja templates.
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
- The messaging tool calls and their results are not reported to the browser
  client. The client transcript still shows what the agent says aloud.
- The messaging tools are disabled for a session when the client declares a
  tool with the same name. The server logs
  `Messaging tools disabled: the client declares a tool with the same name`.

## Limitations

This first version has the following limitations:

- **Text only.** The agent lists received voice messages, but it cannot play
  them yet. It tells you that the message is waiting in Element. It sends only
  text messages.
- **No end-to-end encryption.** The agent cannot read encrypted rooms, and the
  rooms it creates are unencrypted. The homeserver can read these messages.
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
- **No learned messaging sentences.** The `dream` job does not learn new
  messaging sentences yet. Add phrasings to the `messaging.yaml` files.

## Troubleshooting

The following table lists common symptoms and checks:

| Symptom | What to check |
| --- | --- |
| The agent does not offer to check or send messages | Confirm that `MATRIX_HOMESERVER`, `MATRIX_USER_ID`, and `MATRIX_ACCESS_TOKEN` are all set, then restart the server and look for `Matrix: syncing as` in the logs. |
| Sync fails with an authentication error | The token is invalid. Get a new token and do not sign out of the session that issued it. |
| Messages from a contact never arrive | Confirm that the sender's Matrix ID matches `matrix_id` exactly, and that the room is unencrypted and has at most two members. Messages sent before the server first synced are not imported. |
| The agent does not join a contact's invitation | The inviter is not in the contacts file. Add them, then invite the agent again. |
| The agent says a contact belongs to someone else | The contact has an `owner`, and the agent does not know for sure who is talking. Pick the person in **Who's talking** or let the voice ID recognize them. |
| The agent never sends the message | You must confirm the draft in a later turn, within 5 minutes, as the same person who asked for it. |
| The agent says a message was sent, but it never arrives | Look for a `function_result` event of `confirm_send_message` with status `sent` in the turn. Without one, the LLM answered without sending. Turn on the voice dialog, and add the phrasing you used to the `messaging.yaml` files. |
| The LLM answers messaging requests instead of the voice dialog | Confirm that `INTENT_ENGINE_ENABLED=true` is set, that the session language is French or English, and that the `Intent engine enabled` log line shows `server=messaging`. With a custom `INTENT_ENGINE_SENTENCES_DIR`, copy the `messaging.yaml` files there. |
| A dictated message is cut off | Turn detection split the message into two turns, and you continued after the read-back ended. Refer to [Limitations](#limitations). |
