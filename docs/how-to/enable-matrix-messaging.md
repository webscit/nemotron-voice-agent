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
