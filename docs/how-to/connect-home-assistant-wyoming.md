# Connect Home Assistant Through Wyoming

Home Assistant uses the [Wyoming protocol](https://github.com/OHF-Voice/wyoming) to reach speech-to-text (STT) and text-to-speech (TTS) servers. The NeMo-Speech.cpp sidecar serves only Riva gRPC, so the `wyoming` Compose profile adds a small bridge, `wyoming-bridge`, that translates between the two. Home Assistant then transcribes and speaks with the same Nemotron ASR and Magpie TTS models as the voice agent, on the same GPU.

The bridge works with the stock `nvcr.io/nvidia/nemo-speech.cpp` image, so the sidecar needs no custom build.

## Before You Begin

- Deploy a `*/single-gpu` recipe. Every NeMo-Speech.cpp sidecar answers on the `nemo-speech` network alias, which is the bridge's default Riva server. Refer to [Jetson Thor](../03-jetson-thor.md) or [Getting Started](../01-getting-started.md).
- Make sure that Home Assistant can reach the agent host on TCP port `10700`, or on the port that you set in `WYOMING_BRIDGE_PORT`.
- Rebuild the application image after you update the repository, because the bridge adds the `wyoming` Python dependency.

## Start the Bridge

1. Optional: set the bridge defaults in `.env`. A request from Home Assistant names its own language, so `WYOMING_BRIDGE_LANGUAGE` applies only when the request names none or names one that the server does not list.

   ```bash
   WYOMING_BRIDGE_LANGUAGE=fr-FR
   WYOMING_BRIDGE_TTS_VOICE=John
   ```

2. Start the recipe with the `wyoming` overlay profile.

   ```bash
   docker compose --profile multilingual-assistant/single-gpu --profile wyoming up -d --build
   ```

3. Check that the bridge found the Riva server. The bridge waits for the sidecar to load its models, then logs the ASR model and TTS voices that it serves.

   ```bash
   docker compose logs wyoming-bridge | grep -E "Riva|listening"
   ```

## Add the Bridge to Home Assistant

1. In Home Assistant, go to **Settings** > **Devices & services** > **Add integration**, and select **Wyoming Protocol**.
2. Enter the agent host name or IP address as **Host**, and `10700` as **Port**.
3. Go to **Settings** > **Voice assistants**, open your assistant, and select `nemo-speech-asr` for **Speech-to-text** and `nemo-speech-tts` for **Text-to-speech**.

The bridge uses the same program names and port as the built-in Wyoming server of earlier custom NeMo-Speech.cpp builds. If you used such a build, the existing integration entry points at the bridge without changes.

## Configuration

The bridge reads the following settings from `.env`:

| Setting | Default | Description |
| --- | --- | --- |
| `WYOMING_BRIDGE_PORT` | `10700` | Host port that Home Assistant connects to. |
| `WYOMING_BRIDGE_RIVA_SERVER` | `nemo-speech:50051` | Riva gRPC server for both ASR and TTS. |
| `WYOMING_BRIDGE_LANGUAGE` | `en-US` | Locale used when a request names no language, or one that the server does not list. |
| `WYOMING_BRIDGE_TTS_VOICE` | Server default | Voice used when a request names none, for example `John`. |
| `WYOMING_BRIDGE_ASR_STOP_HISTORY_MS` | `2000` | Trailing silence, in milliseconds and a multiple of 80, before the ASR ends a segment. `-1` keeps the server setting. |

## How the Bridge Works

- **Speech-to-text**: the bridge resamples the incoming audio to 16 kHz mono and streams it to Riva while Home Assistant is still sending it. After `audio-stop`, it returns the joined final transcript. Home Assistant runs its own voice activity detection and ends each request, so the bridge asks Riva for a 2,000 ms endpointing window. With the sidecar's 320 ms default, Riva can split one command at a short pause and drop the words that follow the split.
- **Language**: a Home Assistant language such as `fr` maps to a locale that the server lists. An exact match wins, then `WYOMING_BRIDGE_LANGUAGE` when it shares the base language, then the first listed locale with that base language.
- **Text-to-speech**: the bridge supports streaming synthesis. As the reply text arrives, it synthesizes each complete sentence and streams the audio back in a single `audio-start` to `audio-stop` response. When Home Assistant selects a voice by name, the protocol sends no language, so the bridge uses `WYOMING_BRIDGE_LANGUAGE` when the voice speaks it, and the first language of the voice otherwise.

## Limitations

- The bridge returns only the final transcript. It does not stream partial transcripts.
- One Riva server serves both ASR and TTS, without TLS or NVIDIA Cloud Functions credentials. The `*/server` recipes run ASR and TTS in separate NIM containers, so they are not supported.
- Sentence splitting uses punctuation. An abbreviation such as `M. Dupont` starts a new sentence, which changes the pacing of the synthesized audio but not the words.

## Troubleshooting

| Symptom | Cause | Resolution |
| --- | --- | --- |
| The log repeats `Waiting for Riva server` | The sidecar is still loading models, or `WYOMING_BRIDGE_RIVA_SERVER` names a host that the bridge cannot reach | Wait for the sidecar to finish loading. Check `docker compose ps` and the server name. |
| Home Assistant cannot add the integration | Port `10700` is blocked, or another container already publishes it | Open the port in the firewall, or set `WYOMING_BRIDGE_PORT` and enter that port in Home Assistant. |
| Words after a pause are missing from the transcript | The endpointing window is too short for the speaker | Increase `WYOMING_BRIDGE_ASR_STOP_HISTORY_MS`, in multiples of 80. |
