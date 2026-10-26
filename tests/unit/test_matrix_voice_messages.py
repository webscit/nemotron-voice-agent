# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Matrix voice messages: codec, gateway, turn capture, and the dialog that plays and records them."""

import asyncio
import math
import time
from types import SimpleNamespace

import pytest
from nio import DownloadError, MemoryDownloadResponse, RoomSendResponse, UploadResponse
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.tts_service import TTSService
from pipecat.workers.runner import WorkerRunner

from examples.shared.intent_engine.config import IntentEngineConfig
from examples.shared.intent_engine.processor import IntentEngineProcessor, build_intent_engine
from examples.shared.intent_engine.targets import AudioClip
from examples.shared.messaging_intents import MessagingIntents
from examples.shared.messaging_tools import MessagingTools
from examples.shared.voice_capture import UserTurnAudioCapture
from messaging.config import MatrixConfig
from messaging.contacts import Contact
from messaging.gateway import VOICE_FLAG, MatrixError, MatrixGateway
from messaging.inbox import KIND_AUDIO, InboxMessage
from messaging.voice import (
    WAVEFORM_MAX,
    VoiceCodecError,
    decode_audio,
    encode_voice,
    pcm_duration_ms,
    waveform,
)

BOB = "@bob:matrix.org"
ME = "@agent:matrix.org"
CONTACTS = f"""
contacts:
  - name: Bob
    matrix_id: "{BOB}"
"""
ALICE = SimpleNamespace(person={"id": "p-alice", "name": "Alice"})
LLM_REPLY = "Réponse du LLM."
RATE = 16_000
PLAYBACK_RATE = 22_050


def tone(secs: float, rate: int = RATE, amplitude: int = 8000) -> bytes:
    samples = (int(amplitude * math.sin(2 * math.pi * 440 * i / rate)) for i in range(int(secs * rate)))
    return b"".join(s.to_bytes(2, "little", signed=True) for s in samples)


@pytest.fixture(scope="module")
def ogg() -> bytes:
    return encode_voice(tone(2.0), RATE)


def run(coro):
    asyncio.run(coro)


# ------------------------------------------------------------------ codec
def test_voice_round_trip_keeps_the_duration(ogg):
    assert ogg[:4] == b"OggS"
    pcm = decode_audio(ogg, PLAYBACK_RATE)
    assert abs(pcm_duration_ms(pcm, PLAYBACK_RATE) - 2000) < 30
    assert pcm_duration_ms(decode_audio(ogg, PLAYBACK_RATE, max_secs=0.5), PLAYBACK_RATE) == 500


def test_decode_rejects_what_is_not_audio():
    with pytest.raises(VoiceCodecError):
        decode_audio(b"not an audio file", PLAYBACK_RATE)
    with pytest.raises(VoiceCodecError):
        encode_voice(b"", RATE)


def test_waveform_follows_loudness():
    quiet, loud = tone(0.5, amplitude=1000), tone(0.5, amplitude=8000)
    levels = waveform(quiet + loud, points=4)
    assert levels[2:] == [WAVEFORM_MAX, WAVEFORM_MAX]
    assert 0 < levels[0] < WAVEFORM_MAX / 4
    assert waveform(bytes(400), points=4) == [0, 0, 0, 0]
    assert waveform(b"") == []


# ---------------------------------------------------------------- gateway
class MediaClient:
    """The media and send parts of ``nio.AsyncClient``."""

    def __init__(self, media=b""):
        self.rooms = {"!dm:matrix.org": SimpleNamespace(users={ME: None, BOB: None}, invited_users={})}
        self.media = media
        self.uploads = []
        self.sent = []

    async def upload(self, data_provider, content_type, filename=None, filesize=None):
        self.uploads.append((data_provider(0, 0).read(), content_type, filename, filesize))
        return UploadResponse("mxc://matrix.org/voice1"), None

    async def download(self, mxc):
        if mxc != "mxc://matrix.org/bobvoice":
            return DownloadError("not found", "M_NOT_FOUND")
        return MemoryDownloadResponse(self.media, "audio/ogg", "voice.ogg")

    async def room_send(self, room_id, message_type, content):
        self.sent.append((room_id, content))
        return RoomSendResponse("$v1", room_id)

    async def close(self):
        pass


@pytest.fixture
def config(tmp_path):
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(CONTACTS, encoding="utf-8")
    return MatrixConfig("https://matrix.org", ME, "t", "", tmp_path, contacts)


def test_send_voice_uploads_and_flags_a_voice_message(config, ogg):
    client = MediaClient()
    gateway = MatrixGateway(config, client=client)
    gateway.synced.set()
    gateway.inbox.set_room(BOB, "!dm:matrix.org")
    bob = gateway.contacts.by_matrix_id(BOB)

    run(gateway.send_voice(bob, ogg, duration_ms=2000, waveform=[1, 2], body="Alice: coucou"))
    gateway.inbox.close()

    assert client.uploads == [(ogg, "audio/ogg", "voice-message.ogg", len(ogg))]
    room_id, content = client.sent[0]
    assert room_id == "!dm:matrix.org"
    assert content["msgtype"] == "m.audio"
    assert content["url"] == "mxc://matrix.org/voice1"
    assert content["body"] == "Alice: coucou"
    assert content["info"] == {"mimetype": "audio/ogg", "size": len(ogg), "duration": 2000}
    assert content["org.matrix.msc1767.audio"] == {"duration": 2000, "waveform": [1, 2]}
    assert content[VOICE_FLAG] == {}


def test_download_returns_media_or_raises(config):
    gateway = MatrixGateway(config, client=MediaClient(b"abc"))
    assert run_result(gateway.download("mxc://matrix.org/bobvoice")) == b"abc"
    with pytest.raises(MatrixError):
        run_result(gateway.download("mxc://matrix.org/missing"))
    gateway.inbox.close()


def run_result(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------- turn capture
class Sink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        self.frames.append(frame)
        await self.push_frame(frame, direction)


async def _run_frames(processors, frames):
    worker = PipelineWorker(Pipeline(processors), cancel_on_idle_timeout=False, enable_rtvi=False)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    await asyncio.sleep(0.05)
    for frame in frames:
        await worker.queue_frame(frame)
    await asyncio.sleep(0.1)
    await worker.queue_frame(EndFrame())
    await asyncio.wait_for(task, timeout=5)


def _chunk(value: int, secs: float = 0.1) -> InputAudioRawFrame:
    return InputAudioRawFrame(bytes([value, 0]) * int(RATE * secs), RATE, 1)


def test_capture_keeps_the_turn_with_a_short_preroll():
    capture = UserTurnAudioCapture(preroll_secs=0.2)
    frames = [_chunk(1) for _ in range(5)]
    frames += [UserStartedSpeakingFrame(), _chunk(2), _chunk(2), UserStoppedSpeakingFrame(), _chunk(3)]
    run(_run_frames([capture], frames))

    clip = capture.take_last_turn()
    assert clip.sample_rate == RATE
    # 0.2 s of pre-roll, then the 0.2 s turn; the audio after the turn is left out.
    assert clip.audio == bytes([1, 0]) * int(RATE * 0.2) + bytes([2, 0]) * int(RATE * 0.2)
    assert capture.take_last_turn() is None


def test_capture_ends_the_turn_at_its_context_frame_and_caps_its_length():
    capture = UserTurnAudioCapture(preroll_secs=0.0, max_turn_secs=0.15)
    context = LLMContext([{"role": "user", "content": "hi"}])
    frames = [UserStartedSpeakingFrame(), _chunk(2), _chunk(2), LLMContextFrame(context=context), _chunk(3)]
    run(_run_frames([capture], frames))

    assert capture.take_last_turn().audio == bytes([2, 0]) * int(RATE * 0.15)


# ---------------------------------------------------- engine audio order
class TaggedTTS(TTSService):
    """Slow TTS whose audio is all 0x11 bytes."""

    def __init__(self):
        super().__init__(sample_rate=RATE)

    async def run_tts(self, text, context_id):
        await asyncio.sleep(0.1)
        yield TTSAudioRawFrame(b"\x11\x00" * 1600, RATE, 1, context_id=context_id)


class AudioTap(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.sources: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame) and frame.audio:
            source = {0x11: "tts", 0x22: "clip"}.get(frame.audio[0], "other")
            if not self.sources or self.sources[-1] != source:
                self.sources.append(source)
        await self.push_frame(frame, direction)


class Emitter(FrameProcessor):
    """Outputs ``parts`` the way the intent engine does, when a context frame comes."""

    _output = IntentEngineProcessor._output
    _play = IntentEngineProcessor._play
    _speak = IntentEngineProcessor._speak

    def __init__(self, parts):
        super().__init__()
        self.parts = parts

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._output(self.parts)
        else:
            await self.push_frame(frame, direction)


def test_clip_plays_between_the_sentences_around_it():
    """The TTS service keeps the clip after the sentence before it, though synthesis is slower."""
    clip = AudioClip(b"\x22\x00" * 8000, RATE)
    engine = Emitter(["Bob t'a envoyé un message vocal :", clip, "Tu veux lui répondre ?"])
    tts, tap = TaggedTTS(), AudioTap()

    async def scenario():
        worker = PipelineWorker(Pipeline([engine, tts, tap]), cancel_on_idle_timeout=False, enable_rtvi=False)
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        await worker.queue_frame(LLMContextFrame(context=LLMContext()))
        await asyncio.sleep(0.6)
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(task, timeout=5)

    run(scenario())
    assert tap.sources == ["tts", "clip", "tts"]


# ---------------------------------------------------------------- dialog
class FakeGateway(MatrixGateway):
    def __init__(self, config, media: bytes):
        super().__init__(config, client=SimpleNamespace(rooms={}, close=None))
        self.media = media
        self.sent_voice: list[tuple[str, bytes, int, str]] = []

    async def send_text(self, contact, text):
        raise AssertionError("no text message expected")

    async def send_voice(self, contact, audio, *, duration_ms, waveform, body):
        assert waveform and max(waveform) == WAVEFORM_MAX
        self.sent_voice.append((contact.name, audio, duration_ms, body))
        return "$v"

    async def download(self, mxc_url):
        if mxc_url != "mxc://matrix.org/bobvoice":
            raise MatrixError("not found")
        return self.media

    async def mark_read(self, messages):
        await asyncio.to_thread(self.inbox.mark_read, [m.event_id for m in messages])


class FakeLLM(LLMService):
    def __init__(self):
        super().__init__()
        self.requests = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.requests += 1
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(LLM_REPLY))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class OutputTap(FrameProcessor):
    """What the agent outputs: sentences, and ``("audio", seconds)`` for each clip."""

    def __init__(self):
        super().__init__()
        self.output: list[str | tuple[str, float]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame):
                self.output.append(frame.text)
            elif isinstance(frame, OutputAudioRawFrame):
                secs = len(frame.audio) / (2 * frame.sample_rate)
                if self.output and isinstance(self.output[-1], tuple):
                    self.output[-1] = ("audio", self.output[-1][1] + secs)
                else:
                    self.output.append(("audio", secs))
        await self.push_frame(frame, direction)


class Harness:
    def __init__(self, gateway, person=None, language="fr-FR", capture=True):
        self.person = person
        self.recording: AudioClip | None = None
        self.tools = MessagingTools(gateway=gateway, current_person=lambda: self.person, user_turn=lambda: 0)
        self.context = LLMContext([{"role": "system", "content": "sys"}])
        self.llm = FakeLLM()
        self.engine = build_intent_engine(
            IntentEngineConfig(enabled=True),
            language=language,
            context=self.context,
            llm=self.llm,
            server=MessagingIntents(
                self.tools,
                gateway.contacts.all,
                voice_capture=self._take if capture else None,
                playback_sample_rate=PLAYBACK_RATE,
            ),
        )
        self.tap = OutputTap()
        _, assistant_aggregator = LLMContextAggregatorPair(self.context)
        self.worker = PipelineWorker(
            Pipeline([self.engine, self.llm, self.tap, assistant_aggregator]),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )

    def _take(self):
        clip, self.recording = self.recording, None
        return clip

    async def __aenter__(self):
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc):
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._task, timeout=5)

    async def says(self, text: str, recording: AudioClip | None = None, *, heard: bool | None = None) -> list:
        """One user turn; returns everything the agent output for it.

        ``heard=True`` plays the previous reply to the end first; ``False`` talks over it.
        """
        self.recording = recording
        if heard is not None:
            await self.worker.queue_frame(BotStoppedSpeakingFrame() if heard else InterruptionFrame())
        before = len(self.tap.output)
        self.context.add_message({"role": "user", "content": text})
        await self.worker.queue_frame(LLMContextFrame(context=self.context))
        async with asyncio.timeout(5):
            while len(self.tap.output) == before:
                await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        return self.tap.output[before:]


@pytest.fixture
def gateway(config, ogg):
    gw = FakeGateway(config, ogg)
    yield gw
    gw.inbox.close()


def _receive_voice(gateway, event_id="$1", url="mxc://matrix.org/bobvoice", sent_at=None):
    gateway.inbox.add(
        InboxMessage(
            event_id=event_id,
            room_id="!dm:matrix.org",
            sender=BOB,
            kind=KIND_AUDIO,
            body="Voice message",
            sent_at=sent_at or time.time(),
            media_url=url,
            mimetype="audio/ogg",
            duration_ms=2000,
        )
    )


def _sentences(output) -> list[str]:
    return [part for part in output if isinstance(part, str)]


def _audio_secs(output) -> list[float]:
    return [round(part[1], 1) for part in output if isinstance(part, tuple)]


def test_reading_messages_plays_the_voice_message(gateway):
    _receive_voice(gateway)

    async def scenario():
        async with Harness(gateway) as h:
            output = await h.says("Lis les messages de Bob")
            assert isinstance(output[0], str) and output[0].endswith("Bob t'a envoyé un message vocal de 2 secondes :")
            assert _audio_secs(output) == [2.0]
            assert output[-1] == "Tu veux répondre à Bob ?"
            # The reply in the context is the spoken text, without the clip mark.
            assert "⟦" not in h.context.get_messages()[-1]["content"]
            # Played once: read now, so it is not new any more.
            assert _sentences(await h.says("J'ai des nouveaux messages ?")) == ["Tu n'as pas de nouveau message."]

    run(scenario())


def test_a_voice_message_that_cannot_be_downloaded_is_announced(gateway):
    _receive_voice(gateway, url="mxc://matrix.org/gone")

    async def scenario():
        async with Harness(gateway) as h:
            output = await h.says("Lis les messages de Bob")
            assert _audio_secs(output) == []
            assert "message vocal de 2 secondes, que je n'arrive pas à écouter." in output[0]

    run(scenario())


def test_play_the_latest_voice_message_again(gateway):
    _receive_voice(gateway, event_id="$old", sent_at=time.time() - 120)
    gateway.inbox.mark_read(["$old"])

    async def scenario():
        async with Harness(gateway) as h:
            output = await h.says("Fais-moi écouter le message vocal de Bob")
            assert _audio_secs(output) == [2.0]
            assert output[-1] == "Tu veux répondre à Bob ?"
            # "Reply with a voice message" to the one just played.
            assert _sentences(await h.says("Réponds-lui par un message vocal")) == [
                "Je t'écoute : dis ton message pour Bob."
            ]

    run(scenario())


def test_recorded_turn_is_played_back_and_sent_as_a_voice_message(gateway):
    recording = AudioClip(tone(2.0), RATE)

    async def scenario():
        async with Harness(gateway, ALICE) as h:
            assert _sentences(await h.says("Envoie un message vocal à Bob")) == [
                "Je t'écoute : dis ton message pour Bob."
            ]
            output = await h.says("Je rentre ce soir", recording=recording)
            assert output == ["J'ai enregistré 2 secondes pour Bob. Je l'envoie, ou tu veux le réécouter ?"]
            output = await h.says("Réécoute")
            assert output[0] == "Voici ton message :"
            assert _audio_secs(output) == [2.0]
            assert output[-1] == "Je l'envoie à Bob ?"
            assert gateway.sent_voice == []
            assert _sentences(await h.says("Oui")) == ["C'est envoyé à Bob."]
            assert h.llm.requests == 0

    run(scenario())
    [(name, audio, duration_ms, body)] = gateway.sent_voice
    assert (name, duration_ms, body) == ("Bob", 2000, "Alice: Je rentre ce soir")
    assert abs(pcm_duration_ms(decode_audio(audio, RATE), RATE) - 2000) < 30


def test_recording_again_and_nothing_heard(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Envoie à Bob un message vocal")
            # No audio for the turn (or too short): asked again.
            output = await h.says("euh", recording=AudioClip(tone(0.2), RATE))
            assert output == ["Je n'ai rien entendu. Dis ton message pour Bob."]
            await h.says("Premier essai", recording=AudioClip(tone(1.0), RATE))
            assert _sentences(await h.says("Recommence")) == ["Je t'écoute : dis ton message pour Bob."]
            await h.says("Deuxième essai", recording=AudioClip(tone(3.0), RATE))
            assert _sentences(await h.says("Vas-y")) == ["C'est envoyé à Bob."]

    run(scenario())
    # A guest's voice message is captioned with its transcript only.
    assert [(d, b) for _, _, d, b in gateway.sent_voice] == [(3000, "Deuxième essai")]


def test_recording_can_be_cancelled(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Envoie un message vocal à Bob")
            assert _sentences(await h.says("Annule", recording=AudioClip(tone(1.0), RATE))) == [
                "D'accord, je ne l'envoie pas."
            ]

    run(scenario())
    assert gateway.sent_voice == []


def test_words_over_a_recording_read_back_do_not_change_it(gateway):
    async def scenario():
        async with Harness(gateway, ALICE) as h:
            await h.says("Envoie un message vocal à Bob")
            await h.says("Je rentre ce soir", recording=AudioClip(tone(2.0), RATE))
            # Only a text draft goes on over its read-back; a recording stays as recorded.
            await h.says("et que je mange dehors", heard=False)
            draft = h.tools.draft
            assert (draft.text, draft.voice.duration_ms) == ("Je rentre ce soir", 2000)

    run(scenario())
    assert gateway.sent_voice == []


def test_without_microphone_audio_the_llm_answers(gateway):
    async def scenario():
        async with Harness(gateway, capture=False) as h:
            assert _sentences(await h.says("Envoie un message vocal à Bob")) == [LLM_REPLY]

    run(scenario())


def test_english_voice_dialog(gateway):
    _receive_voice(gateway)

    async def scenario():
        async with Harness(gateway, language="en-US") as h:
            output = await h.says("Play the voice message from Bob")
            assert output[0].endswith("Bob sent you a 2-second voice message:")
            assert _audio_secs(output) == [2.0]
            await h.says("Send a voice message to Bob")
            output = await h.says("See you tonight", recording=AudioClip(tone(1.0), RATE))
            assert output == ["I recorded 1 second for Bob. Shall I send it, or do you want to hear it first?"]
            assert _audio_secs(await h.says("Play it")) == [1.0]
            assert _sentences(await h.says("Yes")) == ["Sent to Bob."]

    run(scenario())
    assert [b for _, _, _, b in gateway.sent_voice] == ["See you tonight"]


def test_llm_tool_tells_how_to_hear_a_voice_message(gateway):
    from examples.shared.messaging_tools import describe_message

    _receive_voice(gateway)
    [message] = gateway.inbox.messages([BOB])
    result = describe_message(message, "Bob")
    assert result["voice_message"] is True
    assert "play the voice message from Bob" in result["note"]


def test_voice_draft_limits(gateway):
    tools = MessagingTools(gateway=gateway, current_person=lambda: None, user_turn=lambda: 0)
    bob = Contact(name="Bob", matrix_id=BOB)
    from examples.shared.messaging_tools import VoiceRecording

    assert tools.prepare_voice(bob, VoiceRecording(tone(0.3), RATE), "")["code"] == "no_voice"
    assert tools.prepare_voice(bob, VoiceRecording(bytes(2 * RATE * 121), RATE), "")["code"] == "voice_too_long"
    assert tools.prepare_voice(bob, VoiceRecording(tone(1.0), RATE), "  salut  toi ")["seconds"] == 1
    assert tools.draft.text == "salut toi" and tools.draft.voice is not None
