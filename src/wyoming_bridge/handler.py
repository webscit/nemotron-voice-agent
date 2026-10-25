# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wyoming event handler: speech-to-text and (streaming) text-to-speech over Riva."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from loguru import logger
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped, SynthesizeVoice

from wyoming_bridge.riva_backend import ASR_SAMPLE_RATE, SpeechCatalog, resolve_language
from wyoming_bridge.sentences import SentenceSplitter

_DONE = object()
_ATTRIBUTION = Attribution(name="NVIDIA NeMo", url="https://github.com/NVIDIA/NeMo-Speech.cpp")


class SpeechBackend(Protocol):
    """The blocking calls the handler needs (``RivaBackend`` in production)."""

    def transcribe(self, chunks: Iterator[bytes], language: str) -> str:
        """Return the transcript of 16 kHz mono PCM16 ``chunks``."""
        ...

    def synthesize(self, text: str, voice: str | None, language: str, sample_rate: int) -> Iterator[bytes]:
        """Yield mono PCM16 audio for ``text``."""
        ...


@dataclass(frozen=True)
class BridgeSettings:
    """Defaults applied when a Wyoming request leaves a field out."""

    language: str
    tts_voice: str = ""
    tts_sample_rate: int = 22050


def build_info(catalog: SpeechCatalog, version: str | None = None) -> Info:
    """Describe the server's ASR model and TTS voices as Wyoming programs.

    Program names match the NeMo-Speech.cpp built-in Wyoming server, so Home Assistant
    shows the same names after switching to the bridge.
    """

    def artifact(name: str) -> dict:
        return {"name": name, "attribution": _ATTRIBUTION, "installed": True, "description": None, "version": None}

    models = [AsrModel(**artifact(name), languages=list(catalog.asr_languages)) for name in catalog.asr_models]
    voices = [TtsVoice(**artifact(name), languages=list(langs)) for name, langs in catalog.tts_voices.items()]
    asr = AsrProgram(**{**artifact("nemo-speech-asr"), "version": version}, models=models)
    tts = TtsProgram(
        **{**artifact("nemo-speech-tts"), "version": version}, voices=voices, supports_synthesize_streaming=True
    )
    return Info(asr=[asr] if models else [], tts=[tts] if voices else [])


async def iterate_in_thread[T](make_iterator: Callable[[], Iterator[T]]) -> AsyncIterator[T]:
    """Run a blocking iterator in the default executor and yield its items here."""
    loop = asyncio.get_running_loop()
    items: asyncio.Queue = asyncio.Queue()

    def run() -> None:
        try:
            for item in make_iterator():
                loop.call_soon_threadsafe(items.put_nowait, (item, None))
            loop.call_soon_threadsafe(items.put_nowait, (_DONE, None))
        except BaseException as exc:
            loop.call_soon_threadsafe(items.put_nowait, (_DONE, exc))

    worker = loop.run_in_executor(None, run)
    while True:
        item, exc = await items.get()
        if item is _DONE:
            await worker
            if exc is not None:
                raise exc
            return
        yield item


class _Recognition:
    """One speech-to-text request: audio is forwarded to Riva while it arrives."""

    def __init__(self, backend: SpeechBackend, language: str):
        self.language = language
        self._audio: queue.Queue[bytes | None] = queue.Queue()
        self._converter = AudioChunkConverter(rate=ASR_SAMPLE_RATE, width=2, channels=1)
        self._result = asyncio.ensure_future(
            asyncio.to_thread(backend.transcribe, iter(self._audio.get, None), language)
        )

    def feed(self, chunk: AudioChunk) -> None:
        self._audio.put(self._converter.convert(chunk).audio)

    async def finish(self) -> str:
        self._audio.put(None)
        return await self._result

    def abort(self) -> None:
        self._audio.put(None)


class RivaEventHandler(AsyncEventHandler):
    """Serves one Wyoming client connection."""

    def __init__(self, *args, backend: SpeechBackend, info: Info, catalog: SpeechCatalog, settings: BridgeSettings):
        """Bind the shared backend, the precomputed ``info``, and the request defaults."""
        super().__init__(*args)
        self._backend = backend
        self._info_event = info.event()
        self._catalog = catalog
        self._settings = settings
        self._asr_language: str | None = None
        self._recognition: _Recognition | None = None
        # Streaming synthesis state, set between synthesize-start and synthesize-stop.
        self._splitter: SentenceSplitter | None = None
        self._sentences: asyncio.Queue[str | None] | None = None
        self._speaker: asyncio.Task | None = None

    async def handle_event(self, event: Event) -> bool:
        """Dispatch one client event; always keeps the connection open."""
        if Describe.is_type(event.type):
            await self.write_event(self._info_event)
        elif Transcribe.is_type(event.type):
            self._asr_language = Transcribe.from_event(event).language
        elif AudioStart.is_type(event.type):
            self._start_recognition()
        elif AudioChunk.is_type(event.type):
            if self._recognition is None:
                self._start_recognition()
            self._recognition.feed(AudioChunk.from_event(event))
        elif AudioStop.is_type(event.type):
            await self._finish_recognition()
        elif SynthesizeStart.is_type(event.type):
            await self._start_streaming_synthesis(SynthesizeStart.from_event(event).voice)
        elif SynthesizeChunk.is_type(event.type):
            if self._splitter is not None:
                for sentence in self._splitter.add(SynthesizeChunk.from_event(event).text):
                    self._sentences.put_nowait(sentence)
        elif Synthesize.is_type(event.type):
            # Streaming clients also send the whole text once for older servers; the
            # sentences already arrive as chunks, so only a non-streaming request counts.
            if self._splitter is None:
                await self._synthesize_once(Synthesize.from_event(event))
        elif SynthesizeStop.is_type(event.type):
            await self._finish_streaming_synthesis()
        return True

    async def disconnect(self) -> None:
        """Release the in-flight recognition and synthesis, if any."""
        if self._recognition is not None:
            self._recognition.abort()
            self._recognition = None
        if self._speaker is not None:
            self._speaker.cancel()
            self._speaker = None

    # ---- Speech-to-text ----

    def _start_recognition(self) -> None:
        if self._recognition is not None:
            self._recognition.abort()
        language = resolve_language(self._asr_language, self._catalog.asr_languages, self._settings.language)
        self._recognition = _Recognition(self._backend, language)

    async def _finish_recognition(self) -> None:
        recognition, self._recognition = self._recognition, None
        self._asr_language = None
        if recognition is None:
            return
        try:
            text = await recognition.finish()
        except Exception as exc:
            logger.opt(exception=exc).error("Riva recognition failed")
            await self.write_event(Error(text=f"Speech recognition failed: {exc}", code="asr-failed").event())
            return
        logger.info(f"Transcript ({recognition.language}): {text!r}")
        await self.write_event(Transcript(text=text, language=recognition.language).event())

    # ---- Text-to-speech ----

    def _resolve_voice(self, voice: SynthesizeVoice | None) -> tuple[str | None, str]:
        name = (voice.name if voice else None) or self._settings.tts_voice or None
        if name and name not in self._catalog.tts_voices:
            logger.warning(f"Unknown TTS voice {name!r}; using the server default")
            name = None
        # A named voice arrives without a language (Wyoming drops it), so keep the default
        # language when that voice speaks it, else use the voice's first language.
        requested = voice.language if voice else None
        language = resolve_language(requested, self._catalog.tts_languages, self._settings.language)
        voice_languages = self._catalog.tts_voices.get(name, ()) if name else ()
        if voice_languages and language not in voice_languages:
            language = voice_languages[0]
        return name, language

    async def _speak(self, text: str, voice: str | None, language: str, started: bool) -> bool:
        """Write the audio for ``text``; sends ``audio-start`` first unless ``started``."""
        rate = self._settings.tts_sample_rate
        async for audio in iterate_in_thread(lambda: self._backend.synthesize(text, voice, language, rate)):
            if not started:
                await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
                started = True
            await self.write_event(AudioChunk(rate=rate, width=2, channels=1, audio=audio).event())
        return started

    async def _synthesize_once(self, request: Synthesize) -> None:
        voice, language = self._resolve_voice(request.voice)
        rate = self._settings.tts_sample_rate
        try:
            started = await self._speak(request.text, voice, language, started=False)
        except Exception as exc:
            logger.opt(exception=exc).error("Riva synthesis failed")
            await self.write_event(Error(text=f"Speech synthesis failed: {exc}", code="tts-failed").event())
            return
        if not started:
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())

    async def _start_streaming_synthesis(self, voice: SynthesizeVoice | None) -> None:
        await self._finish_streaming_synthesis()
        name, language = self._resolve_voice(voice)
        self._splitter = SentenceSplitter()
        self._sentences = asyncio.Queue()
        self._speaker = asyncio.create_task(self._speak_sentences(self._sentences, name, language))

    async def _speak_sentences(self, sentences: asyncio.Queue[str | None], voice: str | None, language: str) -> bool:
        """Synthesize queued sentences in order; returns whether audio was started."""
        started = False
        while (sentence := await sentences.get()) is not None:
            logger.debug(f"Synthesizing ({language}): {sentence!r}")
            started = await self._speak(sentence, voice, language, started)
        return started

    async def _finish_streaming_synthesis(self) -> None:
        if self._splitter is None:
            return
        for sentence in self._splitter.finish():
            self._sentences.put_nowait(sentence)
        self._sentences.put_nowait(None)
        speaker = self._speaker
        self._splitter = self._sentences = self._speaker = None
        rate = self._settings.tts_sample_rate
        try:
            started = await speaker
        except Exception as exc:
            logger.opt(exception=exc).error("Riva synthesis failed")
            await self.write_event(Error(text=f"Speech synthesis failed: {exc}", code="tts-failed").event())
            await self.write_event(SynthesizeStopped().event())
            return
        if not started:
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())
        await self.write_event(SynthesizeStopped().event())
