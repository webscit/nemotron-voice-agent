# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Wyoming bridge: sentence splitting, language mapping, info, and event sequences."""

import asyncio
import threading

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeVoice

from wyoming_bridge.handler import BridgeSettings, RivaEventHandler, build_info
from wyoming_bridge.riva_backend import SpeechCatalog, resolve_language
from wyoming_bridge.sentences import SentenceSplitter

CATALOG = SpeechCatalog(
    asr_models=("nemotron-asr",),
    asr_languages=("auto", "en-US", "fr", "fr-FR", "fr-CA", "de-DE"),
    tts_languages=("en-US", "fr-FR", "de-DE"),
    tts_voices={"John": ("en-US", "fr-FR", "de-DE"), "Sofia": ("en-US", "fr-FR", "de-DE")},
)


class FakeBackend:
    """Records calls; ``transcribe`` drains the audio iterator like Riva does."""

    def __init__(self, transcript="allume la lampe", fail=False):
        self.transcript = transcript
        self.fail = fail
        self.asr_calls = []
        self.tts_calls = []
        self.asr_threads = []

    def transcribe(self, chunks, language):
        self.asr_threads.append(threading.current_thread())
        audio = b"".join(chunks)
        self.asr_calls.append((language, audio))
        if self.fail:
            raise RuntimeError("riva down")
        return self.transcript

    def synthesize(self, text, voice, language, sample_rate):
        self.tts_calls.append((text, voice, language, sample_rate))
        if self.fail:
            raise RuntimeError("riva down")
        yield text.encode()[:4]
        yield b"\x00\x00"


def run_events(events, backend=None, settings=None):
    """Feed ``events`` to one handler and return what it wrote back."""
    backend = backend or FakeBackend()
    written = []

    async def main():
        handler = RivaEventHandler(
            None,
            None,
            backend=backend,
            info=build_info(CATALOG),
            catalog=CATALOG,
            settings=settings or BridgeSettings(language="fr-FR"),
        )

        async def write_event(event):
            written.append(event)

        handler.write_event = write_event
        for event in events:
            assert await handler.handle_event(event.event())

    asyncio.run(main())
    return written, backend


def types(events):
    return [e.type for e in events]


def pcm(rate=16000, ms=100):
    return AudioChunk(rate=rate, width=2, channels=1, audio=b"\x01\x00" * (rate * ms // 1000))


# ---- Sentence splitting ----


def test_splitter_releases_sentences_across_chunks_and_keeps_numbers_intact():
    splitter = SentenceSplitter()
    assert splitter.add("Bonjour, il est ") == []
    assert splitter.add("15:30. Le prix est 3.5 €. Bonne") == ["Bonjour, il est 15:30.", "Le prix est 3.5 €."]
    assert splitter.add(" journée !") == []
    assert splitter.finish() == ["Bonne journée !"]
    assert splitter.finish() == []


def test_splitter_splits_on_line_breaks_and_closing_quotes():
    splitter = SentenceSplitter()
    assert splitter.add("Il a dit « oui. » Puis\nil est parti") == ["Il a dit « oui. »", "Puis"]
    assert splitter.finish() == ["il est parti"]


# ---- Language mapping ----


def test_resolve_language_prefers_exact_then_default_then_same_base():
    available = CATALOG.asr_languages
    assert resolve_language("fr", available, "fr-FR") == "fr"
    assert resolve_language("FR_ca", available, "fr-FR") == "fr-CA"
    assert resolve_language(None, available, "fr-FR") == "fr-FR"
    tts = CATALOG.tts_languages
    assert resolve_language("fr", tts, "fr-FR") == "fr-FR"
    assert resolve_language("de", tts, "fr-FR") == "de-DE"
    assert resolve_language("ja", tts, "fr-FR") == "fr-FR"


# ---- Describe ----


def test_describe_lists_the_asr_model_and_streaming_tts_voices():
    written, _ = run_events([Describe()])
    info = Info.from_event(written[0])
    assert [p.name for p in info.asr] == ["nemo-speech-asr"]
    assert info.asr[0].models[0].name == "nemotron-asr"
    assert "fr-FR" in info.asr[0].models[0].languages
    assert [v.name for v in info.tts[0].voices] == ["John", "Sofia"]
    assert info.tts[0].supports_synthesize_streaming is True


def test_build_info_omits_programs_the_server_does_not_serve():
    info = build_info(SpeechCatalog(asr_models=("asr",), asr_languages=("en-US",)))
    assert len(info.asr) == 1 and info.tts == []


# ---- Speech-to-text ----


def test_transcribe_resamples_audio_off_the_event_loop_and_returns_the_transcript():
    events = [
        Transcribe(language="fr"),
        AudioStart(rate=48000, width=2, channels=1),
        pcm(48000),
        pcm(48000),
        AudioStop(),
    ]
    written, backend = run_events(events)
    assert types(written) == ["transcript"]
    transcript = Transcript.from_event(written[0])
    assert transcript.text == "allume la lampe" and transcript.language == "fr"
    language, audio = backend.asr_calls[0]
    assert language == "fr"
    assert len(audio) == 2 * 16000 * 200 // 1000  # 200 ms at 16 kHz mono PCM16
    assert backend.asr_threads[0] is not threading.main_thread()


def test_transcribe_without_a_language_uses_the_default_and_tolerates_missing_audio_start():
    written, backend = run_events([pcm(), AudioStop()])
    assert backend.asr_calls[0][0] == "fr-FR"
    assert types(written) == ["transcript"]


def test_transcribe_failure_is_reported_as_a_wyoming_error():
    written, _ = run_events([AudioStart(rate=16000, width=2, channels=1), pcm(), AudioStop()], FakeBackend(fail=True))
    assert types(written) == ["error"]
    assert Error.from_event(written[0]).code == "asr-failed"


# ---- Text-to-speech ----


def test_synthesize_streams_one_audio_response():
    written, backend = run_events([Synthesize(text="Bonjour.", voice=SynthesizeVoice(name="Sofia", language="fr"))])
    assert types(written) == ["audio-start", "audio-chunk", "audio-chunk", "audio-stop"]
    assert backend.tts_calls == [("Bonjour.", "Sofia", "fr-FR", 22050)]
    assert AudioStart.from_event(written[0]).rate == 22050


def test_streaming_synthesis_speaks_each_sentence_and_ignores_the_compat_synthesize():
    events = [
        SynthesizeStart(voice=SynthesizeVoice(language="fr-FR")),
        SynthesizeChunk(text="Bonjour, il est "),
        SynthesizeChunk(text="quinze heures. Bonne"),
        SynthesizeChunk(text=" journée !"),
        Synthesize(text="Bonjour, il est quinze heures. Bonne journée !"),
        SynthesizeStop(),
    ]
    written, backend = run_events(events, settings=BridgeSettings(language="fr-FR", tts_voice="John"))
    assert [call[0] for call in backend.tts_calls] == ["Bonjour, il est quinze heures.", "Bonne journée !"]
    assert {call[1] for call in backend.tts_calls} == {"John"}
    assert types(written) == ["audio-start"] + ["audio-chunk"] * 4 + ["audio-stop", "synthesize-stopped"]


def test_language_only_voice_picks_the_matching_locale():
    _, backend = run_events([Synthesize(text="Hi.", voice=SynthesizeVoice(language="en"))])
    assert backend.tts_calls == [("Hi.", None, "en-US", 22050)]


def test_unknown_voice_falls_back_to_the_server_default():
    _, backend = run_events([Synthesize(text="Salut.", voice=SynthesizeVoice(name="Nobody"))])
    assert backend.tts_calls == [("Salut.", None, "fr-FR", 22050)]


def test_named_voice_uses_its_own_language_when_it_lacks_the_default():
    catalog_voices = {**CATALOG.tts_voices, "Hans": ("de-DE",)}
    catalog = SpeechCatalog(CATALOG.asr_models, CATALOG.asr_languages, CATALOG.tts_languages, catalog_voices)
    written = []

    async def main():
        handler = RivaEventHandler(
            None, None, backend=backend, info=build_info(catalog), catalog=catalog, settings=BridgeSettings("fr-FR")
        )

        async def write_event(event):
            written.append(event)

        handler.write_event = write_event
        await handler.handle_event(Synthesize(text="Hallo.", voice=SynthesizeVoice(name="Hans")).event())

    backend = FakeBackend()
    asyncio.run(main())
    assert backend.tts_calls == [("Hallo.", "Hans", "de-DE", 22050)]


def test_streaming_synthesis_failure_still_closes_the_stream():
    events = [SynthesizeStart(), SynthesizeChunk(text="Bonjour."), SynthesizeStop()]
    written, _ = run_events(events, FakeBackend(fail=True))
    assert types(written) == ["error", "synthesize-stopped"]
    assert Error.from_event(written[0]).code == "tts-failed"


def test_streaming_synthesis_without_text_sends_an_empty_audio_response():
    written, backend = run_events([SynthesizeStart(), SynthesizeStop()])
    assert backend.tts_calls == []
    assert types(written) == ["audio-start", "audio-stop", "synthesize-stopped"]
