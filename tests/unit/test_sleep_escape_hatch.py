# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""A whole turn of "go to sleep" wins over every dialog and over the reply in progress."""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.workers.runner import WorkerRunner

from examples.multilingual.tool_handlers import ClientToolResultBridge, build_client_tool_handler
from examples.shared.intent_engine.config import IntentEngineConfig
from examples.shared.intent_engine.matcher import IntentMatcher
from examples.shared.intent_engine.processor import build_intent_engine
from examples.shared.intent_engine.targets import AudioClip
from examples.shared.messaging_intents import MessagingIntents
from examples.shared.messaging_tools import MessagingTools
from messaging.config import MatrixConfig
from messaging.contacts import Contact
from messaging.gateway import MatrixGateway

BOB = "@bob:matrix.org"
CONTACTS = f"""
contacts:
  - name: Bob
    matrix_id: "{BOB}"
"""
LLM_WORDS = [f"mot{n} " for n in range(10)]
# One second of silence-free audio at 16 kHz: long enough to be a voice message.
RECORDING = AudioClip(b"\x10\x00" * 16000, 16000)


class FakeGateway(MatrixGateway):
    def __init__(self, config):
        super().__init__(config, client=SimpleNamespace(rooms={}, close=None))
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, contact, text):
        self.sent.append((contact.name, text))
        return "$e"


class SlowLLM(LLMService):
    """Streams its answer one word every 50 ms, so a turn can arrive while it talks."""

    def __init__(self):
        super().__init__()
        self.requests = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.requests += 1
            await self.push_frame(LLMFullResponseStartFrame())
            for word in LLM_WORDS:
                await self.push_frame(LLMTextFrame(word))
                await asyncio.sleep(0.05)
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class Client(FrameProcessor):
    """Records what reaches the client and answers its tool calls (``go_to_sleep``)."""

    def __init__(self, answer):
        super().__init__()
        self.spoken: list[str] = []
        self.calls: list[str] = []
        self._answer = answer

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMTextFrame):
                self.spoken.append(frame.text)
            elif isinstance(frame, FunctionCallInProgressFrame):
                self.calls.append(frame.function_name)
                self.create_task(self._answer(frame))
        await self.push_frame(frame, direction)


class Harness:
    """engine (messaging + go_to_sleep client tool) -> slow LLM -> client -> bridge -> assistant aggregator."""

    def __init__(self, gateway, language="fr-FR"):
        self.gateway = gateway
        self.turn = 0
        self.recording: AudioClip | None = None
        self.tools = MessagingTools(gateway=gateway, current_person=lambda: None, user_turn=lambda: self.turn)
        self.intents = MessagingIntents(self.tools, gateway.contacts.all, voice_capture=self._take)
        self.context = LLMContext([{"role": "system", "content": "sys"}])
        self.llm = SlowLLM()
        bridge = ClientToolResultBridge()
        self.llm.register_function("go_to_sleep", build_client_tool_handler(0.4, bridge), cancel_on_interruption=True)
        self.engine = build_intent_engine(
            IntentEngineConfig(enabled=True),
            language=language,
            context=self.context,
            llm=self.llm,
            client_tool_names=("go_to_sleep",),
            client_tool_timeout_secs=0.4,
            server=self.intents,
        )
        assert self.engine is not None
        self.client = Client(self._client_answers)
        _, assistant_aggregator = LLMContextAggregatorPair(self.context)
        self.worker = PipelineWorker(
            Pipeline([self.engine, self.llm, self.client, bridge, assistant_aggregator]),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )

    def _take(self):
        clip, self.recording = self.recording, None
        return clip

    async def _client_answers(self, call: FunctionCallInProgressFrame) -> None:
        # The client's result enters at the start of the pipeline, as RTVI pushes it.
        await self.worker.queue_frame(
            FunctionCallResultFrame(
                function_name=call.function_name, tool_call_id=call.tool_call_id, arguments={}, result={"ok": True}
            )
        )

    async def __aenter__(self):
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._task = asyncio.create_task(runner.run())
        await asyncio.sleep(0.05)
        return self

    async def __aexit__(self, *exc):
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._task, timeout=5)

    async def says(self, text: str, *, recording: AudioClip | None = None, heard: bool = True) -> None:
        """One user turn. ``heard=False`` sends it with no interruption and no end of bot speech before it."""
        self.recording = recording
        if heard:
            await self.worker.queue_frame(BotStoppedSpeakingFrame())
        self.turn += 1
        self.context.add_message({"role": "user", "content": text})
        await self.worker.queue_frame(LLMContextFrame(context=self.context))

    async def answer(self, text: str, **kwargs) -> str:
        """One user turn; returns the last thing the agent said for it."""
        before = len(self.client.spoken)
        await self.says(text, **kwargs)
        await self.wait_for(lambda: len(self.client.spoken) > before)
        await asyncio.sleep(0.1)
        return self.client.spoken[-1]

    async def wait_for(self, condition, timeout: float = 3.0) -> None:
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.01)

    def assert_asleep_and_cleared(self, good_night: str = "Bonne nuit") -> None:
        assert self.client.spoken[-1] == good_night
        assert self.client.calls.count("go_to_sleep") == 1
        assert self.engine._dialog is None
        assert self.intents._state is None
        assert self.tools.draft is None


@pytest.fixture
def gateway(tmp_path):
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(CONTACTS, encoding="utf-8")
    config = MatrixConfig("https://matrix.org", "@agent:matrix.org", "t", "", tmp_path, contacts)
    gw = FakeGateway(config)
    yield gw
    gw.inbox.close()


def run(coro):
    asyncio.run(coro)


def test_sleep_wins_over_a_pending_messaging_question(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            answer = await h.answer("Dis à Bob que j'arrive")
            assert answer == "Je dis à Bob : j'arrive. Je l'envoie ?"
            assert h.tools.draft is not None
            assert await h.answer("Va dormir") == "Bonne nuit"
            h.assert_asleep_and_cleared()
            # The question is gone: "oui" no longer sends anything, the LLM gets it.
            await h.says("Oui")
            await h.wait_for(lambda: h.llm.requests == 1)

    run(scenario())
    assert gateway.sent == []


def test_sleep_wins_over_a_dictated_message(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            assert await h.answer("Envoie un message à Bob") == "Qu'est-ce que je dis à Bob ?"
            assert await h.answer("Va te coucher.") == "Bonne nuit"
            h.assert_asleep_and_cleared()
            assert h.llm.requests == 0

    run(scenario())
    assert gateway.sent == []


def test_sleep_wins_over_a_voice_recording(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            assert await h.answer("Envoie un message vocal à Bob") == "Je t'écoute : dis ton message pour Bob."
            assert await h.answer("Endors-toi", recording=RECORDING) == "Bonne nuit"
            h.assert_asleep_and_cleared()
            # The turn's audio was not taken as the voice message.
            assert h.recording is RECORDING

    run(scenario())


def test_sleep_cancels_an_intent_in_progress(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            started = asyncio.Event()

            async def slow_unread():
                started.set()
                await asyncio.sleep(1.0)
                return [(Contact(name="Bob", matrix_id=BOB), 1)]

            h.tools.unread = slow_unread
            await h.says("Est-ce que j'ai des messages ?")
            await asyncio.wait_for(started.wait(), timeout=2)
            # The next turn arrives with no interruption before it: the engine drops the turn itself.
            assert await h.answer("Va dormir", heard=False) == "Bonne nuit"
            await asyncio.sleep(1.1)
            assert h.client.spoken == ["Bonne nuit"]
            h.assert_asleep_and_cleared()

    run(scenario())


def test_sleep_stops_the_llm_answer_in_progress(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Raconte-moi une histoire")
            await h.wait_for(lambda: len(h.client.spoken) >= 2)
            # Said while the LLM still streams its answer, with no interruption from the user aggregator.
            await h.says("Va dormir", heard=False)
            await h.wait_for(lambda: "Bonne nuit" in h.client.spoken)
            await asyncio.sleep(0.6)
            words = [text for text in h.client.spoken if text in LLM_WORDS]
            assert len(words) < len(LLM_WORDS)
            assert h.client.spoken[-1] == "Bonne nuit"
            assert h.llm.requests == 1
            h.assert_asleep_and_cleared()

    run(scenario())


def test_a_dictated_message_about_sleeping_is_not_taken_as_sleep(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            assert await h.answer("Envoie un message à Bob") == "Qu'est-ce que je dis à Bob ?"
            answer = await h.answer("Dis à Bob que je vais dormir")
            assert answer == "Je dis à Bob : Dis à Bob que je vais dormir. Je l'envoie ?"
            assert h.tools.draft is not None
            # Outside of a dialog, the same words are a message too.
            assert await h.answer("Non") == "D'accord, je ne l'envoie pas."
            answer = await h.answer("Dis à Bob que je vais dormir")
            assert answer == "Je dis à Bob : je vais dormir. Je l'envoie ?"
            assert "go_to_sleep" not in h.client.calls

    run(scenario())


def test_english_go_to_sleep_wins_over_dictation(gateway):
    async def scenario():
        async with Harness(gateway, language="en-US") as h:
            assert await h.answer("Send a message to Bob") == "What should I tell Bob?"
            assert await h.answer("Go to sleep.") == "Good night"
            h.assert_asleep_and_cleared(good_night="Good night")
            # A dictated message about sleeping is still a message.
            assert await h.answer("Tell Bob I'm going to sleep") != "Good night"
            assert h.tools.draft is not None
            assert h.client.calls.count("go_to_sleep") == 1

    run(scenario())


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en-US", "Go to sleep"),
        ("en-US", "go to bed"),
        ("en-US", "Hey Reachy, it's time to sleep"),
        ("en-US", "Please go back to sleep now"),
        ("fr-FR", "Va dormir"),
        ("fr-FR", "Tu peux aller te coucher maintenant ?"),
        ("fr-FR", "Mets-toi en veille"),
        ("fr-FR", "C'est l'heure d'aller au lit"),
    ],
)
def test_default_sleep_sentences(language, text):
    matcher = IntentMatcher(
        language,
        allowed_intents=(),
        sentences_dir=IntentEngineConfig().sentences_dir,
        load_stock=False,
        wake_words=("Reachy",),
        keep_custom_intent=lambda name: name == "ReachyGoToSleep",
    )
    match = matcher.match(text)
    assert match is not None and match.name == "ReachyGoToSleep"


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en-US", "good night"),
        ("en-US", "tell Bob I'm going to sleep"),
        ("fr-FR", "bonne nuit"),
        ("fr-FR", "dis à Bob que je vais dormir"),
    ],
)
def test_sleep_needs_the_whole_turn(language, text):
    matcher = IntentMatcher(
        language,
        allowed_intents=(),
        sentences_dir=IntentEngineConfig().sentences_dir,
        load_stock=False,
        keep_custom_intent=lambda name: name == "ReachyGoToSleep",
    )
    assert matcher.match(text) is None


def test_english_and_french_cover_the_same_reachy_intents():
    def names(language):
        return IntentMatcher(
            language,
            allowed_intents=(),
            sentences_dir=IntentEngineConfig().sentences_dir,
            load_stock=False,
            keep_custom_intent=lambda name: name.startswith("Reachy"),
        ).custom_intent_names

    assert names("en-US") == names("fr-FR")
    assert "ReachyGoToSleep" in names("en-US")
