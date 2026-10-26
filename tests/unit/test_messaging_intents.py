# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Matrix messaging dialog in the intent engine: multi-turn, deterministic, no LLM."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
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

from examples.shared.intent_engine.config import IntentEngineConfig
from examples.shared.intent_engine.matcher import original_span, respell_slot_values
from examples.shared.intent_engine.processor import build_intent_engine
from examples.shared.messaging_intents import MessagingIntents, contact_slot_list
from examples.shared.messaging_tools import MessagingTools
from messaging.config import MatrixConfig
from messaging.contacts import Contact
from messaging.gateway import MatrixError, MatrixGateway
from messaging.inbox import KIND_AUDIO, KIND_TEXT, InboxMessage

BOB = "@bob:matrix.org"
ELODIE = "@elodie:matrix.org"
CONTACTS = f"""
contacts:
  - name: Bob
    matrix_id: "{BOB}"
    aliases: [Robert]
  - name: Élodie
    matrix_id: "{ELODIE}"
    owner: Alice
"""
ALICE = SimpleNamespace(person={"id": "p-alice", "name": "Alice"})
LLM_REPLY = "Réponse du LLM."


class FakeGateway(MatrixGateway):
    def __init__(self, config):
        super().__init__(config, client=SimpleNamespace(rooms={}, close=None))
        self.sent: list[tuple[str, str]] = []
        self.fail = False

    async def send_text(self, contact, text):
        if self.fail:
            raise MatrixError("offline")
        self.sent.append((contact.name, text))
        return "$e"

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


class Tap(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.spoken: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMTextFrame):
            self.spoken.append(frame.text)
        await self.push_frame(frame, direction)


class Harness:
    """engine (messaging only, no Home Assistant) -> fake LLM -> tap -> assistant aggregator."""

    def __init__(self, gateway, person=None, language="fr-FR"):
        self.person = person
        self.gateway = gateway
        self.tools = MessagingTools(gateway=gateway, current_person=lambda: self.person, user_turn=lambda: 0)
        self.context = LLMContext([{"role": "system", "content": "sys"}])
        self.llm = FakeLLM()
        self.engine = build_intent_engine(
            IntentEngineConfig(enabled=True),
            language=language,
            context=self.context,
            llm=self.llm,
            server=MessagingIntents(self.tools, gateway.contacts.all),
        )
        assert self.engine is not None
        self.tap = Tap()
        _, assistant_aggregator = LLMContextAggregatorPair(self.context)
        self.worker = PipelineWorker(
            Pipeline([self.engine, self.llm, self.tap, assistant_aggregator]),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
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

    async def says(self, text: str) -> str:
        """One user turn; returns what the agent answered."""
        before = len(self.tap.spoken)
        self.context.add_message({"role": "user", "content": text})
        await self.worker.queue_frame(LLMContextFrame(context=self.context))
        async with asyncio.timeout(3):
            while len(self.tap.spoken) == before:
                await asyncio.sleep(0.01)
        # Let the assistant aggregator store the reply.
        await asyncio.sleep(0.05)
        return self.tap.spoken[-1]


@pytest.fixture
def gateway(tmp_path):
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(CONTACTS, encoding="utf-8")
    config = MatrixConfig("https://matrix.org", "@agent:matrix.org", "t", "", tmp_path, contacts)
    gw = FakeGateway(config)
    yield gw
    gw.inbox.close()


def _receive(gateway, event_id, sender, body="Salut", kind=KIND_TEXT, sent_at=None):
    gateway.inbox.add(
        InboxMessage(
            event_id=event_id,
            room_id="!dm:matrix.org",
            sender=sender,
            kind=kind,
            body=body,
            sent_at=sent_at or time.time(),
        )
    )


def run(coro):
    asyncio.run(coro)


def test_dictated_message_is_read_back_and_sent_on_yes(gateway):
    async def scenario():
        async with Harness(gateway, ALICE) as h:
            assert await h.says("[speaker: Alice] Envoie un message à Bob") == "Qu'est-ce que je dis à Bob ?"
            # Dictation: the words are the message, even when they look like a command.
            answer = await h.says("[speaker: Alice] Allume la lumière en rentrant, s'il te plaît")
            assert answer == "Je dis à Bob : Allume la lumière en rentrant, s'il te plaît. Je l'envoie ?"
            assert gateway.sent == []
            assert await h.says("Oui, envoie-le") == "C'est envoyé à Bob."
            assert h.llm.requests == 0

    run(scenario())
    assert gateway.sent == [("Bob", "Alice: Allume la lumière en rentrant, s'il te plaît")]


def test_one_shot_message_keeps_punctuation_and_can_be_declined(gateway):
    async def scenario():
        async with Harness(gateway, ALICE) as h:
            answer = await h.says("Dis à Robert que le dîner est prêt, tu viens ?")
            assert answer == "Je dis à Bob : le dîner est prêt, tu viens ? Je l'envoie ?"
            assert await h.says("Non") == "D'accord, je ne l'envoie pas."
            assert h.tools.draft is None
            # The question is closed: "oui" is no longer an answer, the LLM gets it.
            assert await h.says("Oui") == LLM_REPLY

    run(scenario())
    assert gateway.sent == []


def test_change_redrafts_before_sending(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Écris un message à Bob pour lui dire que j'arrive à 8 heures")
            answer = await h.says("Non, plutôt que j'arrive à 9 heures")
            assert answer == "Je dis à Bob : j'arrive à 9 heures. Je l'envoie ?"
            await h.says("Change le message")
            await h.says("Je serai en retard")
            assert await h.says("Vas-y") == "C'est envoyé à Bob."

    run(scenario())
    # A guest's message is not signed.
    assert gateway.sent == [("Bob", "Je serai en retard")]


def test_unclear_answer_is_asked_again_once_then_goes_to_llm(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Dis à Bob que je pars")
            assert await h.says("Euh je sais pas") == "Je l'envoie à Bob ? Réponds par oui ou par non."
            assert await h.says("Hmm peut-être") == LLM_REPLY
            # The draft stays for the LLM tools, but nothing was sent.
            assert h.tools.draft is not None

    run(scenario())
    assert gateway.sent == []


def test_check_then_read_then_reply(gateway):
    _receive(gateway, "$1", BOB, "Tu es là ?", sent_at=time.time() - 60)
    _receive(gateway, "$2", BOB, "Rappelle-moi", sent_at=time.time())

    async def scenario():
        async with Harness(gateway, ALICE) as h:
            assert await h.says("Est-ce que j'ai des messages ?") == "Tu as 2 nouveaux messages de Bob. Je te les lis ?"
            answer = await h.says("Oui")
            assert "Bob a écrit : Tu es là ?" in answer
            assert "Bob a écrit : Rappelle-moi." in answer
            assert answer.endswith("Tu veux répondre à Bob ?")
            assert await h.says("Oui") == "Qu'est-ce que je dis à Bob ?"
            await h.says("Je te rappelle ce soir")
            assert await h.says("Oui") == "C'est envoyé à Bob."
            assert await h.says("J'ai des nouveaux messages ?") == "Tu n'as pas de nouveau message."

    run(scenario())
    assert gateway.sent == [("Bob", "Alice: Je te rappelle ce soir")]


def test_voice_messages_are_announced(gateway):
    _receive(gateway, "$1", BOB, "voice.ogg", kind=KIND_AUDIO)

    async def scenario():
        async with Harness(gateway) as h:
            answer = await h.says("Lis les messages de Bob")
            assert "Bob t'a envoyé un message vocal" in answer

    run(scenario())


def test_owned_contact_needs_the_person_known_for_sure(gateway):
    _receive(gateway, "$1", ELODIE, "Secret")

    async def scenario():
        async with Harness(gateway) as h:
            answer = await h.says("Envoie un message à Elodie")
            assert answer == "Je ne sais pas avec certitude qui me parle, alors je ne peux pas utiliser ce contact."
            assert await h.says("Est-ce que j'ai des messages ?") == "Tu n'as pas de nouveau message."
            h.person = ALICE
            answer = await h.says("Qu'est-ce qu'Élodie m'a écrit")
            assert "Élodie a écrit : Secret." in answer

    run(scenario())


def test_failed_send_keeps_the_draft(gateway):
    gateway.fail = True

    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Dis à Bob que j'arrive")
            assert await h.says("Oui") == "Je n'arrive pas à envoyer le message pour l'instant."
            assert h.tools.draft is not None

    run(scenario())


def test_cancel_dictation(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Envoie un message à Bob")
            assert await h.says("Annule") == "D'accord, je ne l'envoie pas."
            assert await h.says("Réponds-lui") == LLM_REPLY

    run(scenario())


def test_english_dialog(gateway):
    async def scenario():
        async with Harness(gateway, ALICE, language="en-US") as h:
            assert await h.says("Tell Bob that I'm on my way!") == "I'll tell Bob: I'm on my way! Shall I send it?"
            assert await h.says("Yes please") == "Sent to Bob."

    run(scenario())
    assert gateway.sent == [("Bob", "Alice: I'm on my way!")]


def test_contact_slot_list_has_names_and_aliases():
    lists = contact_slot_list([Contact(name="Élodie", matrix_id=ELODIE, aliases=("Lolo",))])
    assert {(str(value.text_in.text), value.value_out) for value in lists.values} == {
        ("Élodie", ELODIE),
        ("Lolo", ELODIE),
    }


def test_contact_matches_whatever_the_accents(tmp_path):
    # The contacts file says "Frederic", the ASR writes "Frédéric" (session aba977fa6ca5).
    contacts = tmp_path / "contacts.yaml"
    contacts.write_text(f'contacts:\n  - name: Frederic\n    matrix_id: "{BOB}"\n', encoding="utf-8")
    gw = FakeGateway(MatrixConfig("https://matrix.org", "@agent:matrix.org", "t", "", tmp_path, contacts))

    async def scenario():
        async with Harness(gw, ALICE) as h:
            for asked in ("Je vais envoyer un message à Frédéric", "Écrit un message à FRÉDÉRIC."):
                assert await h.says(asked) == "Qu'est-ce que je dis à Frederic ?"
                assert await h.says("Non") == "D'accord, je ne l'envoie pas."
            answer = await h.says("Di à Frédéric que le repas est prêt.")
            assert answer == "Je dis à Frederic : le repas est prêt. Je l'envoie ?"
            assert await h.says("Oui") == "C'est envoyé à Frederic."
            assert await h.says("A tu de nouveau message pour moi?") == "Tu n'as pas de nouveau message."
            assert h.llm.requests == 0

    try:
        run(scenario())
    finally:
        gw.inbox.close()
    assert gw.sent == [("Frederic", "Alice: le repas est prêt.")]


def test_respell_slot_values():
    lists = contact_slot_list([Contact(name="Élodie", matrix_id=ELODIE, aliases=("Tante Zoé",))])
    assert respell_slot_values("dis à ELODIE et à tante zoe que oui", {"contact": lists}) == (
        "dis à Élodie et à Tante Zoé que oui"
    )
    # Whole words only.
    assert respell_slot_values("les élodies", {"contact": lists}) == "les élodies"


def test_original_span():
    text = "dis à Bob que le dîner est prêt, tu viens ?"
    assert original_span(text, "le dîner est prêt tu viens") == "le dîner est prêt, tu viens ?"
    assert original_span(text, "absent") == "absent"


def test_system_note_does_not_answer_the_pending_question(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Dis à Bob que j'arrive")
            # Voice ID greets somebody who came into view: the LLM answers, the question stays open.
            assert await h.says("[presence: Alice came into view]") == LLM_REPLY
            assert await h.says("Oui") == "C'est envoyé à Bob."

    run(scenario())
    assert gateway.sent == [("Bob", "j'arrive")]
