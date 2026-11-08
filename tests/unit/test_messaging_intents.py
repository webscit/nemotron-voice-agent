# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""Matrix messaging dialog in the intent engine: multi-turn, deterministic, no LLM."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.workers.runner import WorkerRunner

from examples.shared import messaging_announcer
from examples.shared.conversation_activity import ConversationActivity
from examples.shared.intent_engine.config import IntentEngineConfig
from examples.shared.intent_engine.matcher import original_span, respell_slot_values
from examples.shared.intent_engine.processor import build_intent_engine
from examples.shared.messaging_announcer import MessagingAnnouncer
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

    def __init__(self, gateway, person=None, language="fr-FR", dry_run=False, announce=False):
        self.person = person
        self.gateway = gateway
        self.turn = 0
        self.tools = MessagingTools(gateway=gateway, current_person=lambda: self.person, user_turn=lambda: self.turn)
        self.context = LLMContext([{"role": "system", "content": "sys"}])
        self.llm = FakeLLM()
        self.engine = build_intent_engine(
            IntentEngineConfig(enabled=True, dry_run=dry_run),
            language=language,
            context=self.context,
            llm=self.llm,
            server=(intents := MessagingIntents(self.tools, gateway.contacts.all)),
        )
        assert self.engine is not None
        self.tap = Tap()
        _, assistant_aggregator = LLMContextAggregatorPair(self.context)
        self.activity = ConversationActivity(quiet_secs=0.0)
        announcer = MessagingAnnouncer(
            gateway=gateway, tools=self.tools, intents=intents, engine=self.engine, activity=self.activity
        )
        self.worker = PipelineWorker(
            Pipeline([*([announcer] if announce else []), self.engine, self.llm, self.tap, assistant_aggregator]),
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

    async def says(self, text: str, *, heard: bool = True) -> str:
        """One user turn; returns what the agent answered.

        ``heard`` plays the previous reply to the end first; otherwise the user talks over it.
        """
        await self.worker.queue_frame(BotStoppedSpeakingFrame() if heard else InterruptionFrame())
        before = len(self.tap.spoken)
        self.turn += 1
        self.context.add_message({"role": "user", "content": text})
        await self.worker.queue_frame(LLMContextFrame(context=self.context))
        return await self._reply(before)

    async def heard(self) -> None:
        """The bot finished speaking its last reply."""
        await self.worker.queue_frame(BotStoppedSpeakingFrame())
        await asyncio.sleep(0.02)

    async def announced(self, before: int) -> str:
        """What the agent said by itself after ``before`` replies."""
        return await self._reply(before)

    async def llm_calls(self, tool: str, **arguments) -> tuple[dict, object, str | None]:
        """The LLM calls ``tool``; returns its result, the result properties and what was spoken."""
        before = len(self.tap.spoken)
        calls = []

        async def callback(result, *, properties=None):
            calls.append((result, properties))

        params = SimpleNamespace(arguments=arguments, result_callback=callback, llm=self.llm)
        await getattr(self.tools, tool)(params)
        ((result, properties),) = calls
        if properties is not None and properties.run_llm is False:
            return result, properties, await self._reply(before)
        return result, properties, None

    async def _reply(self, before: int) -> str:
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


# ------------------------------------------------------- glued yes/no answers
def test_answer_glued_to_the_next_sentence_counts(gateway):
    _receive(gateway, "$1", BOB, "Je vais bien merci")

    async def scenario():
        async with Harness(gateway, ALICE) as h:
            await h.says("J'ai reçu de nouveaux messages.")
            # Session 321b8c1eb79d: a short "Oui" glued to the next turn by the ASR.
            answer = await h.says("Oui.  Li mes nouveaux messages.")
            assert "Bob a écrit : Je vais bien merci." in answer
            assert await h.says("Oui, oui  Oui") == "Qu'est-ce que je dis à Bob ?"
            await h.says("Moi, ça va")
            assert await h.says("Oui  Envoie le message.") == "C'est envoyé à Bob."
            assert h.llm.requests == 0

    run(scenario())
    assert gateway.sent == [("Bob", "Alice: Moi, ça va")]


def test_conflicting_answers_ask_again(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Dis à Bob que je pars")
            assert await h.says("Oui  Oui.  Non.") == "Je l'envoie à Bob ? Réponds par oui ou par non."
            assert await h.says("Oui, je confirme") == "C'est envoyé à Bob."

    run(scenario())
    assert gateway.sent == [("Bob", "je pars")]


def test_a_later_change_wins_over_a_leading_no(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Dis à Bob que je pars")
            assert await h.says("Non, change le message") == "Qu'est-ce que je dis à Bob ?"
            assert h.tools.draft is not None

    run(scenario())


def test_dictation_starting_with_no_is_the_message(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Écris à Bob")
            answer = await h.says("Non, je ne viens pas ce soir")
            assert answer == "Je dis à Bob : Non, je ne viens pas ce soir. Je l'envoie ?"

    run(scenario())


def test_english_answer_glued_to_the_next_sentence(gateway):
    async def scenario():
        async with Harness(gateway, language="en-US") as h:
            await h.says("Tell Bob that I'm late")
            assert await h.says("Yes. Thank you so much") == "Sent to Bob."

    run(scenario())
    assert gateway.sent == [("Bob", "I'm late")]


# ------------------------------------------------------------ split dictation
def test_words_said_over_the_read_back_continue_the_message(gateway):
    async def scenario():
        async with Harness(gateway, ALICE) as h:
            assert await h.says("Dis à Bob que je rentre tard") == "Je dis à Bob : je rentre tard. Je l'envoie ?"
            # Turn detection cut the dictation: the user goes on over the read-back.
            answer = await h.says("et que je mange dehors", heard=False)
            assert answer == "Je dis à Bob : je rentre tard et que je mange dehors. Je l'envoie ?"
            assert await h.says("Oui", heard=False) == "C'est envoyé à Bob."

    run(scenario())
    assert gateway.sent == [("Bob", "Alice: je rentre tard et que je mange dehors")]


def test_words_after_the_read_back_do_not_continue_the_message(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Dis à Bob que je rentre tard")
            answer = await h.says("et que je mange dehors")
            assert answer == "Je l'envoie à Bob ? Réponds par oui ou par non."
            # Talking over the question asked again is an answer, not more of the message.
            assert await h.says("euh attends", heard=False) == LLM_REPLY
            assert h.tools.draft.text == "je rentre tard"

    run(scenario())
    assert gateway.sent == []


# ------------------------------------------------- send steps run by the LLM
def test_llm_draft_is_read_back_and_confirmed_by_the_engine(gateway):
    async def scenario():
        async with Harness(gateway, ALICE) as h:
            h.turn = 1
            result, properties, spoken = await h.llm_calls("send_message", contact="Robert", text="Le repas est prêt.")
            assert spoken == "Je dis à Bob : Le repas est prêt. Je l'envoie ?"
            assert result["spoken"] == spoken and "next" not in result
            assert properties.run_llm is False
            # The yes goes to the engine, which says what the gateway did.
            assert await h.says("Oui, je confirme") == "C'est envoyé à Bob."
            assert h.llm.requests == 0

    run(scenario())
    assert gateway.sent == [("Bob", "Alice: Le repas est prêt.")]


def test_llm_confirming_without_a_draft_never_says_sent(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            # Session aba977fa6ca5: the LLM confirmed a draft it never made, then said it was sent.
            result, properties, spoken = await h.llm_calls("confirm_send_message")
            assert result["code"] == "no_draft"
            assert properties.run_llm is False
            assert spoken == "Il n'y a pas de message en attente."

    run(scenario())
    assert gateway.sent == []


def test_llm_confirm_says_what_the_gateway_did(gateway):
    async def scenario():
        async with Harness(gateway) as h:
            await h.says("Dis à Bob que j'arrive")
            gateway.fail = True
            h.turn += 1
            _, _, spoken = await h.llm_calls("confirm_send_message")
            assert spoken == "Je n'arrive pas à envoyer le message pour l'instant."
            gateway.fail = False
            _, _, spoken = await h.llm_calls("confirm_send_message")
            assert spoken == "C'est envoyé à Bob."

    run(scenario())
    assert gateway.sent == [("Bob", "j'arrive")]


def test_llm_speaks_its_send_steps_in_a_dry_run(gateway):
    async def scenario():
        async with Harness(gateway, dry_run=True) as h:
            result, properties, spoken = await h.llm_calls("send_message", contact="Bob", text="Coucou")
            assert properties is None and spoken is None
            assert "next" in result

    run(scenario())


# ------------------------------------------------------------ announcements
@pytest.fixture
def fast_announcer(monkeypatch):
    monkeypatch.setattr(messaging_announcer, "POLL_SECS", 0.01)
    monkeypatch.setattr(messaging_announcer, "OWNER_WAIT_SECS", 0.01)


def _arrive(gateway, event_id, sender, body="Salut"):
    """A contact's message arrives during the session (stored, then the listeners told)."""
    _receive(gateway, event_id, sender, body)
    gateway._notify(gateway.inbox.messages([sender], unread_only=False, limit=50)[-1], None)


def test_new_message_is_announced_once_idle_and_yes_reads_it(gateway, fast_announcer):
    _receive(gateway, "$old", BOB, "Avant la session", sent_at=time.time() - 600)

    async def scenario():
        async with Harness(gateway, announce=True) as h:
            await h.heard()
            _arrive(gateway, "$1", BOB, "Tu rentres quand ?")
            # What just arrived, and what a yes reads: the backlog too.
            answer = await h.announced(0)
            assert answer == "Tu viens de recevoir un nouveau message de Bob, et tu en as 2 non lus. Je te les lis ?"
            answer = await h.says("Oui")
            assert "Bob a écrit : Tu rentres quand ?" in answer
            assert h.llm.requests == 0
            # Announced once.
            await h.heard()
            await asyncio.sleep(0.1)
            assert h.tap.spoken[-1] == answer

    run(scenario())


def test_backlog_alone_is_not_announced(gateway, fast_announcer):
    _receive(gateway, "$old", BOB, "Avant la session", sent_at=time.time() - 600)

    async def scenario():
        async with Harness(gateway, announce=True) as h:
            await h.heard()
            await asyncio.sleep(0.1)
            assert h.tap.spoken == []

    run(scenario())


def test_announcement_waits_for_a_pending_draft_and_its_reply(gateway, fast_announcer):
    async def scenario():
        async with Harness(gateway, ALICE, announce=True) as h:
            assert await h.says("Dis à Bob que je pars") == "Je dis à Bob : je pars. Je l'envoie ?"
            await h.heard()
            _arrive(gateway, "$1", BOB, "Attends-moi")
            _arrive(gateway, "$2", BOB, "J'arrive")
            await asyncio.sleep(0.1)
            # The read-back question stays the one answered.
            assert len(h.tap.spoken) == 1
            assert await h.says("Oui") == "C'est envoyé à Bob."
            # Not while the reply is on its way: only once it was heard.
            await asyncio.sleep(0.1)
            assert len(h.tap.spoken) == 2
            await h.heard()
            # Arrived close together: announced together.
            assert await h.announced(2) == "Tu viens de recevoir 2 nouveaux messages de Bob. Je te les lis ?"

    run(scenario())
    assert gateway.sent == [("Bob", "Alice: je pars")]


def test_announcement_is_not_asked_again(gateway, fast_announcer):
    async def scenario():
        async with Harness(gateway, announce=True) as h:
            await h.heard()
            _arrive(gateway, "$1", BOB)
            await h.announced(0)
            # Another question is an ordinary turn, not a non-answer to repeat the offer for.
            assert await h.says("Quel temps fait-il ?") == LLM_REPLY

    run(scenario())
    assert gateway.inbox.unread_counts([BOB]) == {BOB: 1}


def test_message_read_before_it_was_announced_is_not(gateway, fast_announcer):
    async def scenario():
        async with Harness(gateway, announce=True) as h:
            await h.worker.queue_frame(UserStartedSpeakingFrame())
            _arrive(gateway, "$1", BOB, "Coucou")
            await asyncio.sleep(0.1)
            # Not over the user.
            assert h.tap.spoken == []
            await h.worker.queue_frame(UserStoppedSpeakingFrame())
            assert "Bob a écrit : Coucou." in await h.says("Lis les messages de Bob")
            await h.heard()
            await asyncio.sleep(0.1)
            assert len(h.tap.spoken) == 1

    run(scenario())


def test_owned_contact_is_announced_to_its_owner_only(gateway, fast_announcer):
    async def scenario():
        async with Harness(gateway, announce=True) as h:
            await h.heard()
            _arrive(gateway, "$1", ELODIE, "Secret")
            await asyncio.sleep(0.1)
            assert h.tap.spoken == []
            h.person = ALICE
            answer = await h.announced(0)
            assert answer == "Alice, tu viens de recevoir un nouveau message d'Élodie. Je te le lis ?"
            assert "Élodie a écrit : Secret." in await h.says("Oui")

    run(scenario())


def test_english_announcement(gateway, fast_announcer):
    async def scenario():
        async with Harness(gateway, language="en-US", announce=True) as h:
            await h.heard()
            _arrive(gateway, "$1", BOB, "Call me")
            assert await h.announced(0) == "You just got a new message from Bob. Shall I read it?"

    run(scenario())


def test_announcement_talked_over_is_said_again(gateway, fast_announcer):
    async def scenario():
        async with Harness(gateway, announce=True) as h:
            await h.heard()
            _arrive(gateway, "$1", BOB)
            await h.announced(0)
            # The user speaks over it before it was heard: another turn, then the announcement again.
            assert await h.says("Quel temps fait-il ?", heard=False) == LLM_REPLY
            await h.heard()
            assert await h.announced(2) == "Tu viens de recevoir un nouveau message de Bob. Je te le lis ?"
            # Heard this time: not said a third time.
            await h.heard()
            await asyncio.sleep(0.1)
            assert len(h.tap.spoken) == 3

    run(scenario())
