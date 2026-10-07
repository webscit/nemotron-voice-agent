# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D105, D107

"""Processor-level tests: the engine inside a real aggregator pipeline with a fake LLM."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
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

from examples.multilingual.multilingual_processor import PerTurnReminderProcessor
from examples.multilingual.tool_handlers import ClientToolResultBridge, build_client_tool_handler
from examples.shared.intent_engine import targets
from examples.shared.intent_engine.config import IntentEngineConfig
from examples.shared.intent_engine.matcher import IntentMatcher, build_slot_lists
from examples.shared.intent_engine.processor import IntentEngineProcessor, build_intent_engine
from examples.shared.intent_engine.targets import ClientToolMap, ClientToolTarget, HomeAssistantTarget

REMINDER = "Reminder: reply only in French."
SYNC_PAYLOAD = {
    "entities": [
        {"id": "light.ceiling_lights", "name": "Ceiling Lights", "domain": "light", "device_class": None, "area": None}
    ],
    "areas": [{"id": "salon", "name": "Salon"}],
    "floors": [],
}
ACTION_DONE = {
    "speech": {},
    "card": {},
    "language": "fr",
    "response_type": "action_done",
    "data": {"success": [{"name": "Ceiling Lights", "type": "entity", "id": "light.ceiling_lights"}], "failed": []},
}
MATCH_FAILED = {
    "speech": {"plain": {"speech": "<MatchFailedError result=MatchTargetsResult(is_match=False)>"}},
    "response_type": "error",
    "data": {"code": "failed_to_handle"},
}


class FakeLLM(LLMService):
    """Answers every context frame with a fixed reply and records what it was asked."""

    def __init__(self):
        super().__init__()
        self.requests: list[list[dict]] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            self.requests.append(list(frame.context.get_messages()))
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame("Réponse du LLM."))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class Tap(FrameProcessor):
    """Collects the downstream frames that reach the end of the pipeline."""

    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)

    def of_type(self, frame_type: type) -> list[Frame]:
        return [frame for frame in self.frames if isinstance(frame, frame_type)]


class FakeRecorder:
    def __init__(self):
        self.events: list[dict] = []
        self.metrics: list[str] = []

    def event(self, kind, *, processor=None, **data):
        self.events.append({"kind": kind, **data})

    def metric(self, name, value, *, processor=None, model=None):
        self.metrics.append(name)


class Harness:
    """Pipeline: reminder -> engine -> fake LLM -> tap -> bridge -> assistant aggregator."""

    def __init__(
        self,
        *,
        handler=None,
        tools=("move_head", "volume_control"),
        dry_run=False,
        timeout_secs=1.0,
        sentences_dir=None,
    ):
        targets._SYNC_CACHE.clear()
        self.http_requests: list[tuple[str, dict]] = []
        self.handler = handler or (lambda path, body: httpx.Response(200, json=ACTION_DONE))

        def transport(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if request.url.path == "/api/template" and "variables" not in body:
                return httpx.Response(200, text=json.dumps(SYNC_PAYLOAD))
            self.http_requests.append((request.url.path, body))
            return self.handler(request.url.path, body)

        config = IntentEngineConfig(
            enabled=True,
            home_assistant_url="http://ha.test",
            home_assistant_token="secret-token",
            area="Salon",
            dry_run=dry_run,
            timeout_secs=timeout_secs,
            wake_words=("Reachy", "Richie"),
            **({"sentences_dir": sentences_dir} if sentences_dir else {}),
        )
        tool_map = ClientToolMap.load(config.sentences_dir)
        matcher = IntentMatcher(
            "fr-FR",
            allowed_intents=config.allowed_intents,
            sentences_dir=config.sentences_dir,
            wake_words=config.wake_words,
            area=config.area,
            keep_custom_intent=lambda name: (
                tool_map.tool(name) in tools if name in tool_map else name in config.allowed_intents
            ),
        )
        self.context = LLMContext([{"role": "system", "content": "sys"}])
        self.llm = FakeLLM()
        self.bridge = ClientToolResultBridge()
        for tool in tools:
            self.llm.register_function(tool, build_client_tool_handler(0.4, self.bridge), cancel_on_interruption=True)
        self.recorder = FakeRecorder()
        self.engine = IntentEngineProcessor(
            config=config,
            language="fr-FR",
            matcher=matcher,
            context=self.context,
            home_assistant=HomeAssistantTarget(
                config.home_assistant_url,
                config.home_assistant_token,
                templates=matcher,
                timeout_secs=timeout_secs,
                area=config.area,
                blocked_domains=config.blocked_domains,
                transport=httpx.MockTransport(transport),
            ),
            client_tools=ClientToolTarget(self.llm, self.context, tool_map, templates=matcher, timeout_secs=1.5),
            tool_map=tool_map,
            recorder=self.recorder,
        )
        self.tap = Tap()
        _, assistant_aggregator = LLMContextAggregatorPair(self.context)
        pipeline = Pipeline(
            [PerTurnReminderProcessor(REMINDER), self.engine, self.llm, self.tap, self.bridge, assistant_aggregator]
        )
        self.worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)
        self._runner_task: asyncio.Task | None = None

    async def __aenter__(self):
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._runner_task = asyncio.create_task(runner.run())
        await self.wait_for(lambda: self.engine._home_assistant.slot_lists is not None)
        return self

    async def __aexit__(self, *exc):
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._runner_task, timeout=5)

    async def wait_for(self, condition, timeout: float = 3.0) -> None:
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.01)

    async def user_says(self, text: str) -> None:
        """Emit what the user aggregator does at the end of a user turn."""
        self.context.add_message({"role": "user", "content": text})
        await self.worker.queue_frame(LLMContextFrame(context=self.context))

    async def settle(self) -> None:
        """Let the reply reach the assistant aggregator (it stores it on the response end frame)."""
        if self.spoken():
            await self.wait_for(lambda: self.roles()[-1] == "assistant")
        await asyncio.sleep(0.15)

    def spoken(self) -> list[str]:
        return [frame.text for frame in self.tap.of_type(LLMTextFrame)]

    def roles(self) -> list[str]:
        return [message["role"] for message in self.context.get_messages()]


class IntentEngineProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_miss_goes_to_llm_unchanged_with_reminder(self) -> None:
        async with Harness() as h:
            await h.user_says("Que connais-tu sur la ville de Rennes ?")
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(h.llm.requests[0][-1]["content"], f"Que connais-tu sur la ville de Rennes ?\n\n{REMINDER}")
            self.assertEqual(h.http_requests, [])
            self.assertEqual(
                h.recorder.events[-1], {"kind": "intent_engine", "handled_by": "llm", "reason": "no_match"}
            )
            self.assertIn("intent_match_secs", h.recorder.metrics)

    async def test_hit_skips_llm_and_speaks_packaged_response(self) -> None:
        async with Harness() as h:
            # The reminder processor rewrites the frame's context copy; matching must not see it.
            await h.user_says("Bonjour Richie, allume Ceiling Lights")
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Allumé"])
            self.assertEqual(h.llm.requests, [])
            path, body = h.http_requests[0]
            self.assertEqual(path, "/api/intent/handle")
            self.assertEqual(body["name"], "HassTurnOn")
            self.assertEqual(body["data"], {"name": "Ceiling Lights", "preferred_area_id": "salon"})
            self.assertEqual(body["language"], "fr")
            self.assertEqual(body["assistant"], "conversation")
            # Same frames as an LLM reply, and the reply is stored as an assistant message.
            kinds = [
                type(f) for f in h.tap.frames if isinstance(f, (LLMFullResponseStartFrame, LLMFullResponseEndFrame))
            ]
            self.assertEqual(kinds, [LLMFullResponseStartFrame, LLMFullResponseEndFrame])
            self.assertEqual(h.context.get_messages()[-1], {"role": "assistant", "content": "Allumé"})
            self.assertEqual(h.context.get_messages()[-2]["content"], "Bonjour Richie, allume Ceiling Lights")
            event = h.recorder.events[-1]
            self.assertEqual(
                {k: event[k] for k in ("handled_by", "intent", "target", "response_type")},
                {
                    "handled_by": "intent",
                    "intent": "HassTurnOn",
                    "target": "home_assistant",
                    "response_type": "action_done",
                },
            )
            self.assertIn("intent_target_secs", h.recorder.metrics)

    async def test_reminder_does_not_accumulate_over_turns(self) -> None:
        async with Harness() as h:
            await h.user_says("allume Ceiling Lights")
            await h.wait_for(lambda: len(h.spoken()) == 1)
            await h.settle()  # the engine's reply must be stored before the next user turn
            await h.user_says("Raconte une blague")
            await h.wait_for(lambda: len(h.spoken()) == 2)
            await h.settle()
            self.assertEqual(h.spoken(), ["Allumé", "Réponse du LLM."])
            stored = json.dumps(h.context.get_messages(), ensure_ascii=False)
            self.assertNotIn(REMINDER, stored)
            # The LLM sees the engine's earlier reply as normal history, and one reminder.
            sent = h.llm.requests[0]
            self.assertEqual([m["role"] for m in sent], ["system", "user", "assistant", "user"])
            self.assertEqual(json.dumps(sent, ensure_ascii=False).count(REMINDER), 1)

    async def test_error_response_speaks_packaged_error_not_returned_speech(self) -> None:
        async with Harness(handler=lambda path, body: httpx.Response(200, json=MATCH_FAILED)) as h:
            await h.user_says("allume Ceiling Lights")
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Une erreur est intervenue pendant le traitement"])
            self.assertEqual(h.llm.requests, [])
            self.assertEqual(h.recorder.events[-1]["response_type"], "error")

    async def test_unreachable_target_falls_through_to_llm(self) -> None:
        def refuse(path, body):
            raise httpx.ConnectError("connection refused")

        async with Harness(handler=refuse) as h:
            await h.user_says("allume Ceiling Lights")
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(h.llm.requests[0][-1]["content"], f"allume Ceiling Lights\n\n{REMINDER}")
            self.assertEqual(h.recorder.events[-1]["reason"], "target_unavailable")
            self.assertEqual(h.recorder.events[-1]["handled_by"], "llm")

    async def test_http_500_and_timeout_fall_through_to_llm(self) -> None:
        async with Harness(handler=lambda path, body: httpx.Response(500, text="Server got itself in trouble")) as h:
            await h.user_says("allume Ceiling Lights")
            await h.wait_for(lambda: h.spoken())
            self.assertEqual(h.spoken(), ["Réponse du LLM."])

        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(5)
            return httpx.Response(200, json=ACTION_DONE)

        async with Harness(timeout_secs=0.2) as h:
            h.engine._home_assistant._client = httpx.AsyncClient(
                base_url="http://ha.test", transport=httpx.MockTransport(slow)
            )
            await h.user_says("allume Ceiling Lights")
            await h.wait_for(lambda: h.spoken())
            self.assertEqual(h.spoken(), ["Réponse du LLM."])

    async def test_unexpected_target_failure_still_answers_through_llm(self) -> None:
        async with Harness() as h:

            async def crash(request):
                raise RuntimeError("boom")

            h.engine._home_assistant.handle = crash
            await h.user_says("allume Ceiling Lights")
            await h.wait_for(lambda: h.spoken())
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(h.recorder.events[-1]["reason"], "engine_error")

    async def test_dry_run_logs_and_leaves_turn_to_llm(self) -> None:
        async with Harness(dry_run=True) as h:
            await h.user_says("allume Ceiling Lights")
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(h.http_requests, [])
            event = h.recorder.events[-1]
            self.assertEqual((event["handled_by"], event["intent"], event["reason"]), ("llm", "HassTurnOn", "dry_run"))

    async def test_state_questions_other_than_single_entity_go_to_llm(self) -> None:
        async with Harness() as h:
            await h.user_says("Ceiling Lights est-elle allumée ?")
            await h.wait_for(lambda: h.spoken())
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(h.http_requests, [])
            self.assertEqual(h.recorder.events[-1]["reason"], "state_question")

    async def test_interruption_cancels_pending_intent(self) -> None:
        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.5)
            return httpx.Response(200, json=ACTION_DONE)

        async with Harness() as h:
            h.engine._home_assistant._client = httpx.AsyncClient(
                base_url="http://ha.test", transport=httpx.MockTransport(slow)
            )
            await h.user_says("allume Ceiling Lights")
            await asyncio.sleep(0.1)
            await h.worker.queue_frame(InterruptionFrame())
            await asyncio.sleep(0.7)
            self.assertEqual(h.spoken(), [])

    async def test_blocked_domain_is_never_sent_and_area_wide_request_is_constrained(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "fr").mkdir()
            (Path(tmp) / "fr" / "risky.yaml").write_text(
                "language: fr\n"
                "intents:\n"
                "  HassTurnOff:\n"
                "    data:\n"
                "      - sentences: ['ouvre toutes les serrures']\n"
                "        slots: {domain: lock}\n"
                "      - sentences: ['coupe les alarmes et les serrures']\n"
                "        slots: {domain: [alarm_control_panel, lock]}\n"
                "      - sentences: ['éteins tout ici']\n"
                "        requires_context: {area: {slot: true}}\n",
                encoding="utf-8",
            )
            async with Harness(sentences_dir=Path(tmp)) as h:
                for text in ("Ouvre toutes les serrures", "Coupe les alarmes et les serrures"):
                    spoken = len(h.spoken())
                    await h.user_says(text)
                    await h.wait_for(lambda spoken=spoken: len(h.spoken()) > spoken)
                    event = h.recorder.events[-1]
                    self.assertEqual(
                        (event["handled_by"], event["intent"], event["reason"]),
                        ("llm", "HassTurnOff", "blocked_domain"),
                    )
                    await h.settle()
                self.assertEqual(h.http_requests, [])
                self.assertEqual(set(h.spoken()), {"Réponse du LLM."})

                # An area-wide request goes to Home Assistant, limited to unblocked domains.
                await h.user_says("Éteins tout ici")
                await h.wait_for(lambda: "Éteint" in h.spoken())
                _, body = h.http_requests[0]
                self.assertEqual(body["data"], {"area": "Salon", "preferred_area_id": "salon", "domain": ["light"]})

    async def test_non_user_context_frames_pass_through(self) -> None:
        async with Harness() as h:
            h.context.add_message({"role": "user", "content": "allume Ceiling Lights"})
            h.context.add_message({"role": "assistant", "content": "D'accord"})
            await h.worker.queue_frame(LLMContextFrame(context=h.context))
            await h.wait_for(lambda: h.spoken())
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(h.http_requests, [])


class IntentEngineClientToolTests(unittest.IsolatedAsyncioTestCase):
    async def _in_progress(self, h: Harness) -> FunctionCallInProgressFrame:
        await h.wait_for(lambda: h.tap.of_type(FunctionCallInProgressFrame))
        return h.tap.of_type(FunctionCallInProgressFrame)[0]

    async def _client_answers(self, h: Harness, call: FunctionCallInProgressFrame, result) -> None:
        """Deliver the result the way RTVIProcessor does: downstream from the pipeline head."""
        await h.worker.queue_frame(
            FunctionCallResultFrame(
                function_name=call.function_name,
                tool_call_id=call.tool_call_id,
                arguments=call.arguments,
                result=result,
            )
        )

    async def test_client_tool_hit_waits_for_result_then_speaks_without_llm(self) -> None:
        async with Harness() as h:
            await h.user_says("Est-ce que tu peux regarder à droite ?")
            call = await self._in_progress(h)
            self.assertEqual((call.function_name, call.arguments), ("move_head", {"direction": "right"}))
            await h.settle()
            self.assertEqual(h.spoken(), [])  # no optimistic reply before the client answered

            await self._client_answers(h, call, {"status": "ok"})
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Voilà"])
            self.assertEqual(h.llm.requests, [])  # the tool result did not re-trigger the LLM

            # The call, its result, and the spoken reply are all in the LLM context.
            self.assertEqual(h.roles(), ["system", "user", "assistant", "tool", "assistant"])
            messages = h.context.get_messages()
            self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "move_head")
            self.assertEqual(json.loads(messages[3]["content"]), {"status": "ok"})
            self.assertEqual(messages[4], {"role": "assistant", "content": "Voilà"})
            event = h.recorder.events[-1]
            self.assertEqual((event["handled_by"], event["target"]), ("intent", "client_tool"))

            # A later LLM turn still works and sees a well-formed history.
            await h.user_says("Merci beaucoup pour ton aide")
            await h.wait_for(lambda: len(h.spoken()) == 2)
            self.assertEqual(h.spoken()[-1], "Réponse du LLM.")

    async def test_client_tool_error_result_speaks_packaged_error(self) -> None:
        async with Harness() as h:
            await h.user_says("Mets ton volume à soixante-dix")
            call = await self._in_progress(h)
            self.assertEqual(call.arguments, {"device": "speaker", "level": 70})
            await self._client_answers(h, call, {"error": "Traceback: device busy"})
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Une erreur est intervenue pendant le traitement"])
            self.assertEqual(h.llm.requests, [])

    async def test_client_timeout_hands_turn_to_llm_once(self) -> None:
        async with Harness() as h:
            await h.user_says("regarde à gauche")
            await self._in_progress(h)
            # The registered handler times out (0.4 s) and reports an error result itself.
            await h.wait_for(lambda: h.spoken(), timeout=3)
            await asyncio.sleep(1.8)
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(len(h.llm.requests), 1)
            self.assertEqual(h.recorder.events[-1]["reason"], "target_unavailable")

    async def test_relative_volume_reads_then_sets_without_llm(self) -> None:
        async with Harness() as h:
            await h.user_says("Parle un peu plus fort.")
            read = await self._in_progress(h)
            self.assertEqual((read.function_name, read.arguments), ("volume_control", {"device": "speaker"}))
            await self._client_answers(h, read, {"device": "speaker", "volume": 95})
            await h.wait_for(lambda: len(h.tap.of_type(FunctionCallInProgressFrame)) == 2)
            write = h.tap.of_type(FunctionCallInProgressFrame)[1]
            self.assertEqual(write.arguments, {"device": "speaker", "level": 100})  # 95 + 10, clamped
            self.assertEqual(h.spoken(), [])
            await self._client_answers(h, write, {"device": "speaker", "volume": 100})
            await h.wait_for(lambda: h.spoken())
            await h.settle()
            self.assertEqual(h.spoken(), ["Volume à 100 pour cent"])
            self.assertEqual(h.llm.requests, [])
            self.assertEqual(h.roles(), ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"])

    async def test_relative_volume_with_unusable_reading_speaks_error(self) -> None:
        async with Harness() as h:
            await h.user_says("moins fort")
            read = await self._in_progress(h)
            await self._client_answers(h, read, {"device": "speaker", "volume": None})
            await h.wait_for(lambda: h.spoken())
            self.assertEqual(h.spoken(), ["Une erreur est intervenue pendant le traitement"])
            self.assertEqual(len(h.tap.of_type(FunctionCallInProgressFrame)), 1)
            self.assertEqual(h.llm.requests, [])

    async def test_deferred_and_undeclared_intents_go_to_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "fr").mkdir()
            (Path(tmp) / "fr" / "robot.yaml").write_text(
                "language: fr\nintents:\n  RobotWave:\n    data:\n      - sentences: ['fais coucou']\n",
                encoding="utf-8",
            )
            (Path(tmp) / "client_tools.yaml").write_text(
                "intents:\n  RobotWave:\n    tool: move_head\n    defer_to_llm: true\n", encoding="utf-8"
            )
            async with Harness(sentences_dir=Path(tmp)) as h:
                await h.user_says("Fais coucou")
                await h.wait_for(lambda: h.spoken())
                self.assertEqual(h.spoken(), ["Réponse du LLM."])
                self.assertEqual(h.recorder.events[-1]["reason"], "deferred_to_llm")
                self.assertEqual(h.tap.of_type(FunctionCallInProgressFrame), [])

        async with Harness(tools=("move_head",)) as h:
            # go_to_sleep was not declared by the client, so its intent is inactive.
            await h.user_says("Va dormir.")
            await h.wait_for(lambda: h.spoken())
            self.assertEqual(h.spoken(), ["Réponse du LLM."])
            self.assertEqual(h.recorder.events[-1]["reason"], "no_match")


class BuildIntentEngineTests(unittest.IsolatedAsyncioTestCase):
    def build(self, config: IntentEngineConfig, tools=("move_head",), language="fr-FR"):
        return build_intent_engine(
            config, language=language, context=LLMContext([]), llm=FakeLLM(), client_tool_names=tools
        )

    async def test_disabled_by_default(self) -> None:
        self.assertIsNone(self.build(IntentEngineConfig()))

    async def test_client_tools_only_without_home_assistant(self) -> None:
        engine = self.build(IntentEngineConfig(enabled=True))
        self.assertIsNotNone(engine)
        self.assertIsNone(engine._home_assistant)
        self.assertIsNotNone(engine._client_tools)
        self.assertEqual(engine._matcher.custom_intent_names, ["ReachyMoveHead"])
        # Stock Home Assistant sentences are not loaded, even with entity names at hand.
        lists = build_slot_lists([{"name": "Lampe", "domain": "light"}], [], [])
        self.assertIsNone(engine._matcher.match("Allume Lampe", lists))

    async def test_nothing_to_match_means_no_engine(self) -> None:
        self.assertIsNone(self.build(IntentEngineConfig(enabled=True), tools=()))
        self.assertIsNone(self.build(IntentEngineConfig(enabled=True), language="xx-YY"))

    async def test_home_assistant_target_gets_blocked_domains(self) -> None:
        config = IntentEngineConfig(enabled=True, home_assistant_url="http://ha.test", home_assistant_token="t")
        engine = self.build(config, tools=())
        self.assertEqual(engine._home_assistant._blocked_domains, ("lock", "alarm_control_panel"))
        self.assertIsNone(engine._client_tools)
        await engine._home_assistant.aclose()


class ClientToolsOnlyPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_tool_turn_and_llm_turn_without_home_assistant(self) -> None:
        context = LLMContext([{"role": "system", "content": "sys"}])
        llm, bridge, tap = FakeLLM(), ClientToolResultBridge(), Tap()
        llm.register_function("go_to_sleep", build_client_tool_handler(0.4, bridge), cancel_on_interruption=True)
        engine = build_intent_engine(
            IntentEngineConfig(enabled=True, wake_words=("Richie",)),
            language="fr-FR",
            context=context,
            llm=llm,
            client_tool_names=("go_to_sleep",),
            client_tool_timeout_secs=0.4,
        )
        _, assistant_aggregator = LLMContextAggregatorPair(context)
        worker = PipelineWorker(
            Pipeline([engine, llm, tap, bridge, assistant_aggregator]), cancel_on_idle_timeout=False, enable_rtvi=False
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run())

        async def wait_for(condition):
            async with asyncio.timeout(3):
                while not condition():
                    await asyncio.sleep(0.01)

        context.add_message({"role": "user", "content": "Merci Richie, tu peux aller dormir."})
        await worker.queue_frame(LLMContextFrame(context=context))
        await wait_for(lambda: tap.of_type(FunctionCallInProgressFrame))
        call = tap.of_type(FunctionCallInProgressFrame)[0]
        await worker.queue_frame(
            FunctionCallResultFrame(
                function_name=call.function_name, tool_call_id=call.tool_call_id, arguments={}, result={"status": "ok"}
            )
        )
        await wait_for(lambda: tap.of_type(LLMTextFrame))
        self.assertEqual([f.text for f in tap.of_type(LLMTextFrame)], ["Bonne nuit"])
        self.assertEqual(llm.requests, [])

        context.add_message({"role": "user", "content": "Allume la lampe du salon"})
        await worker.queue_frame(LLMContextFrame(context=context))
        await wait_for(lambda: len(tap.of_type(LLMTextFrame)) == 2)
        self.assertEqual(tap.of_type(LLMTextFrame)[-1].text, "Réponse du LLM.")
        await worker.queue_frame(EndFrame())
        await asyncio.wait_for(task, timeout=5)


class SentencesDirTests(unittest.TestCase):
    def test_packaged_sentences_exist(self) -> None:
        directory = IntentEngineConfig().sentences_dir
        self.assertTrue((Path(directory) / "client_tools.yaml").is_file())
        self.assertTrue((Path(directory) / "fr" / "reachy_mini.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
