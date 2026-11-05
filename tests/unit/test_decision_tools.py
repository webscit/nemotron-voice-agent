# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D105, D107

"""Decision-controlled client tools: catalog, System One client, and the processor in a pipeline."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from pipecat.frames.frames import (
    EndFrame,
    Frame,
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

from examples.multilingual.tools import build_client_tools, without_tools
from examples.shared.decision_tools.client import DecisionError, SystemOneClient
from examples.shared.decision_tools.config import (
    MAX_CHOICES,
    NONE_CHOICE,
    DecisionModelConfig,
    DecisionTool,
    active_decision_tools,
    load_decision_tools,
)
from examples.shared.decision_tools.processor import (
    DECISION_CALL_PREFIX,
    WARM_UP_STATE,
    WARM_UP_TIMEOUT_SECS,
    DecisionToolsProcessor,
    build_state,
    last_user_text,
)
from examples.shared.intent_engine.processor import INTENT_REPLY_METADATA

URL = "http://ollama.test:11434/v1/systemone"
# The play_emotion enum of the Reachy Mini conversation app (tools/play_emotion.py).
REACHY_EMOTIONS = [
    *("random", "happy", "excited", "loving", "grateful", "success", "thinking", "attentive", "confused"),
    *("uncertain", "sad", "downcast", "lonely", "angry", "irritated", "displeased", "disgusted", "scared"),
    *("anxious", "surprised", "amazed", "calming", "relief", "impatient", "embarrassed", "bored", "tired"),
    *("sleepy", "yes", "yes_understanding", "no", "no_sad", "no_excited", "no_firm", "welcoming", "greeting"),
    *("goodbye", "go_away", "helpful", "dance", "electric", "dying"),
]
PLAY_EMOTION = {
    "name": "play_emotion",
    "description": "Play an emotion",
    "parameters": {
        "type": "object",
        "properties": {"emotion": {"type": "string", "enum": REACHY_EMOTIONS}},
    },
}
MOVE_HEAD = {
    "name": "move_head",
    "description": "Move the head",
    "parameters": {"type": "object", "properties": {"direction": {"type": "string"}}},
}


def emotion_tool(**overrides) -> DecisionTool:
    fields = {
        "name": "play_emotion",
        "argument": "emotion",
        "instructions": "Which emotion?",
        "none": "No emotion",
        "criteria": {"happy": "Joy", "sad": "Sadness"},
    }
    return DecisionTool(**{**fields, **overrides})


def answer(choice: str, confidence: float = 0.9, tool: str = "play_emotion") -> dict:
    choice_answer = {"type": "choice", "choice": choice, "probabilities": {choice: 0.9}, "confidence": confidence}
    return {"answers": {tool: choice_answer}}


class CatalogTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "decision_tools.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_packaged_catalog_fits_reachy_and_the_option_limit(self) -> None:
        tool = load_decision_tools()["play_emotion"]
        self.assertEqual(tool.argument, "emotion")
        self.assertEqual(len(tool.criteria), MAX_CHOICES)
        self.assertLessEqual(set(tool.criteria), set(REACHY_EMOTIONS))
        question = tool.question()
        self.assertEqual(question["type"], "choice")
        self.assertEqual(len(question["criteria"]), MAX_CHOICES + 1)
        self.assertIn(NONE_CHOICE, question["criteria"])

    def test_malformed_entries_are_skipped(self) -> None:
        too_many = "\n".join(f"      v{i}: value {i}" for i in range(MAX_CHOICES + 1))
        path = self.write(
            "tools:\n"
            "  ok:\n    argument: a\n    instructions: q\n    none: n\n    criteria:\n      x: X\n"
            f"  too_many:\n    argument: a\n    instructions: q\n    none: n\n    criteria:\n{too_many}\n"
            "  reserved:\n    argument: a\n    instructions: q\n    none: n\n    criteria:\n      none: N\n"
            "  bad_confidence:\n    argument: a\n    instructions: q\n    none: n\n    min_confidence: 2\n"
            "    criteria:\n      x: X\n"
            "  no_none:\n    argument: a\n    instructions: q\n    criteria:\n      x: X\n"
        )
        self.assertEqual(list(load_decision_tools(path)), ["ok"])

    def test_unreadable_catalog_is_empty(self) -> None:
        self.assertEqual(load_decision_tools(Path("/does/not/exist.yaml")), {})
        self.assertEqual(load_decision_tools(self.write("- not a mapping\n")), {})

    def test_active_tools_follow_the_client_declaration(self) -> None:
        catalog = {"play_emotion": emotion_tool(criteria={"happy": "Joy", "sad": "Sadness", "unknown": "?"})}
        raw = [PLAY_EMOTION, MOVE_HEAD]

        self.assertEqual(active_decision_tools(catalog, raw, ("move_head",)), ())
        (tool,) = active_decision_tools(catalog, raw, ("play_emotion", "move_head"))
        self.assertEqual(set(tool.criteria), {"happy", "sad"})

        no_enum = {**PLAY_EMOTION, "parameters": {"type": "object", "properties": {"emotion": {"type": "string"}}}}
        (tool,) = active_decision_tools(catalog, [no_enum], ("play_emotion",))
        self.assertEqual(set(tool.criteria), {"happy", "sad", "unknown"})

        other_enum = {
            **PLAY_EMOTION,
            "parameters": {"type": "object", "properties": {"emotion": {"type": "string", "enum": ["joy"]}}},
        }
        self.assertEqual(active_decision_tools(catalog, [other_enum], ("play_emotion",)), ())

    def test_model_config_from_env(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(DecisionModelConfig.from_env().enabled)
        env = {"DECISION_MODEL_URL": f" {URL} ", "DECISION_MODEL_API_KEY": "secret-key", "DECISION_MODEL": "tev1"}
        with mock.patch.dict("os.environ", env, clear=True):
            config = DecisionModelConfig.from_env()
        self.assertTrue(config.enabled)
        self.assertEqual((config.url, config.model, config.timeout_secs), (URL, "tev1", 3.0))
        self.assertNotIn("secret-key", repr(config))
        self.assertNotIn("secret-key", json.dumps(config.snapshot()))

    def test_decision_tools_are_hidden_from_the_llm_schema(self) -> None:
        schema, names = build_client_tools([PLAY_EMOTION, MOVE_HEAD])
        kept = without_tools(schema, {"play_emotion"})
        self.assertEqual(names, ("play_emotion", "move_head"))
        (kept_tools,) = kept.custom_tools.values()
        self.assertEqual([tool["function"]["name"] for tool in kept_tools], ["move_head"])
        self.assertIsNone(without_tools(schema, {"play_emotion", "move_head"}))
        self.assertIsNone(without_tools(None, {"play_emotion"}))


class SystemOneClientTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler, **config) -> SystemOneClient:
        return SystemOneClient(DecisionModelConfig(url=URL, **config), transport=httpx.MockTransport(handler))

    async def test_request_shape_and_answer(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=answer("happy", 0.8))

        client = self.client(handler, api_key="secret-key")
        (decision,) = await client.decide("User: hi", (emotion_tool(),))
        await client.close()

        self.assertEqual((decision.tool, decision.choice, decision.confidence), ("play_emotion", "happy", 0.8))
        body = json.loads(seen[0].content)
        self.assertEqual((body["model"], body["state"]), ("d3:2b", "User: hi"))
        self.assertEqual(
            body["questions"]["play_emotion"]["criteria"], {"happy": "Joy", "sad": "Sadness", "none": "No emotion"}
        )
        self.assertEqual(seen[0].headers["authorization"], "Bearer secret-key")

    async def test_no_key_sends_no_authorization(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=answer("sad"))

        await self.client(handler).decide("x", (emotion_tool(),))
        self.assertNotIn("authorization", seen[0].headers)

    async def test_failures_raise_decision_error(self) -> None:
        for response in (
            httpx.Response(500, text="boom"),
            httpx.Response(200, text="not json"),
            httpx.Response(200, json=["list"]),
            httpx.Response(200, json={"model": "nimble"}),
        ):
            with self.subTest(response=response.content), self.assertRaises(DecisionError):
                await self.client(lambda request, r=response: r).decide("x", (emotion_tool(),))

    async def test_missing_or_malformed_answer_is_skipped(self) -> None:
        body = {"answers": {"play_emotion": {"choice": 3}}}
        self.assertEqual(await self.client(lambda r: httpx.Response(200, json=body)).decide("x", (emotion_tool(),)), [])


class FakeLLM(LLMService):
    """Answers every context frame with a fixed reply."""

    def __init__(self, reply: str = "Super nouvelle, bravo !"):
        super().__init__()
        self.reply = reply

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(LLMFullResponseStartFrame())
            for word in self.reply.split(" "):
                await self.push_frame(LLMTextFrame(f"{word} "))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


class Tap(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.frames: list[Frame] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            self.frames.append(frame)
        await self.push_frame(frame, direction)


class FakeRecorder:
    def __init__(self):
        self.events: list[dict] = []
        self.metrics: list[tuple[str, float]] = []

    def event(self, kind, *, processor=None, **data):
        self.events.append({"kind": kind, **data})

    def metric(self, name, value, *, processor=None, model=None):
        self.metrics.append((name, value))


class Harness:
    """Pipeline: fake LLM -> decision processor -> tap -> assistant aggregator."""

    def __init__(self, handler, tools=None):
        self.requests: list[dict] = []
        self.warm_ups: list[httpx.Request] = []

        async def transport(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body["state"] == WARM_UP_STATE:
                self.warm_ups.append(request)
            else:
                self.requests.append(body)
            result = handler(request)
            return await result if asyncio.iscoroutine(result) else result

        self.sent: list = []

        async def send_message(message) -> None:
            self.sent.append(message)

        self.context = LLMContext([{"role": "system", "content": "sys"}])
        self.recorder = FakeRecorder()
        self.processor = DecisionToolsProcessor(
            context=self.context,
            tools=tools or (emotion_tool(),),
            client=SystemOneClient(DecisionModelConfig(url=URL), transport=httpx.MockTransport(transport)),
            send_message=send_message,
            recorder=self.recorder,
        )
        self.tap = Tap()
        _, assistant_aggregator = LLMContextAggregatorPair(self.context)
        pipeline = Pipeline([FakeLLM(), self.processor, self.tap, assistant_aggregator])
        self.worker = PipelineWorker(pipeline, cancel_on_idle_timeout=False, enable_rtvi=False)

    async def __aenter__(self):
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(self.worker)
        self._runner_task = asyncio.create_task(runner.run())
        return self

    async def __aexit__(self, *exc):
        await self.worker.queue_frame(EndFrame())
        await asyncio.wait_for(self._runner_task, timeout=5)

    async def wait_for(self, condition, timeout: float = 3.0) -> None:
        async with asyncio.timeout(timeout):
            while not condition():
                await asyncio.sleep(0.01)

    async def user_says(self, text: str) -> None:
        self.context.add_message({"role": "user", "content": text})
        await self.worker.queue_frame(LLMContextFrame(context=self.context))


class DecisionToolsProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_triggers_the_chosen_call(self) -> None:
        async with Harness(lambda r: httpx.Response(200, json=answer("happy"))) as h:
            await h.user_says("[speaker: Alice] J'ai eu mon examen !")
            await h.wait_for(lambda: h.sent)

        self.assertEqual(h.requests[0]["state"], "User: J'ai eu mon examen !\nAssistant: Super nouvelle, bravo !")
        message = h.sent[0]
        self.assertEqual(message.type, "llm-function-call-in-progress")
        self.assertEqual((message.data.function_name, message.data.arguments), ("play_emotion", {"emotion": "happy"}))
        self.assertTrue(message.data.tool_call_id.startswith(DECISION_CALL_PREFIX))
        decision, sent = h.recorder.events
        self.assertEqual(
            (decision["kind"], decision["outcome"], decision["choice"]), ("decision_tool", "called", "happy")
        )
        # The state is kept for the redecide job, and the call's send time for the tool-call metrics.
        self.assertEqual(decision["state"], h.requests[0]["state"])
        self.assertEqual(decision["tool_call_id"], message.data.tool_call_id)
        self.assertEqual(
            (sent["kind"], sent["name"], sent["tool_call_id"]),
            ("function_call_in_progress", "play_emotion", message.data.tool_call_id),
        )
        ((name, secs),) = h.recorder.metrics
        self.assertEqual(name, "decision_secs")
        self.assertGreaterEqual(secs, 0.0)
        # The answer still reached TTS / the aggregator unchanged.
        self.assertIn(LLMFullResponseEndFrame, [type(frame) for frame in h.tap.frames])

    async def test_none_low_confidence_and_unknown_choices_send_nothing(self) -> None:
        tool = emotion_tool(min_confidence=0.5)
        for body, outcome in (
            (answer("none"), "none"),
            (answer("happy", 0.2), "low_confidence"),
            (answer("random"), "none"),
        ):
            with self.subTest(outcome=outcome):
                async with Harness(lambda r, b=body: httpx.Response(200, json=b), tools=(tool,)) as h:
                    await h.user_says("Quelle heure est-il ?")
                    await h.wait_for(lambda: h.recorder.events)
                    await asyncio.sleep(0.05)
                self.assertEqual(h.sent, [])
                self.assertEqual(h.recorder.events[-1]["outcome"], outcome)

    async def test_decision_model_failure_skips_the_turn(self) -> None:
        async with Harness(lambda r: httpx.Response(503)) as h:
            await h.user_says("Bonjour")
            await h.wait_for(lambda: h.recorder.events)
        self.assertEqual(h.sent, [])
        self.assertEqual(h.recorder.events[-1]["outcome"], "error")

    async def test_intent_engine_replies_are_skipped(self) -> None:
        async with Harness(lambda r: httpx.Response(200, json=answer("happy"))) as h:
            start = LLMFullResponseStartFrame()
            start.metadata[INTENT_REPLY_METADATA] = True
            await h.worker.queue_frames([start, LLMTextFrame("Voilà"), LLMFullResponseEndFrame()])
            await h.wait_for(lambda: any(isinstance(f, LLMFullResponseEndFrame) for f in h.tap.frames))
            await asyncio.sleep(0.1)
        self.assertEqual((h.requests, h.sent), ([], []))

    async def test_decision_results_stay_out_of_the_context(self) -> None:
        async with Harness(lambda r: httpx.Response(200, json=answer("happy"))) as h:
            await h.worker.queue_frames(
                [
                    FunctionCallResultFrame(
                        function_name="play_emotion",
                        tool_call_id=f"{DECISION_CALL_PREFIX}abc",
                        arguments={"emotion": "happy"},
                        result={"status": "ok"},
                    ),
                    FunctionCallResultFrame(
                        function_name="move_head", tool_call_id="call-1", arguments={}, result={"status": "ok"}
                    ),
                ]
            )
            await h.wait_for(lambda: any(isinstance(f, FunctionCallResultFrame) for f in h.tap.frames))
            await asyncio.sleep(0.05)
        ids = [frame.tool_call_id for frame in h.tap.frames if isinstance(frame, FunctionCallResultFrame)]
        self.assertEqual(ids, ["call-1"])

    async def test_session_start_warms_up_the_model_with_a_long_timeout(self) -> None:
        async with Harness(lambda r: httpx.Response(200, json=answer("happy"))) as h:
            await h.wait_for(lambda: h.warm_ups)
            await asyncio.sleep(0.05)
        self.assertEqual(len(h.warm_ups), 1)
        self.assertEqual(h.warm_ups[0].extensions["timeout"]["read"], WARM_UP_TIMEOUT_SECS)
        # A warm-up is neither a call nor a recorded decision.
        self.assertEqual((h.sent, h.recorder.events, h.requests), ([], [], []))

    async def test_interruption_cancels_a_pending_decision(self) -> None:
        async def slow(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.5)
            return httpx.Response(200, json=answer("happy"))

        async with Harness(slow) as h:
            await h.user_says("Raconte-moi une blague")
            await h.wait_for(lambda: h.requests)
            await h.worker.queue_frame(InterruptionFrame())
            await asyncio.sleep(0.7)
        self.assertEqual(h.sent, [])


class HelperTests(unittest.TestCase):
    def test_last_user_text_reads_the_latest_user_message(self) -> None:
        context = LLMContext(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": [{"type": "text", "text": "[speaker: Bob] second"}, {"type": "image_url"}]},
                {"role": "assistant", "content": "", "tool_calls": []},
            ]
        )
        self.assertEqual(last_user_text(context), "second")
        self.assertEqual(last_user_text(LLMContext([{"role": "system", "content": "sys"}])), "")

    def test_build_state(self) -> None:
        self.assertEqual(build_state("", "  Bonjour \n toi "), "Assistant: Bonjour toi")


if __name__ == "__main__":
    unittest.main()
