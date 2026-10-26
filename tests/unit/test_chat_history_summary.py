# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Chat history summarization: summaries run in batches, not after every turn."""

import asyncio

from pipecat.processors.aggregators.llm_context import LLMContext

from examples.shared.pipeline_utils import SUMMARY_PREFIX, apply_pinned_prompt_summary, summary_batch_turns

PROMPT = {"role": "system", "content": "prompt"}


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def run_inference(self, context, max_tokens=None, system_instruction=None):
        self.calls += 1
        return f"summary {self.calls}"


def _turns(start: int, count: int) -> list[dict]:
    messages = []
    for i in range(start, start + count):
        messages += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    return messages


def _summarize(context: LLMContext, llm: FakeLLM, recent_turns: int = 10) -> None:
    asyncio.run(
        apply_pinned_prompt_summary(context=context, llm=llm, preserve_prompt_messages=1, recent_turns=recent_turns)
    )


def test_batch_is_half_the_window():
    assert summary_batch_turns(10) == 5
    assert summary_batch_turns(1) == 1


def test_no_summary_until_the_history_is_a_batch_past_the_window():
    llm = FakeLLM()
    context = LLMContext([PROMPT, *_turns(0, 14)])
    _summarize(context, llm)
    assert llm.calls == 0
    assert len(context.get_messages()) == 29


def test_summary_folds_the_batch_and_keeps_the_recent_turns():
    llm = FakeLLM()
    context = LLMContext([PROMPT, *_turns(0, 15)])
    _summarize(context, llm)
    messages = context.get_messages()
    assert llm.calls == 1
    assert messages[0] == PROMPT
    assert messages[1] == {"role": "user", "content": f"{SUMMARY_PREFIX}summary 1"}
    assert messages[2:] == _turns(5, 10)


def test_the_summary_message_is_not_counted_as_a_turn():
    llm = FakeLLM()
    context = LLMContext([PROMPT, *_turns(0, 15)])
    _summarize(context, llm)
    # Four more turns stay below the next batch: no summary after each one.
    for i in range(15, 19):
        context.add_messages(_turns(i, 1))
        _summarize(context, llm)
    assert llm.calls == 1
    context.add_messages(_turns(19, 1))
    _summarize(context, llm)
    assert llm.calls == 2
    assert context.get_messages()[2:] == _turns(10, 10)
