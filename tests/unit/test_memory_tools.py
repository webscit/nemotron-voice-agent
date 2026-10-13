# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Server-side ``remember`` / ``forget`` memory tools."""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.adapters.schemas.tools_schema import AdapterType, ToolsSchema
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.shared.memory_tools import (
    MAX_FACT_LEN,
    MEMORY_TOOL_NAMES,
    UNKNOWN_PERSON_ERROR,
    MemoryTools,
    latest_user_text,
    with_memory_tools,
)
from examples.shared.person_memory import load_person_context
from monitoring import memories
from monitoring.store import SessionStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    monkeypatch.setattr(memories, "live_store", lambda: store)
    return store


class _Tools:
    """``MemoryTools`` on a picked person, recording prompt refreshes."""

    def __init__(self, person_id):
        self.person = asyncio.run(load_person_context(person_id)) if person_id else None
        self.refreshes = 0
        self.context = LLMContext([{"role": "user", "content": "[speaker: Alice] Retiens que j'ai un chien."}])
        self.tools = MemoryTools(
            current_person=lambda: self.person,
            refresh=self._refresh,
            session_id=lambda: "s1",
            turn_idx=lambda: 3,
            language="fr-FR",
        )

    async def _refresh(self):
        self.refreshes += 1
        self.person = await load_person_context(self.person.person["id"])

    def call(self, name, **arguments):
        results = []

        async def callback(result, **_):
            results.append(result)

        params = SimpleNamespace(arguments=arguments, result_callback=callback, context=self.context)
        asyncio.run(getattr(self.tools, name)(params))
        return results[0]


def test_with_memory_tools_keeps_client_tools_and_is_idempotent():
    assert [t.name for t in with_memory_tools(None).standard_tools] == list(MEMORY_TOOL_NAMES)
    custom = {AdapterType.OPENAI: [{"type": "function", "function": {"name": "dance"}}]}
    tools = with_memory_tools(with_memory_tools(ToolsSchema(standard_tools=[], custom_tools=custom)))
    assert [t.name for t in tools.standard_tools] == list(MEMORY_TOOL_NAMES)
    assert tools.custom_tools == custom


def test_latest_user_text_strips_tags():
    context = LLMContext(
        [
            {"role": "user", "content": "old"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[speaker: Bob; also in view: Alice]"},
                    {"type": "text", "text": "hi"},
                ],
            },
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert latest_user_text(context) == "hi"
    assert latest_user_text(None) == ""


def test_remember_saves_an_active_unreviewed_conversation_memory(store):
    alice = memories.create_person(store, "Alice")
    t = _Tools(alice["id"])
    assert t.call("remember", fact=" Alice a un  chien. ") == {"status": "saved", "fact": "Alice a un chien."}
    assert t.refreshes == 1 and [m["text"] for m in t.person.memories] == ["Alice a un chien."]

    row = memories.list_memories(store, status="open")["memories"][0]
    assert row["status"] == "active" and row["reviewed_at"] is None
    assert row["source"] == "live:s1" and row["origin"] == "conversation" and row["language"] == "fr-FR"
    assert [(e["session_id"], e["turn_idx"], e["quote"]) for e in row["evidence"]] == [
        ("s1", 3, "Retiens que j'ai un chien.")
    ]

    assert t.call("remember", fact="alice a un chien.")["status"] == "already remembered"
    assert "error" in t.call("remember", fact="  ")
    assert "error" in t.call("remember", fact="x" * (MAX_FACT_LEN + 1))
    assert t.refreshes == 1


def test_forget_marks_the_best_match_forgotten(store):
    alice = memories.create_person(store, "Alice")
    for text in ("Alice a un chien, Mochi.", "Alice promène son chien le matin.", "Alice aime le thé."):
        memories.add_memory(
            store, {"person_id": alice["id"], "text": text, "status": "active", "source": "dream:x"}, []
        )
    t = _Tools(alice["id"])

    missing = t.call("forget", query="café crème")
    assert "error" in missing and t.refreshes == 0

    result = t.call("forget", query="chien Mochi")
    assert result == {
        "status": "forgotten",
        "forgotten": "Alice a un chien, Mochi.",
        "other_matches": ["Alice promène son chien le matin."],
    }
    forgotten = memories.list_memories(store, status="forgotten")["memories"][0]
    assert forgotten["reviewed_by"] == "live:s1"
    assert t.refreshes == 1 and "Mochi" not in t.person.prompt_replacements()["memories"]
    assert "error" in t.call("forget", query=" ")


def test_tools_refuse_an_unknown_person_or_store(store, monkeypatch):
    t = _Tools(None)
    assert t.call("remember", fact="Quelqu'un a un chien.") == {"error": UNKNOWN_PERSON_ERROR}
    assert t.call("forget", query="chien") == {"error": UNKNOWN_PERSON_ERROR}
    assert memories.list_memories(store, status="all")["total"] == 0

    alice = memories.create_person(store, "Alice")
    t = _Tools(alice["id"])
    monkeypatch.setattr(memories, "live_store", lambda: None)
    assert "error" in t.call("remember", fact="Alice a un chien.")

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(memories, "live_store", boom)
    assert t.call("forget", query="chien") == {"error": "The memory could not be forgotten."}
