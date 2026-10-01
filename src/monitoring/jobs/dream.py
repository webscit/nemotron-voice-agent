# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dream``: extract long-term memories about people from a recorded session.

1. Resolve who spoke in each user turn (turn override, else the session person);
   sessions without an attributed person are skipped.
2. Build the transcript: human reference > reference-model transcript > live ASR
   text for user turns, plus the assistant replies.
3. Ask an OpenAI-compatible LLM (by default the session's own LLM) for durable
   facts, given the person's current memories so it can skip duplicates and
   supersede outdated ones.
4. Store each fact as a memory: ``active`` (used in live prompts right away) at or
   above ``auto_use_threshold``, ``proposed`` below it or when it would replace a
   memory a human already reviewed. Reviewed memories are never changed here.

Re-running the job first supersedes the unreviewed memories whose only evidence
is this session, then extracts again.
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from loguru import logger

from monitoring import memories
from monitoring.jobs.base import Job, JobContext, register
from monitoring.jobs.reasr import reasr_endpoints

DEFAULTS: dict[str, Any] = {
    # Memories at or above this confidence are used live before review.
    "auto_use_threshold": 0.8,
    # Override the session LLM: {base_url, model, extra}.
    "llm": None,
    # Let the model think before answering (better extraction, slower).
    "reasoning": True,
    "temperature": 0.2,
    "max_tokens": 8192,
    "ready_timeout_secs": 900,
    "request_timeout_secs": 600,
}

CATEGORIES = ("identity", "preference", "relationship", "routine", "event", "health", "other")

SYSTEM_PROMPT = """You maintain the long-term memory of a voice assistant about the people it talks to.
Read the conversation and extract durable facts about the people speaking to the assistant:
identity (name, age, job, where they live), preferences, relationships, routines,
upcoming or past events worth remembering, health information they shared, other stable facts.

Rules:
- Only facts stated or clearly implied by the person themselves, never by the assistant.
- Skip small talk, one-off requests (weather, timers, jokes) and anything true only for this conversation.
- One short third-person sentence per fact, written in {language_name}, naming the person (e.g. "Alice is vegetarian.").
- Do not repeat a fact already in the existing memories. When the conversation contradicts or updates
  an existing memory, output the new fact with "supersedes" set to that memory id.
- confidence: 0.9+ when stated explicitly, 0.5-0.8 when inferred, below 0.5 when unsure.
- turns: the user turn numbers the fact comes from; quote: the shortest supporting excerpt.

Answer with JSON only, no prose:
{{"memories": [{{"person_id": "...", "text": "...", "category": "one of {categories}",
"confidence": 0.0, "turns": [1], "quote": "...", "supersedes": null}}]}}
Return {{"memories": []}} when there is nothing worth remembering."""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def dream_config(config: dict[str, Any]) -> dict[str, Any]:
    """The ``dream`` section of ``dreamer.yaml`` merged over the defaults."""
    return {**DEFAULTS, **(config.get("dream") or {})}


def parse_memories(raw: str) -> list[dict[str, Any]]:
    """Extract the ``memories`` list from a model answer (tolerates think tags and prose)."""
    text = _THINK_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in LLM answer: {text[:500]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in LLM answer ({exc}): {text[:500]!r}") from None
    items = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("LLM answer has no 'memories' list")
    return [item for item in items if isinstance(item, dict)]


def _language_name(code: str) -> str:
    base = (code or "en").split("-")[0].lower()
    return {
        "en": "English",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "hi": "Hindi",
        "ja": "Japanese",
        "zh": "Chinese",
    }.get(base, code or "English")


class LlmClient:
    """Minimal blocking chat client for OpenAI-compatible servers."""

    def __init__(self, *, base_url: str, model: str, extra: dict[str, Any] | None, cfg: dict[str, Any]):
        """Create the client; the API key comes from ``NVIDIA_API_KEY`` (dummy for local servers)."""
        from openai import OpenAI

        from utils import nvidia_api_key

        self.model = model
        self.cfg = cfg
        self.extra = copy.deepcopy(extra or {})
        # Reasoning on/off through the chat template, like the pipeline's summarizer.
        extra_body = self.extra.setdefault("extra_body", {})
        if isinstance(extra_body, dict):
            kwargs = extra_body.setdefault("chat_template_kwargs", {})
            if isinstance(kwargs, dict):
                kwargs["enable_thinking"] = bool(cfg["reasoning"])
        self._client = OpenAI(base_url=base_url, api_key=nvidia_api_key(), timeout=cfg["request_timeout_secs"])

    def wait_ready(self, ctx: JobContext) -> None:
        """Poll ``/models`` until the server answers (yielding to live sessions)."""
        deadline = time.monotonic() + float(self.cfg["ready_timeout_secs"])
        while True:
            ctx.check_preempted()
            try:
                models = [m.id for m in self._client.models.list()]
                self.model = self.model or (models[0] if models else "")
                return
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(5.0)

    def complete(self, system: str, user: str) -> str:
        """One chat completion; returns the answer text."""
        allowed = {"extra_body", "extra_headers", "extra_query", "top_p", "frequency_penalty", "presence_penalty"}
        kwargs = {k: v for k, v in self.extra.items() if k in allowed}
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=float(self.cfg["temperature"]),
            max_tokens=int(self.cfg["max_tokens"]),
            **kwargs,
        )
        return response.choices[0].message.content or ""


def make_client(session_config: dict[str, Any], cfg: dict[str, Any]) -> LlmClient:
    """Client for the configured override, else the session's own LLM."""
    llm = {**(session_config.get("llm") or {}), **(cfg.get("llm") or {})}
    if not llm.get("base_url"):
        raise RuntimeError("no LLM endpoint: the session config has no llm.base_url and dream.llm is not set")
    return LlmClient(base_url=llm["base_url"], model=llm.get("model") or "", extra=llm.get("extra"), cfg=cfg)


def _ints(values: Any) -> list[int]:
    out = []
    for value in values if isinstance(values, list) else []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _reference_name(config: dict[str, Any]) -> str | None:
    reference, _ = reasr_endpoints(config)
    return reference.name if reference else None


def session_transcript(ctx: JobContext, session_id: str, speakers: dict[str, Any]) -> list[dict[str, Any]]:
    """Turns with the best available user text and who said it."""
    reference = _reference_name(ctx.config)
    best: dict[int, tuple[int, float, str]] = {}  # turn -> (priority, created_at, text)
    for row in ctx.store.annotations_for(session_id, kind="transcript"):
        idx = int(row["target_id"].rsplit(":", 1)[1])
        text = (row["value"] or {}).get("text", "")
        priority = 2 if row["source"].startswith("human:") else 1 if row["source"] == reference else 0
        if priority and (idx not in best or (priority, row["created_at"]) > best[idx][:2]):
            best[idx] = (priority, row["created_at"], text)
    turns = []
    for turn in sorted(ctx.store.rows("turns", session_id), key=lambda t: t["idx"]):
        idx = turn["idx"]
        user_text = best[idx][2] if idx in best else (turn["user_text"] or "")
        turns.append(
            {
                "idx": idx,
                "person_id": memories.resolve_speaker(speakers, idx),
                "user_text": user_text.strip(),
                "bot_text": (turn["bot_text"] or "").strip(),
            }
        )
    return turns


def build_prompt(turns: list[dict[str, Any]], people: dict[str, dict[str, Any]], existing: list[dict[str, Any]]) -> str:
    """User message: people, their current memories and the conversation."""
    lines = ["People speaking in this conversation:"]
    lines += [f'- person_id "{pid}": {person["name"]}' for pid, person in people.items()]
    lines.append("")
    lines.append("Existing memories:")
    lines += [f'- id {m["id"]} (person_id "{m["person_id"]}"): {m["text"]}' for m in existing] or ["- (none)"]
    lines.append("")
    lines.append("Conversation:")
    for turn in turns:
        if turn["user_text"]:
            speaker = people.get(turn["person_id"] or "", {}).get("name", "Unknown speaker")
            lines.append(f"[turn {turn['idx']}] {speaker}: {turn['user_text']}")
        if turn["bot_text"]:
            lines.append(f"[turn {turn['idx']}] Assistant: {turn['bot_text']}")
    return "\n".join(lines)


@register
class DreamJob(Job):
    """Memory extraction over a recorded session."""

    kind = "dream"

    def required_services(self, ctx: JobContext) -> list[str]:
        """An on-demand LLM container when ``dream.service`` is configured."""
        service = dream_config(ctx.config).get("service")
        return [service] if service else []

    def run(self, ctx: JobContext) -> None:
        """Extract memories for the people attributed to the session."""
        session_id = ctx.job["target"]
        if ctx.progress.get("written"):
            return
        cfg = dream_config(ctx.config)
        session = ctx.store.get_session(session_id) or {}
        session_config = session.get("config") or {}
        language = session_config.get("language") or "en-US"

        speakers = memories.speakers_for(ctx.store, session_id)
        turns = [t for t in session_transcript(ctx, session_id, speakers) if t["idx"] >= 0]
        spoken = [t for t in turns if t["user_text"] and t["person_id"]]
        if not spoken:
            reason = "no speaker" if not (speakers["session"] or speakers["turns"]) else "no attributed user turn"
            ctx.save_progress(skipped=reason)
            logger.info(f"dream {session_id}: skipped ({reason})")
            return
        people = {
            pid: person
            for pid in sorted({t["person_id"] for t in spoken})
            if (person := memories.get_person(ctx.store, pid)) is not None
        }

        retracted = memories.retract_session_memories(ctx.store, session_id)
        existing = memories.usable_memories(ctx.store, list(people))
        by_id = {m["id"]: m for m in existing}

        ctx.check_preempted()
        client = make_client(session_config, cfg)
        client.wait_ready(ctx)
        ctx.check_preempted()
        system = SYSTEM_PROMPT.format(language_name=_language_name(language), categories="|".join(CATEGORIES))
        answer = client.complete(system, build_prompt(turns, people, existing))
        items = parse_memories(answer)
        ctx.check_preempted()

        turn_speaker = {t["idx"]: t["person_id"] for t in spoken}
        threshold = float(cfg["auto_use_threshold"])
        created = []
        for item in items:
            person_id = str(item.get("person_id") or "")
            text = str(item.get("text") or "").strip()[:500]
            if person_id not in people or not text:
                continue
            turn_ids = [i for i in _ints(item.get("turns")) if turn_speaker.get(i) == person_id]
            if not turn_ids:
                continue
            try:
                confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            old = next((by_id.get(i) for i in _ints([item.get("supersedes")])), None)
            if old is not None and old["person_id"] != person_id:
                old = None
            replaces_reviewed = old is not None and old["reviewed_at"] is not None
            status = "active" if confidence >= threshold and not replaces_reviewed else "proposed"
            category = item.get("category") if item.get("category") in CATEGORIES else "other"
            quote = str(item.get("quote") or "").strip()[:500] or None
            memory_id = memories.add_memory(
                ctx.store,
                {
                    "person_id": person_id,
                    "text": text,
                    "category": category,
                    "language": language,
                    "confidence": confidence,
                    "status": status,
                    "source": f"dream:{client.model}",
                    "source_version": client.model,
                    "supersedes": old["id"] if old else None,
                },
                [{"session_id": session_id, "turn_idx": idx, "quote": quote} for idx in turn_ids],
            )
            if old is not None and status == "active":
                memories.supersede(ctx.store, old["id"], memory_id)
            created.append({"id": memory_id, "status": status})
        ctx.save_progress(
            written=True,
            created=len(created),
            active=sum(1 for c in created if c["status"] == "active"),
            retracted=retracted,
        )
        logger.info(f"dream {session_id}: {len(created)} memories ({retracted} retracted from a previous run)")
