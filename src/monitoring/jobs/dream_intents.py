# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intent learning for the ``dream`` job.

1. Rebuild the intent catalog the session's engine used: custom sentence files,
   stock Home Assistant sentences (when the session used Home Assistant) and the
   learned suggestions that were active, restricted to the intents the session
   could run (declared client tools, allow-list).
2. Show the LLM the catalog and the conversation, marking the turns the engine
   did not answer and the tool calls the LLM made for them. Ask for new sentences
   for existing intents and for new client-tool intents.
3. Check every proposal with hassil, the way the live engine matches: the
   sentence must parse with the catalog's lists and rules, the person's own words
   must match it (and not already match the catalog), and the slots it yields
   must produce the tool call the LLM actually made. A proposal no evidence turn
   matches is dropped.
4. Store the rest as ``intent_suggestions``: merged into an existing suggestion
   proposing the same thing (evidence accumulates across sessions), skipped when
   a reviewer rejected it before.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from monitoring import intents

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    # Validated sentences for existing intents at or above this confidence are used
    # live before review. None: every suggestion waits for approval.
    "auto_use_threshold": None,
    # Templates shown per intent in the prompt.
    "max_examples": 4,
    # List values shown per list in the prompt.
    "max_list_values": 20,
}

SYSTEM_PROMPT = """You tune the intent engine of a voice assistant. The engine answers device commands
without the language model by matching what the person said against sentence templates (hassil format).
In the turns marked "missed", the engine did not match and the language model answered instead.

Look only at the missed turns and propose, in {language_name}:
1. "sentences": a missed turn asks for exactly what an existing intent of the catalog does, phrased in a
   way its templates do not match. Write templates that cover that phrasing and its close variants.
2. "intents": a missed turn made the assistant call one client tool with arguments that would be the
   same every time the person asks this again (no free-form content such as a message, a search or a
   question), and no intent of the catalog does it.

Template syntax (hassil): (a|b) alternatives, [word] optional, {{list}} a slot value from a list,
<rule> an expansion rule. Only use lists and rules from the catalog, never invent one. Lowercase,
no final punctuation, no wake word or greeting. Keep templates specific: they must not match a
request for something else.

Rules:
- Skip questions, small talk, requests needing reasoning or knowledge, and anything no intent or tool can do.
- slots: fixed values implied by the sentence (e.g. {{"direction": "left"}}), else {{}}.
- New intent names are CamelCase. arguments: literal values, or {{"slot": "<slot name>"}} for a
  value taken from a {{list}} of the sentence.
- response: a short spoken confirmation in {language_name}; it may use {{{{ slots.<name> }}}}.
- turns: the missed turn numbers a proposal comes from.
- confidence: 0.9+ when the turn clearly asks for exactly that, lower when unsure.

Answer with JSON only, no prose:
{{"sentences": [{{"intent": "...", "sentences": ["..."], "slots": {{}}, "turns": [1], "confidence": 0.0}}],
"intents": [{{"name": "...", "tool": "...", "arguments": {{}}, "sentences": ["..."], "response": "...",
"turns": [1], "confidence": 0.0}}]}}
Return {{"sentences": [], "intents": []}} when nothing qualifies."""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_NAME_RE = re.compile(r"^[A-Z][A-Za-z0-9]{2,63}$")
# Marks the proposal under test in the merged sentence data.
_CANDIDATE_ID = -1


def intents_config(dream_cfg: dict[str, Any]) -> dict[str, Any]:
    """The ``dream.intents`` section merged over the defaults."""
    return {**DEFAULTS, **(dream_cfg.get("intents") or {})}


def parse_proposals(raw: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract ``(sentences, intents)`` from a model answer (tolerates think tags and prose)."""
    text = _THINK_RE.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in LLM answer: {text[:500]!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in LLM answer ({exc}): {text[:500]!r}") from None
    if not isinstance(data, dict):
        raise ValueError("LLM answer is not a JSON object")

    def items(key: str) -> list[dict[str, Any]]:
        value = data.get(key)
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    return items("sentences"), items("intents")


def _same_value(a: Any, b: Any) -> bool:
    """Tool argument equality that ignores ``5`` vs ``"5"`` and case."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return str(a).strip().casefold() == str(b).strip().casefold()


def _same_arguments(built: dict[str, Any], called: dict[str, Any]) -> bool:
    return set(built) == set(called) and all(_same_value(built[k], called[k]) for k in built)


def _strings(values: Any) -> list[str]:
    out = [" ".join(str(v).split()) for v in values if isinstance(v, str)] if isinstance(values, list) else []
    return list(dict.fromkeys(s for s in out if s))


def _scalar_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(k): v for k, v in value.items() if isinstance(v, (str, int, float, bool))}


@dataclass
class Check:
    """Outcome of checking one proposal against its evidence turns."""

    matched: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unchecked: bool = False


class IntentCatalog:
    """The sentence data and tool mapping a session's intent engine used, rebuilt offline."""

    def __init__(self, store, session_config: dict[str, Any], cfg: dict[str, Any]):
        """Rebuild the catalog for ``session_config`` (see ``available``)."""
        from examples.shared.intent_engine.config import IntentEngineConfig
        from examples.shared.intent_engine.learned import LearnedIntent, merge_learned, tool_entries
        from examples.shared.intent_engine.matcher import (
            _stock_data,
            custom_parse_data,
            custom_sentence_data,
        )
        from examples.shared.intent_engine.targets import ClientToolMap

        self.cfg = cfg
        self.notes: list[str] = []
        snapshot = session_config.get("intent_engine") or {}
        self.enabled = bool(snapshot.get("enabled"))
        self.language = session_config.get("language") or "en-US"
        self.base = intents.base_language(self.language)
        self.env = IntentEngineConfig.from_env()
        self.allowed = tuple(snapshot.get("allowed_intents") or self.env.allowed_intents)
        self.blocked_domains = tuple(snapshot.get("blocked_domains") or self.env.blocked_domains)
        self.wake_words = tuple(snapshot.get("wake_words") or ())
        self.area = snapshot.get("area") or ""
        self.use_home_assistant = bool(snapshot.get("home_assistant"))
        self.client_tools = tuple(session_config.get("client_tools") or ())

        self.learned = [LearnedIntent.from_row(r) for r in intents.active_suggestions(store, self.language)]
        self.tool_map = ClientToolMap.load(self.env.sentences_dir).with_entries(tool_entries(self.learned))

        def keep(name: str) -> bool:
            if name in self.tool_map:
                return self.tool_map.tool(name) in self.client_tools
            return self.use_home_assistant and name in self.allowed

        files = custom_sentence_data(self.env.sentences_dir, self.language) or {}
        custom_names = set(files.get("intents") or {}) | {i.intent for i in self.learned if i.kind == "intent"}
        merged, _ = merge_learned(files, self.learned, add_intents=True)
        self.custom_responses = (merged.get("responses") or {}).get("intents") or {}
        self.custom = custom_parse_data(merged, self.language, keep)
        self.stock = None
        if self.use_home_assistant and (stock := _stock_data(self.language)):
            pruned = {n: b for n, b in (stock.get("intents") or {}).items() if n in self.allowed}
            stock_learned = [i for i in self.learned if i.kind == "sentence" and i.intent not in custom_names]
            self.stock, _ = merge_learned({**stock, "intents": pruned}, stock_learned, add_intents=False)
        self.slot_lists = self._sync_home_assistant() if self.stock is not None else None

    @property
    def available(self) -> bool:
        """Whether the session ran the engine with sentences for its language."""
        return self.enabled and bool(self.custom or self.stock)

    # -------------------------------------------------------------- catalog
    def _sync_home_assistant(self):
        """Home Assistant entity, area and floor names (None when the dreamer cannot reach it)."""
        from examples.shared.intent_engine.targets import HomeAssistantTarget, TargetUnavailable

        if not self.env.home_assistant_configured:
            self.notes.append("Home Assistant is not configured for the dreamer; its names are not checked")
            return None

        class _NoTemplates:
            def response_template(self, *_args, **_kwargs):
                return None

            def error_template(self, *_args, **_kwargs):
                return None

        async def sync():
            target = HomeAssistantTarget(
                self.env.home_assistant_url,
                self.env.home_assistant_token,
                templates=_NoTemplates(),
                timeout_secs=self.env.timeout_secs,
                cache_ttl_secs=0.0,
                area=self.area,
                blocked_domains=self.blocked_domains,
            )
            try:
                return await target.sync()
            finally:
                await target.aclose()

        try:
            return asyncio.run(sync())
        except (TargetUnavailable, OSError) as exc:
            self.notes.append(f"Home Assistant sync failed ({exc}); its names are not checked")
            return None

    def owner(self, name: str) -> dict[str, Any] | None:
        """The sentence data (custom first, like the live engine) that defines an intent."""
        for data in (self.custom, self.stock):
            if data and name in (data.get("intents") or {}):
                return data
        return None

    def learnable(self, name: str) -> bool:
        """Whether sentences may be learned for an intent (defined, and not reserved for the LLM)."""
        if self.owner(name) is None:
            return False
        return not (name in self.tool_map and self.tool_map.defers_to_llm(name))

    def names(self) -> list[str]:
        """Learnable intent names, custom first."""
        out = []
        for data in (self.custom, self.stock):
            out += [n for n in (data or {}).get("intents") or {} if n not in out and self.learnable(n)]
        return out

    def _templates(self, name: str) -> list[str]:
        body = (self.owner(name) or {}).get("intents", {}).get(name) or {}
        return [s for block in body.get("data") or [] for s in block.get("sentences") or []]

    def describe(self) -> str:
        """Catalog section of the prompt: intents, what they run, examples, lists and rules."""
        from examples.shared.intent_engine.learned import sentence_references

        max_examples = int(self.cfg["max_examples"])
        lines = ["Intent catalog:"]
        lists: dict[str, dict[str, Any]] = {}
        rules: dict[str, str] = {}
        for name in self.names():
            data = self.owner(name) or {}
            if name in self.tool_map:
                entry = self.tool_map.entry(name)
                runs = f"client tool {entry.get('tool')} with arguments {json.dumps(entry.get('arguments') or {})}"
            else:
                runs = "a Home Assistant intent"
            examples = self._templates(name)[:max_examples]
            lines.append(f"- {name}: runs {runs}")
            for template in examples:
                lines.append(f"    {template}")
                try:
                    used_lists, used_rules, _ = sentence_references(template)
                except ValueError:
                    continue
                for list_name in used_lists:
                    if list_name in (data.get("lists") or {}):
                        lists[list_name] = data["lists"][list_name]
                for rule in used_rules:
                    if rule in (data.get("expansion_rules") or {}):
                        rules[rule] = data["expansion_rules"][rule]
        for list_name, body in (self.custom or {}).get("lists", {}).items():
            lists.setdefault(list_name, body)
        for rule, body in (self.custom or {}).get("expansion_rules", {}).items():
            rules.setdefault(rule, body)
        lines.append("")
        lines.append("Lists:")
        for list_name, body in sorted(lists.items()):
            lines.append(f"- {{{list_name}}}: {self._describe_list(body)}")
        if self.stock is not None:
            names = ""
            if self.slot_lists:
                values = [v.text_in.text for v in self.slot_lists["name"].values][: int(self.cfg["max_list_values"])]
                names = f" (e.g. {', '.join(values)})" if values else ""
            lines.append(f"- {{name}}, {{area}}, {{floor}}: Home Assistant entity, area and floor names{names}")
        lines.append("")
        lines.append("Expansion rules:")
        lines += [f"- <{rule}> = {body}" for rule, body in sorted(rules.items())] or ["- (none)"]
        lines.append("")
        lines.append(f"Client tools of this session: {', '.join(self.client_tools) or '(none)'}")
        return "\n".join(lines)

    def _describe_list(self, body: Any) -> str:
        if not isinstance(body, dict):
            return "values"
        if "range" in body:
            r = body["range"] or {}
            return f"numbers {r.get('from')}..{r.get('to')}"
        if body.get("wildcard"):
            return "any text (do not use)"
        values = []
        for value in (body.get("values") or [])[: int(self.cfg["max_list_values"])]:
            if isinstance(value, dict):
                values.append(f"{value.get('in')} -> {value.get('out', value.get('in'))}")
            else:
                values.append(str(value))
        return "; ".join(values) or "values"

    # ------------------------------------------------------------- matching
    def normalize(self, text: str) -> str:
        """The user text as the live engine matches it."""
        from examples.shared.intent_engine.matcher import normalize_text

        return normalize_text(text, self.language, self.wake_words)

    def recognize(self, text: str, *, custom: dict | None = None, stock: dict | None = None):
        """``(intent name, slots, learned id)`` for ``text`` through custom then stock data, or None."""
        from hassil import Intents, recognize_best

        from examples.shared.intent_engine.learned import LEARNED_ID_KEY
        from examples.shared.intent_engine.matcher import _EMPTY_SLOT_LISTS

        context = {"area": {"value": self.area, "text": self.area}} if self.area else None
        for data in (custom or self.custom, stock or self.stock):
            if not data:
                continue
            try:
                result = recognize_best(
                    text,
                    Intents.from_dict(data),
                    slot_lists=self.slot_lists or _EMPTY_SLOT_LISTS,
                    intent_context=context,
                    language=self.base,
                    best_slot_name="name",
                )
            except Exception as exc:
                logger.debug(f"dream intents: matching failed ({type(exc).__name__}: {exc})")
                continue
            if result is not None:
                slots = {entity.name: entity.value for entity in result.entities_list}
                return result.intent.name, slots, (result.intent_metadata or {}).get(LEARNED_ID_KEY)
        return None

    def check(self, candidate, turns: list[dict[str, Any]]) -> Check:
        """Match a ``LearnedIntent`` candidate against the person's words in its evidence turns."""
        from examples.shared.intent_engine.learned import (
            HOME_ASSISTANT_LISTS,
            merge_learned,
            sentence_references,
        )
        from examples.shared.intent_engine.targets import ClientToolMap

        result = Check()
        if candidate.kind == "intent":
            empty = {"language": self.base, "intents": {}, "lists": {}, "expansion_rules": {}}
            custom, _ = merge_learned(self.custom or empty, [candidate], add_intents=True)
            stock = self.stock
            tool_map = self.tool_map.with_entries({candidate.intent: candidate.tool or {}})
        else:
            owner = self.owner(candidate.intent)
            merged, _ = merge_learned(owner or {}, [candidate], add_intents=False)
            custom, stock = (merged, self.stock) if owner is self.custom else (self.custom, merged)
            tool_map = self.tool_map
        needs_names = any(sentence_references(s)[0] & HOME_ASSISTANT_LISTS for s in candidate.sentences if _parses(s))
        if needs_names and not self.slot_lists:
            result.unchecked = True
            result.notes.append("not matched: Home Assistant names unavailable")
            result.matched = [t["idx"] for t in turns]
            return result
        follow_up = candidate.intent in tool_map and tool_map.has_follow_up(candidate.intent)
        for turn in turns:
            text = self.normalize(turn["user_text"])
            before = self.recognize(text)
            if before is not None:
                result.notes.append(f"turn {turn['idx']} already matches {before[0]}")
                continue
            after = self.recognize(text, custom=custom, stock=stock)
            if after is None or after[0] != candidate.intent or after[2] != _CANDIDATE_ID:
                result.notes.append(f"turn {turn['idx']} does not match")
                continue
            if candidate.intent in tool_map and not follow_up:
                tool = tool_map.tool(candidate.intent)
                built = ClientToolMap({candidate.intent: tool_map.entry(candidate.intent)}).arguments(
                    candidate.intent, after[1]
                )
                calls = [c for c in turn["tool_calls"] if c.get("name") == tool]
                if calls and not any(_same_arguments(built, c.get("arguments") or {}) for c in calls):
                    result.notes.append(f"turn {turn['idx']} matches but the tool call differs")
                    continue
                if not calls and candidate.kind == "intent":
                    result.notes.append(f"turn {turn['idx']} has no {tool} call")
                    continue
            result.matched.append(turn["idx"])
        return result


def _parses(template: str) -> bool:
    from examples.shared.intent_engine.learned import sentence_references

    try:
        sentence_references(template)
    except ValueError:
        return False
    return True


def missed_turns(ctx, session_id: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """User turns the engine did not answer, with the LLM's tool calls and the engine's reason."""
    handled: dict[int, str] = {}
    reasons: dict[int, str] = {}
    for event in ctx.store.rows("events", session_id, kind=intents.INTENT_EVENT):
        data = event["data"] or {}
        if event["turn_idx"] is None:
            continue
        if data.get("handled_by") == "intent":
            handled[event["turn_idx"]] = data.get("intent") or "?"
        else:
            reasons[event["turn_idx"]] = data.get("reason") or ""
    calls: dict[int, list[dict[str, Any]]] = {}
    for row in ctx.store.rows("llm_calls", session_id):
        for call in row["function_calls"] or []:
            if isinstance(call, dict) and call.get("name"):
                calls.setdefault(row["turn_idx"], []).append(
                    {"name": call["name"], "arguments": call.get("arguments") or {}}
                )
    out = []
    for turn in turns:
        idx = turn["idx"]
        out.append(
            {
                **turn,
                "handled_by_intent": handled.get(idx),
                "reason": reasons.get(idx),
                "tool_calls": calls.get(idx, []),
                # Matched but routed to the LLM on purpose (blocked domain, dry run, ...) is not a miss.
                "missed": bool(turn["user_text"]) and idx not in handled and reasons.get(idx) in (None, "", "no_match"),
            }
        )
    return out


def build_prompt(catalog: IntentCatalog, turns: list[dict[str, Any]]) -> str:
    """User message: the catalog, then the conversation with missed turns marked."""
    lines = [catalog.describe(), "", "Conversation:"]
    for turn in turns:
        if turn["user_text"]:
            if turn["handled_by_intent"]:
                mark = f" (answered by the engine: {turn['handled_by_intent']})"
            elif turn["missed"]:
                mark = " (missed)"
            else:
                mark = ""
            lines.append(f"[turn {turn['idx']}] User{mark}: {turn['user_text']}")
        for call in turn["tool_calls"]:
            lines.append(f"[turn {turn['idx']}] Tool call: {call['name']}({json.dumps(call['arguments'])})")
        if turn["bot_text"]:
            lines.append(f"[turn {turn['idx']}] Assistant: {turn['bot_text']}")
    return "\n".join(lines)


def _confidence(item: dict[str, Any]) -> float:
    try:
        return min(1.0, max(0.0, float(item.get("confidence", 0.5))))
    except (TypeError, ValueError):
        return 0.5


def _turn_ids(item: dict[str, Any], missed: dict[int, dict[str, Any]]) -> list[int]:
    out = []
    for value in item.get("turns") if isinstance(item.get("turns"), list) else []:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx in missed and idx not in out:
            out.append(idx)
    return out


def _as_existing_intent(catalog: IntentCatalog, tool: str, arguments: dict[str, Any]) -> str | None:
    """A catalog intent that already sends exactly this tool call (literal arguments only)."""
    for name in catalog.names():
        if name not in catalog.tool_map or catalog.tool_map.has_follow_up(name):
            continue
        entry = catalog.tool_map.entry(name)
        specs = entry.get("arguments") or {}
        literal = not any(isinstance(v, dict) for v in specs.values())
        if entry.get("tool") == tool and literal and _same_arguments(specs, arguments):
            return name
    return None


def candidates(
    catalog: IntentCatalog, sentence_items: list[dict[str, Any]], intent_items: list[dict[str, Any]], missed
) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn the LLM proposals into checked suggestion rows (without ``source``/``status``), plus drop notes."""
    from examples.shared.intent_engine.learned import LearnedIntent, sentence_problems, sentence_references
    from examples.shared.intent_engine.responses import is_local_template, render_local

    dropped: list[str] = []
    out: list[dict[str, Any]] = []
    known_names = set(catalog.names()) | set((catalog.custom or {}).get("intents") or {})

    def problems_for(sentences: list[str], data: dict[str, Any] | None) -> list[str]:
        from examples.shared.intent_engine.learned import HOME_ASSISTANT_LISTS

        extra = HOME_ASSISTANT_LISTS if catalog.stock is not None else ()
        return [p for s in sentences for p in sentence_problems(s, data or {}, extra_lists=extra)]

    def finish(item, kind, intent, sentences, slots, tool, response, turn_ids):
        candidate = LearnedIntent(
            id=_CANDIDATE_ID,
            kind=kind,
            intent=intent,
            sentences=tuple(sentences),
            slots=slots,
            tool=tool,
            response=response,
        )
        check = catalog.check(candidate, [missed[i] for i in turn_ids])
        if not check.matched:
            dropped.append(f"{intent}: {'; '.join(check.notes) or 'no evidence turn matches'}")
            return
        out.append(
            {
                "kind": kind,
                "language": catalog.base,
                "intent": intent,
                "sentences": list(sentences),
                "slots": slots or None,
                "tool": tool,
                "response": response,
                "confidence": _confidence(item),
                "validated": not check.unchecked and len(check.matched) == len(turn_ids),
                "validation": {"matched_turns": check.matched, "notes": check.notes},
                "evidence": [
                    {
                        "turn_idx": i,
                        "text": missed[i]["user_text"],
                        "tool_calls": missed[i]["tool_calls"] or None,
                    }
                    for i in check.matched
                ],
            }
        )

    for item in sentence_items:
        intent = str(item.get("intent") or "")
        sentences, turn_ids = _strings(item.get("sentences")), _turn_ids(item, missed)
        if not catalog.learnable(intent) or not sentences or not turn_ids:
            dropped.append(f"{intent or '?'}: unknown intent, no sentence or no missed turn")
            continue
        problems = problems_for(sentences, catalog.owner(intent))
        if problems:
            dropped.append(f"{intent}: {'; '.join(problems)}")
            continue
        finish(item, "sentence", intent, sentences, _scalar_dict(item.get("slots")), None, None, turn_ids)

    for item in intent_items:
        name, tool = str(item.get("name") or ""), str(item.get("tool") or "")
        sentences, turn_ids = _strings(item.get("sentences")), _turn_ids(item, missed)
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        response = " ".join(str(item.get("response") or "").split())[:300]
        if tool not in catalog.client_tools or not sentences or not turn_ids:
            dropped.append(f"{name or '?'}: tool {tool!r} is not a client tool of the session, or no sentence/turn")
            continue
        existing = _as_existing_intent(catalog, tool, arguments)
        if existing:
            finish(item, "sentence", existing, sentences, {}, None, None, turn_ids)
            continue
        if not _NAME_RE.match(name) or name in known_names:
            dropped.append(f"{name or '?'}: invalid or already used intent name")
            continue
        problems = problems_for(sentences, catalog.custom)
        slot_names = set()
        for sentence in sentences:
            if _parses(sentence):
                slot_names |= sentence_references(sentence)[2]
        clean_args: dict[str, Any] = {}
        for key, value in arguments.items():
            from_slot = (
                isinstance(value, dict) and set(value) <= {"slot", "default"} and value.get("slot") in slot_names
            )
            if from_slot or isinstance(value, (str, int, float, bool)):
                clean_args[str(key)] = value
            else:
                problems.append(f"argument {key!r} is neither a literal nor a slot of the sentences")
        if not response or not is_local_template(response):
            problems.append("response is empty or not a slots-only template")
        else:
            try:
                render_local(response, {})
            except Exception as exc:
                problems.append(f"response does not render: {exc}")
        if problems:
            dropped.append(f"{name}: {'; '.join(problems)}")
            continue
        finish(item, "intent", name, sentences, {}, {"tool": tool, "arguments": clean_args}, response, turn_ids)
    return out, dropped


def store_candidates(
    store, session_id: str, rows: list[dict[str, Any]], *, source: str, model: str, auto_use_threshold
) -> dict[str, int]:
    """Insert new suggestions or merge into equivalent ones; returns counts."""
    counts = {"created": 0, "merged": 0, "rejected_before": 0, "active": 0}
    for row in rows:
        evidence = [{**e, "session_id": session_id} for e in row.pop("evidence")]
        existing = intents.suggestions_for(store, row["language"], (*intents.USABLE_STATUSES, "rejected"))
        same = next((r for r in existing if intents.same_definition(r, row)), None)
        if same is not None and same["status"] == "rejected":
            counts["rejected_before"] += 1
            continue
        if same is not None:
            intents.add_evidence(store, same["id"], evidence, sentences=row["sentences"])
            counts["merged"] += 1
            continue
        auto = (
            auto_use_threshold is not None
            and row["kind"] == "sentence"
            and row["validated"]
            and row["confidence"] >= float(auto_use_threshold)
        )
        status = "active" if auto else "proposed"
        intents.add_suggestion(store, {**row, "status": status, "source": source, "source_version": model}, evidence)
        counts["created"] += 1
        counts["active"] += int(auto)
    return counts


def learn_intents(ctx, session_id: str, session_config: dict[str, Any], turns, get_client, dream_cfg) -> dict[str, Any]:
    """Run intent learning over one session; returns the progress to save.

    ``get_client`` returns the job's LLM client; it is only called when there is something to learn.
    """
    from monitoring.jobs.dream import _language_name

    cfg = intents_config(dream_cfg)
    if not cfg["enabled"]:
        return {"intents_skipped": "disabled"}
    catalog = IntentCatalog(ctx.store, session_config, cfg)
    if not catalog.available:
        return {"intents_skipped": "intent engine off or no sentences for the session language"}
    annotated = missed_turns(ctx, session_id, turns)
    missed = {t["idx"]: t for t in annotated if t["missed"]}
    if not missed:
        return {"intents_skipped": "no missed turn"}

    retracted = intents.retract_session_suggestions(ctx.store, session_id)
    client = get_client()
    client.wait_ready(ctx)
    ctx.check_preempted()
    system = SYSTEM_PROMPT.format(language_name=_language_name(catalog.language))
    answer = client.complete(system, build_prompt(catalog, annotated))
    ctx.check_preempted()
    sentence_items, intent_items = parse_proposals(answer)
    rows, dropped = candidates(catalog, sentence_items, intent_items, missed)
    for note in dropped:
        logger.info(f"dream {session_id}: intent proposal dropped: {note}")
    counts = store_candidates(
        ctx.store,
        session_id,
        rows,
        source=f"{intents.DREAM_PREFIX}{client.model}",
        model=client.model,
        auto_use_threshold=cfg["auto_use_threshold"],
    )
    return {"intents": {**counts, "dropped": len(dropped), "retracted": retracted, "notes": catalog.notes}}


def correction_problems(
    store, row: dict[str, Any], sentences: list[str] | None, name: str | None, response: str | None
) -> list[str]:
    """Why a reviewer's correction of a suggestion cannot be used (empty when it can).

    Sentences must parse with the lists and rules of the sentence set the intent
    belongs to; a renamed intent must keep a free CamelCase name; a response must
    be a slots-only template.
    """
    from examples.shared.intent_engine.config import IntentEngineConfig
    from examples.shared.intent_engine.learned import (
        HOME_ASSISTANT_LISTS,
        LearnedIntent,
        merge_learned,
        sentence_problems,
    )
    from examples.shared.intent_engine.matcher import _stock_data, custom_sentence_data
    from examples.shared.intent_engine.responses import is_local_template

    language = row["language"]
    learned = [LearnedIntent.from_row(r) for r in intents.active_suggestions(store, language) if r["id"] != row["id"]]
    files = custom_sentence_data(IntentEngineConfig.from_env().sentences_dir, language) or {}
    custom, _ = merge_learned(files, learned, add_intents=True)
    stock = _stock_data(language) or {}
    problems: list[str] = []
    if row["kind"] == "intent":
        data = custom
        if name is not None and name != row["intent"]:
            taken = set(custom.get("intents") or {}) | set(stock.get("intents") or {})
            if not _NAME_RE.match(name) or name in taken:
                problems.append(f"intent name {name!r} is not CamelCase or is already used")
        if response is not None and (not response.strip() or not is_local_template(response)):
            problems.append("response is empty or not a slots-only template")
    elif row["intent"] in (custom.get("intents") or {}):
        data = custom
    else:
        data = stock
    for sentence in sentences or []:
        problems += [f"{sentence!r}: {p}" for p in sentence_problems(sentence, data, extra_lists=HOME_ASSISTANT_LISTS)]
    return problems
