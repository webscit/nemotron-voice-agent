# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``reasr``: re-transcribe recorded user turns and score ASR variants by WER.

For each user turn with recorded audio:

1. transcribe it with the configured ``reference`` endpoint (pseudo ground truth,
   the strongest offline model) and every ``candidates`` endpoint, storing
   ``kind="transcript"`` annotations (source = endpoint name);
2. pick the reference text: the latest human correction
   (``kind="transcript"``, ``source="human:<name>"``) wins over the reference model;
3. score the live transcript (``source="live:<asr model>"``) and each candidate
   against it: ``kind="wer"`` per turn plus a ``kind="wer_summary"`` per session
   (corpus-level WER = total errors / total reference words).
"""

from __future__ import annotations

import io
import wave
from collections import defaultdict
from typing import Any

from loguru import logger

from monitoring.jobs.asr_client import AsrEndpoint, RivaTranscriber
from monitoring.jobs.base import Job, JobContext, register
from monitoring.jobs.wer import error_rates


def _endpoints(config: dict[str, Any]) -> tuple[AsrEndpoint | None, list[AsrEndpoint]]:
    section = config.get("reasr") or {}
    reference = AsrEndpoint.from_config(section["reference"]) if section.get("reference") else None
    candidates = [AsrEndpoint.from_config(raw) for raw in section.get("candidates") or []]
    return reference, candidates


def _turn_wav(ctx: JobContext, segments: list[dict[str, Any]]) -> bytes:
    """Concatenate a turn's user audio segments into one mono WAV."""
    pcm, sample_rate = bytearray(), None
    for segment in segments:
        with wave.open(io.BytesIO(ctx.artifacts.get(segment["artifact_key"]))) as wf:
            sample_rate = sample_rate or wf.getframerate()
            pcm.extend(wf.readframes(wf.getnframes()))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate or 16000)
        wf.writeframes(bytes(pcm))
    return buffer.getvalue()


@register
class ReAsrJob(Job):
    """Re-transcription + WER scoring of a recorded session."""

    kind = "reasr"

    def required_services(self, ctx: JobContext) -> list[str]:
        """Containers of the endpoints that apply to this session's language."""
        return [e.service for e in self._endpoints_for(ctx) if e.service]

    @staticmethod
    def _session_language(ctx: JobContext) -> str:
        session = ctx.store.get_session(ctx.job["target"]) or {}
        return (session.get("config") or {}).get("language") or "en-US"

    def _endpoints_for(self, ctx: JobContext) -> list[AsrEndpoint]:
        reference, candidates = _endpoints(ctx.config)
        language = self._session_language(ctx)
        return [e for e in [reference, *candidates] if e and e.supports(language)]

    def run(self, ctx: JobContext) -> None:
        """Transcribe missing (turn, endpoint) pairs, then (re)compute WER."""
        session_id = ctx.job["target"]
        session = ctx.store.get_session(session_id) or {}
        session_config = session.get("config") or {}
        language = self._session_language(ctx)
        reference, candidates = _endpoints(ctx.config)
        endpoints = self._endpoints_for(ctx)

        segments_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in ctx.store.rows("media", session_id, modality="audio_user"):
            segments_by_turn[row["turn_idx"]].append(row)

        done = {tuple(item) for item in ctx.progress.get("done", [])}
        for endpoint in endpoints:
            transcriber = None
            for turn_idx, segments in sorted(segments_by_turn.items()):
                if (endpoint.name, turn_idx) in done:
                    continue
                ctx.check_preempted()
                if transcriber is None:
                    transcriber = RivaTranscriber(endpoint)
                    transcriber.wait_ready()
                text = transcriber.transcribe_wav(_turn_wav(ctx, segments), language)
                ctx.store.add_annotations(
                    [
                        {
                            "session_id": session_id,
                            "target_type": "turn",
                            "target_id": f"{session_id}:{turn_idx}",
                            "source": endpoint.name,
                            "source_version": endpoint.model or endpoint.server,
                            "kind": "transcript",
                            "value": {"text": text, "language": language},
                        }
                    ]
                )
                done.add((endpoint.name, turn_idx))
                ctx.save_progress(done=sorted(done))

        self._score(ctx, session_id, session_config, reference, candidates, set(segments_by_turn))

    def _score(
        self,
        ctx: JobContext,
        session_id: str,
        session_config: dict[str, Any],
        reference: AsrEndpoint | None,
        candidates: list[AsrEndpoint],
        audio_turns: set[int],
    ) -> None:
        # Latest transcript per (turn target, source).
        transcripts: dict[tuple[str, str], str] = {}
        for row in ctx.store.annotations_for(session_id, kind="transcript"):
            transcripts[(row["target_id"], row["source"])] = (row["value"] or {}).get("text", "")
        live_source = f"live:{(session_config.get('asr') or {}).get('model') or 'asr'}"
        for turn in ctx.store.rows("turns", session_id):
            if turn["idx"] in audio_turns and turn["user_text"] is not None:
                transcripts[(f"{session_id}:{turn['idx']}", live_source)] = turn["user_text"]

        hypothesis_sources = [live_source, *(c.name for c in candidates)]
        per_turn, totals = [], defaultdict(lambda: {"word_errors": 0, "ref_words": 0, "turns": 0, "refs": set()})
        for turn_idx in sorted(audio_turns):
            target = f"{session_id}:{turn_idx}"
            humans = [src for (tgt, src) in transcripts if tgt == target and src.startswith("human:")]
            ref_source = humans[-1] if humans else (reference.name if reference else None)
            if not ref_source or (target, ref_source) not in transcripts:
                continue
            ref_text = transcripts[(target, ref_source)]
            for source in hypothesis_sources:
                if (target, source) not in transcripts:
                    continue
                rates = error_rates(ref_text, transcripts[(target, source)])
                per_turn.append(
                    {
                        "session_id": session_id,
                        "target_type": "turn",
                        "target_id": target,
                        "source": source,
                        "source_version": None,
                        "kind": "wer",
                        "value": {**rates, "reference_source": ref_source},
                    }
                )
                agg = totals[source]
                agg["word_errors"] += rates["word_errors"]
                agg["ref_words"] += rates["ref_words"]
                agg["turns"] += 1
                agg["refs"].add(ref_source)

        summaries = [
            {
                "session_id": session_id,
                "target_type": "session",
                "target_id": session_id,
                "source": source,
                "source_version": None,
                "kind": "wer_summary",
                "value": {
                    "wer": agg["word_errors"] / agg["ref_words"] if agg["ref_words"] else None,
                    "word_errors": agg["word_errors"],
                    "ref_words": agg["ref_words"],
                    "turns": agg["turns"],
                    "reference_sources": sorted(agg["refs"]),
                },
            }
            for source, agg in totals.items()
        ]
        ctx.store.add_annotations(per_turn + summaries)
        if not per_turn:
            logger.info(f"reasr {session_id}: no reference transcript available (configure reasr.reference)")
        for summary in summaries:
            logger.info(f"reasr {session_id}: {summary['source']} WER={summary['value']['wer']}")
