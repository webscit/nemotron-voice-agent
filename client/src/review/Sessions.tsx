// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import {
  artifactUrl,
  useLlmCall,
  useRequeueJob,
  useSaveReferences,
  useSessionDetail,
  useSessions,
  type LlmCallBrief,
  type TurnDetail,
} from "./api";
import { AudioClips, ImageStrip, StatusLine, TranscriptTable, WerBadge } from "./components";
import { formatDate, formatDuration, formatSecs, reviewHref, useAnnotator } from "./utils";

export function SessionsList() {
  const sessions = useSessions();
  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Sessions</h2>
          <p className="text-sm text-muted">Recorded conversations, newest first.</p>
        </div>
      </div>
      <StatusLine loading={sessions.isLoading} error={sessions.error} />
      {sessions.data && sessions.data.sessions.length === 0 && (
        <p className="text-sm text-muted">No recorded sessions yet. Set MONITORING_ENABLED=true and have a conversation.</p>
      )}
      {sessions.data && sessions.data.sessions.length > 0 && (
        <table className="rv-table">
          <thead>
            <tr>
              <th>Started</th>
              <th>Session</th>
              <th>Lang</th>
              <th>Models (ASR · LLM · TTS)</th>
              <th>Turns</th>
              <th>Reviewed</th>
              <th>Live WER</th>
              <th>Jobs</th>
            </tr>
          </thead>
          <tbody>
            {sessions.data.sessions.map((s) => {
              const liveWer = Object.entries(s.wer).find(([source]) => source.startsWith("live:"))?.[1];
              const duration = s.ended_at ? s.ended_at - s.started_at : null;
              return (
                <tr key={s.id} onClick={() => (globalThis.location.hash = reviewHref("session", s.id))}>
                  <td>
                    {formatDate(s.started_at)}
                    <div className="text-xs text-muted">{s.ended_at ? formatDuration(duration) : "live"}</div>
                  </td>
                  <td className="rv-mono">{s.id}</td>
                  <td>{s.language ?? "–"}</td>
                  <td className="text-xs">
                    {s.models.asr ?? "?"} · {s.models.llm ?? "?"} · {s.models.tts ?? "?"}
                  </td>
                  <td>{s.user_turns}</td>
                  <td>
                    {s.reviewed_turns}/{s.audio_turns}
                  </td>
                  <td>{liveWer === undefined ? "–" : <WerBadge value={liveWer} />}</td>
                  <td className="text-xs">
                    {Object.entries(s.jobs)
                      .map(([kind, status]) => `${kind}: ${status}`)
                      .join(", ") || "–"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </section>
  );
}

function LlmCallRow({ sessionId, call }: Readonly<{ sessionId: string; call: LlmCallBrief }>) {
  const [open, setOpen] = useState(false);
  const detail = useLlmCall(sessionId, open ? call.id : null);
  return (
    <div className="rv-llm-call">
      <button className="btn-ghost text-xs" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} LLM {call.model ?? ""} · TTFT {formatSecs(call.ttfb)} · {call.prompt_tokens ?? "?"}→
        {call.completion_tokens ?? "?"} tok
        {call.n_images ? ` · ${call.n_images} image(s)` : ""}
        {call.function_calls?.length ? ` · tools: ${call.function_calls.map((f) => f.name).join(", ")}` : ""}
        {call.interrupted ? " · interrupted" : ""}
      </button>
      {open && (
        <pre className="rv-pre">
          {detail.isLoading ? "Loading…" : JSON.stringify(detail.data?.messages ?? detail.error?.message, null, 2)}
        </pre>
      )}
    </div>
  );
}

function ReferenceEditor({ sessionId, turn }: Readonly<{ sessionId: string; turn: TurnDetail }>) {
  const [annotator] = useAnnotator();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(turn.human_reference?.text ?? turn.user_text ?? "");
  const save = useSaveReferences();
  if (!editing) {
    return (
      <button className="btn-ghost text-xs" onClick={() => setEditing(true)}>
        {turn.human_reference ? "Edit reference" : "Set reference"}
      </button>
    );
  }
  return (
    <div className="rv-inline-edit">
      <input className="input" value={draft} onChange={(e) => setDraft(e.target.value)} />
      <button
        className="btn-primary"
        disabled={!annotator || !draft.trim() || save.isPending}
        onClick={() =>
          save.mutate(
            { annotator, items: [{ session_id: sessionId, turn_idx: turn.idx, text: draft }] },
            { onSuccess: () => setEditing(false) }
          )
        }
      >
        Save
      </button>
      <button className="btn-ghost" onClick={() => setEditing(false)}>
        Cancel
      </button>
      {!annotator && <span className="text-xs rv-error">set your annotator name first</span>}
    </div>
  );
}

function TurnCard({ sessionId, liveSource, turn }: Readonly<{ sessionId: string; liveSource: string; turn: TurnDetail }>) {
  const reference = turn.human_reference?.text;
  const liveWer = turn.wer[liveSource];
  return (
    <div className="card rv-turn">
      <div className="rv-card-head">
        <span className="rv-badge rv-badge-muted">turn {turn.idx}</span>
        {turn.idx === 0 && <span className="text-xs text-muted">greeting</span>}
        {liveWer && <WerBadge value={liveWer.wer} title={`vs ${liveWer.reference_source}`} />}
        {turn.user_bot_latency !== null && (
          <span className="rv-badge rv-badge-muted">user→bot {formatSecs(turn.user_bot_latency)}</span>
        )}
        {turn.interrupted && <span className="rv-badge rv-badge-warn">interrupted</span>}
      </div>
      {(turn.user_text || turn.audio.user.length > 0) && (
        <div className="rv-side rv-side-user">
          <span className="rv-label">User</span>
          <AudioClips clips={turn.audio.user} />
          <TranscriptTable transcripts={turn.transcripts} referenceText={reference} />
          {reference && (
            <p className="text-sm">
              <span className="rv-badge rv-badge-good">{turn.human_reference?.source}</span> {reference}
            </p>
          )}
          {turn.audio.user.length > 0 && <ReferenceEditor key={reference ?? ""} sessionId={sessionId} turn={turn} />}
        </div>
      )}
      <ImageStrip images={turn.images} />
      {turn.llm_calls.map((call) => (
        <LlmCallRow key={call.id} sessionId={sessionId} call={call} />
      ))}
      {(turn.bot_text || turn.audio.bot.length > 0) && (
        <div className="rv-side rv-side-bot">
          <span className="rv-label">Assistant</span>
          <AudioClips clips={turn.audio.bot} />
          <p className="text-sm">{turn.bot_text}</p>
        </div>
      )}
    </div>
  );
}

export function SessionDetailView({ sessionId }: Readonly<{ sessionId: string }>) {
  const detail = useSessionDetail(sessionId);
  const requeue = useRequeueJob();
  const data = detail.data;
  return (
    <section className="rv-page">
      <a className="text-sm" href={reviewHref("sessions")}>
        ← Sessions
      </a>
      <StatusLine loading={detail.isLoading} error={detail.error} />
      {data && (
        <>
          <div className="rv-page-head">
            <div>
              <h2 className="text-lg font-semibold rv-mono">{data.session.id}</h2>
              <p className="text-sm text-muted">
                {formatDate(data.session.started_at)} · {data.session.language} · ASR {data.session.models.asr} · LLM{" "}
                {data.session.models.llm} · TTS {data.session.models.tts} ({data.session.models.voice})
              </p>
            </div>
            <div className="rv-actions">
              {Object.entries(data.wer_summary).map(([source, summary]) => (
                <WerBadge key={source} value={summary.wer} title={`${source} over ${summary.turns} turns`} />
              ))}
              <button
                className="btn-secondary"
                disabled={requeue.isPending}
                onClick={() => requeue.mutate({ kind: "reasr", session_ids: [sessionId] })}
                title="Transcribes missing turns with the reference/candidate ASR when the machine is idle"
              >
                Re-run reasr
              </button>
            </div>
          </div>
          {data.jobs.length > 0 && (
            <p className="text-xs text-muted">
              Jobs: {data.jobs.map((j) => `${j.kind} ${j.status}${j.error ? ` (${j.error})` : ""}`).join(" · ")}
            </p>
          )}
          {data.conversation_audio && (
            <div className="rv-audio">
              <span className="rv-label">Whole call (user left · assistant right)</span>
              <audio controls preload="none" src={artifactUrl(data.conversation_audio.key)} />
            </div>
          )}
          {data.turns.map((turn) => (
            <TurnCard key={turn.idx} sessionId={sessionId} liveSource={data.live_source} turn={turn} />
          ))}
        </>
      )}
    </section>
  );
}
