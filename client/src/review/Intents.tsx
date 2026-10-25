// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import {
  intentExportUrl,
  useIntentSuggestions,
  useReviewIntent,
  type IntentSuggestion,
  type IntentSuggestionFilter,
  type IntentSuggestionKind,
} from "./api";
import { AudioClips, StatusLine } from "./components";
import { formatDate, reviewHref, useAnnotator } from "./utils";

const FILTERS: { id: IntentSuggestionFilter; label: string }[] = [
  { id: "open", label: "To review" },
  { id: "active", label: "In use" },
  { id: "rejected", label: "Rejected" },
  { id: "all", label: "All" },
];

const KINDS: { id: IntentSuggestionKind; label: string; title: string }[] = [
  { id: "sentence", label: "New sentences", title: "Another way to say a command an intent already handles" },
  { id: "intent", label: "New intent", title: "A recurring request the LLM answered with one client tool call" },
];

const STATUS_TONE: Record<string, string> = {
  active: "rv-badge-good",
  proposed: "rv-badge-warn",
  rejected: "rv-badge-bad",
  superseded: "rv-badge-muted",
};

function Evidence({ suggestion }: Readonly<{ suggestion: IntentSuggestion }>) {
  if (!suggestion.evidence.length) return <p className="text-xs text-muted">No transcript evidence.</p>;
  return (
    <div className="rv-evidence">
      {suggestion.evidence.map((ev) => (
        <div key={`${ev.session_id}-${ev.turn_idx}`} className="rv-side rv-side-user">
          <span className="rv-label">
            <a href={reviewHref("session", ev.session_id)}>
              {ev.session_id} · turn {ev.turn_idx}
            </a>
          </span>
          <AudioClips clips={ev.audio} />
          {ev.text && <p className="text-sm">“{ev.text}”</p>}
          {ev.tool_calls?.map((call, i) => (
            <p key={`${call.name}-${i}`} className="text-xs text-muted rv-mono">
              LLM called {call.name}({JSON.stringify(call.arguments)})
            </p>
          ))}
        </div>
      ))}
    </div>
  );
}

function SuggestionCard({ suggestion }: Readonly<{ suggestion: IntentSuggestion }>) {
  const [annotator] = useAnnotator();
  const review = useReviewIntent();
  const [editing, setEditing] = useState(false);
  const [sentences, setSentences] = useState(suggestion.sentences.join("\n"));
  const [name, setName] = useState(suggestion.intent);
  const [response, setResponse] = useState(suggestion.response ?? "");
  const isIntent = suggestion.kind === "intent";
  const usable = suggestion.status === "active" || suggestion.status === "proposed";
  const kind = KINDS.find((k) => k.id === suggestion.kind);
  const act = (action: "approve" | "correct" | "reject" | "restore") =>
    review.mutate(
      {
        id: suggestion.id,
        annotator,
        action,
        sentences:
          action === "correct"
            ? sentences
                .split("\n")
                .map((s) => s.trim())
                .filter(Boolean)
            : undefined,
        intent: action === "correct" && isIntent ? name.trim() : undefined,
        response: action === "correct" && isIntent ? response : undefined,
      },
      { onSuccess: () => setEditing(false) }
    );
  const notes = suggestion.validation?.notes ?? [];

  return (
    <div className="card rv-turn">
      <div className="rv-card-head">
        <span className={`rv-badge ${STATUS_TONE[suggestion.status] ?? "rv-badge-muted"}`}>
          {suggestion.status === "active" && !suggestion.reviewed_at ? "active · unreviewed" : suggestion.status}
        </span>
        <span className="rv-badge rv-badge-dreamer" title={kind?.title}>
          {kind?.label ?? suggestion.kind}
        </span>
        <span className="rv-badge rv-badge-muted">{suggestion.intent}</span>
        <span className="rv-badge rv-badge-muted">{suggestion.language}</span>
        <span
          className={`rv-badge ${suggestion.validated ? "rv-badge-good" : "rv-badge-warn"}`}
          title={
            suggestion.validated
              ? "Every evidence turn matches these sentences and yields the tool call the LLM made"
              : "Not fully checked against the evidence; read the notes"
          }
        >
          {suggestion.validated ? "checked" : "unchecked"}
        </span>
        {suggestion.confidence !== null && (
          <span className="rv-badge rv-badge-muted" title="Proposal confidence">
            {Math.round(suggestion.confidence * 100)}%
          </span>
        )}
        <span className="text-xs text-muted">
          {suggestion.source} · {formatDate(suggestion.created_at)}
          {suggestion.reviewed_by ? ` · reviewed by ${suggestion.reviewed_by.replace(/^human:/, "")}` : ""}
        </span>
      </div>
      {editing ? (
        <div className="d-flex flex-col gap-2">
          {isIntent && (
            <input
              className="input"
              value={name}
              maxLength={64}
              aria-label="Intent name"
              onChange={(e) => setName(e.target.value)}
            />
          )}
          <textarea
            className="input rv-mono"
            rows={Math.max(2, sentences.split("\n").length)}
            value={sentences}
            aria-label="Sentences, one per line"
            onChange={(e) => setSentences(e.target.value)}
          />
          {isIntent && (
            <input
              className="input"
              value={response}
              maxLength={300}
              aria-label="Spoken response"
              onChange={(e) => setResponse(e.target.value)}
            />
          )}
          <div className="rv-actions">
            <button
              className="btn-primary"
              disabled={!annotator || !sentences.trim() || review.isPending}
              onClick={() => act("correct")}
            >
              Save and use
            </button>
            <button className="btn-ghost" onClick={() => setEditing(false)}>
              Cancel
            </button>
            {review.error && <span className="text-xs rv-error">{review.error.message}</span>}
          </div>
        </div>
      ) : (
        <>
          {suggestion.sentences.map((sentence) => (
            <p key={sentence} className="rv-memory-text rv-mono">
              {sentence}
            </p>
          ))}
          {suggestion.slots && Object.keys(suggestion.slots).length > 0 && (
            <p className="text-xs text-muted rv-mono">slots: {JSON.stringify(suggestion.slots)}</p>
          )}
          {suggestion.tool && (
            <p className="text-xs text-muted rv-mono">
              runs {suggestion.tool.tool}({JSON.stringify(suggestion.tool.arguments)}) · says “{suggestion.response}”
            </p>
          )}
        </>
      )}
      <p className="text-xs text-muted">
        Seen {suggestion.occurrences} time{suggestion.occurrences === 1 ? "" : "s"} in {suggestion.sessions} session
        {suggestion.sessions === 1 ? "" : "s"} · answered {suggestion.live_uses} live turn
        {suggestion.live_uses === 1 ? "" : "s"}
      </p>
      {notes.length > 0 && <p className="text-xs text-muted">Check notes: {notes.join("; ")}</p>}
      <Evidence suggestion={suggestion} />
      {usable && !editing && (
        <div className="rv-actions">
          {!suggestion.reviewed_at && (
            <button className="btn-primary" disabled={!annotator || review.isPending} onClick={() => act("approve")}>
              {suggestion.status === "proposed" ? "Approve" : "Keep"}
            </button>
          )}
          <button className="btn-secondary" disabled={!annotator} onClick={() => setEditing(true)}>
            Correct
          </button>
          <button className="btn-ghost" disabled={!annotator || review.isPending} onClick={() => act("reject")}>
            Reject
          </button>
          {!annotator && <span className="text-xs rv-error">set your annotator name first</span>}
          {review.error && <span className="text-xs rv-error">{review.error.message}</span>}
        </div>
      )}
      {suggestion.status === "rejected" && (
        <div className="rv-actions">
          <button className="btn-secondary" disabled={!annotator || review.isPending} onClick={() => act("restore")}>
            Restore
          </button>
          {!annotator && <span className="text-xs rv-error">set your annotator name first</span>}
          {review.error && <span className="text-xs rv-error">{review.error.message}</span>}
        </div>
      )}
    </div>
  );
}

export function IntentsView() {
  const [filter, setFilter] = useState<IntentSuggestionFilter>("open");
  const [kind, setKind] = useState<IntentSuggestionKind | "">("");
  const [page, setPage] = useState(0);
  const pageSize = 20;
  const suggestions = useIntentSuggestions(filter, kind, page, pageSize);
  const pages = Math.max(1, Math.ceil((suggestions.data?.total ?? 0) / pageSize));
  // Languages with active suggestions, whatever the filter and page: what the export contains.
  const languages = suggestions.data?.languages ?? [];

  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Intents</h2>
          <p className="text-sm text-muted">
            Sentences and intents the dream job learned from turns the intent engine missed, most recurrent first.
            Approved ones are used by new sessions, which then answer these requests without the LLM.
          </p>
        </div>
        {languages.length > 0 && (
          <div className="d-flex gap-2">
            {languages.map((language) => (
              <a
                key={language}
                className="btn-ghost"
                href={intentExportUrl(language)}
                download={`learned-${language}.yaml`}
                title="Approved sentences and intents as sentence-file YAML, to commit them to the repository"
              >
                Export {language}
              </a>
            ))}
          </div>
        )}
      </div>
      <div className="rv-filters">
        <div className="rv-segmented">
          {FILTERS.map((f) => (
            <button
              key={f.id}
              className={filter === f.id ? "btn-primary" : "btn-ghost"}
              onClick={() => {
                setFilter(f.id);
                setPage(0);
              }}
            >
              {f.label}
            </button>
          ))}
        </div>
        <select
          className="input rv-select"
          value={kind}
          aria-label="Kind"
          onChange={(e) => {
            setKind(e.target.value as IntentSuggestionKind | "");
            setPage(0);
          }}
        >
          <option value="">Any kind</option>
          {KINDS.map((k) => (
            <option key={k.id} value={k.id}>
              {k.label}
            </option>
          ))}
        </select>
      </div>
      <StatusLine loading={suggestions.isLoading} error={suggestions.error} />
      {suggestions.data && suggestions.data.suggestions.length === 0 && (
        <p className="text-sm text-muted">
          {filter === "open" ? "Nothing to review." : "No suggestions."} The dream job proposes sentences from sessions
          that ran the intent engine (INTENT_ENGINE_ENABLED=true) when the engine missed a turn.
        </p>
      )}
      {suggestions.data?.suggestions.map((s) => (
        <SuggestionCard key={`${s.id}-${s.status}-${s.reviewed_at ?? ""}`} suggestion={s} />
      ))}
      {pages > 1 && (
        <div className="rv-pager text-sm">
          <button className="btn-ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>
            ← Previous
          </button>
          <span className="text-muted">
            Page {page + 1} / {pages}
          </span>
          <button className="btn-ghost" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>
            Next →
          </button>
        </div>
      )}
    </section>
  );
}
