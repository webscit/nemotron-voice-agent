// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import {
  MEMORY_CATEGORIES,
  useAddMemory,
  useMemories,
  usePeople,
  useReviewMemory,
  type MemoryCategory,
  type MemoryDetail,
  type MemoryFilter,
  type MemoryOrigin,
  type Person,
} from "./api";
import { AudioClips, StatusLine } from "./components";
import { formatDate, reviewHref, useAnnotator } from "./utils";

const FILTERS: { id: MemoryFilter; label: string }[] = [
  { id: "open", label: "To review" },
  { id: "usable", label: "In use" },
  { id: "proposed", label: "Proposed" },
  { id: "forgotten", label: "Forgotten" },
  { id: "all", label: "All" },
];

const ORIGINS: { id: MemoryOrigin; label: string; title: string }[] = [
  { id: "dreamer", label: "Dreamer", title: "Extracted from a recorded session by the dream job" },
  { id: "conversation", label: "Conversation", title: "Saved by the person during a conversation (remember tool)" },
  { id: "manual", label: "Manual", title: "Written or corrected by a reviewer" },
];

const STATUS_TONE: Record<string, string> = {
  active: "rv-badge-good",
  proposed: "rv-badge-warn",
  forgotten: "rv-badge-bad",
  superseded: "rv-badge-muted",
};

export function MemoryStatusBadge({ memory }: Readonly<{ memory: { status: string; reviewed_at: number | null } }>) {
  const label = memory.status === "active" && !memory.reviewed_at ? "active · unreviewed" : memory.status;
  return <span className={`rv-badge ${STATUS_TONE[memory.status] ?? "rv-badge-muted"}`}>{label}</span>;
}

export function MemoryOriginBadge({ origin }: Readonly<{ origin: MemoryOrigin | null }>) {
  const known = ORIGINS.find((o) => o.id === origin);
  if (!known) return null;
  return (
    <span className={`rv-badge ${origin === "manual" ? "rv-badge-muted" : `rv-badge-${origin}`}`} title={known.title}>
      {known.label}
    </span>
  );
}

/** ``human:<name>`` → ``<name>``; ``live:<session>`` → ``conversation <session>``. */
function reviewerLabel(reviewedBy: string): string {
  if (reviewedBy.startsWith("live:")) return `conversation ${reviewedBy.slice("live:".length)}`;
  return reviewedBy.replace(/^human:/, "");
}

function Evidence({ memory }: Readonly<{ memory: MemoryDetail }>) {
  if (!memory.evidence.length) {
    const why = memory.origin === "manual" ? "Written by a reviewer" : "Not recorded";
    return <p className="text-xs text-muted">{why} (no transcript evidence).</p>;
  }
  return (
    <div className="rv-evidence">
      {memory.evidence.map((ev) => (
        <div key={`${ev.session_id}-${ev.turn_idx}`} className="rv-side rv-side-user">
          <span className="rv-label">
            <a href={reviewHref("session", ev.session_id)}>
              {ev.session_id} · turn {ev.turn_idx}
            </a>
          </span>
          <AudioClips clips={ev.audio} />
          {ev.user_text && <p className="text-sm">“{ev.user_text}”</p>}
          {ev.quote && ev.quote !== ev.user_text && <p className="text-xs text-muted">quote: {ev.quote}</p>}
        </div>
      ))}
    </div>
  );
}

function MemoryCard({ memory, people }: Readonly<{ memory: MemoryDetail; people: Person[] }>) {
  const [annotator] = useAnnotator();
  const review = useReviewMemory();
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(memory.text);
  const [personId, setPersonId] = useState(memory.person_id);
  const usable = memory.status === "active" || memory.status === "proposed";
  const act = (action: "approve" | "correct" | "forget" | "restore") =>
    review.mutate(
      {
        id: memory.id,
        annotator,
        action,
        text: action === "correct" ? text : undefined,
        personId: action === "correct" && personId !== memory.person_id ? personId : undefined,
      },
      { onSuccess: () => setEditing(false) }
    );

  return (
    <div className="card rv-turn">
      <div className="rv-card-head">
        <MemoryStatusBadge memory={memory} />
        <MemoryOriginBadge origin={memory.origin} />
        <span className="rv-badge rv-badge-muted">{memory.person_name ?? memory.person_id}</span>
        {memory.category && <span className="rv-badge rv-badge-muted">{memory.category}</span>}
        {memory.confidence !== null && (
          <span className="rv-badge rv-badge-muted" title="Extraction confidence">
            {Math.round(memory.confidence * 100)}%
          </span>
        )}
        <span className="text-xs text-muted">
          {memory.source} · {formatDate(memory.created_at)}
          {memory.reviewed_by
            ? ` · ${memory.status === "forgotten" ? "forgotten" : "reviewed"} by ${reviewerLabel(memory.reviewed_by)}`
            : ""}
        </span>
      </div>
      {editing ? (
        <div className="rv-inline-edit">
          <input className="input" value={text} maxLength={500} onChange={(e) => setText(e.target.value)} />
          <select className="input rv-select" value={personId} onChange={(e) => setPersonId(e.target.value)}>
            {people.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <button
            className="btn-primary"
            disabled={!annotator || !text.trim() || review.isPending}
            onClick={() => act("correct")}
          >
            Save
          </button>
          <button className="btn-ghost" onClick={() => setEditing(false)}>
            Cancel
          </button>
        </div>
      ) : (
        <p className="rv-memory-text">{memory.text}</p>
      )}
      {memory.supersedes_text && (
        <p className="text-xs text-muted">
          Replaces: <span className="rv-diff-del">{memory.supersedes_text}</span>
        </p>
      )}
      <p className="text-xs text-muted">
        Used in {memory.used_in_sessions} session{memory.used_in_sessions === 1 ? "" : "s"} / {memory.used_in_replies}{" "}
        {memory.used_in_replies === 1 ? "reply" : "replies"}
      </p>
      <Evidence memory={memory} />
      {usable && !editing && (
        <div className="rv-actions">
          {(memory.status === "proposed" || !memory.reviewed_at) && (
            <button className="btn-primary" disabled={!annotator || review.isPending} onClick={() => act("approve")}>
              {memory.status === "proposed" ? "Approve" : "Keep"}
            </button>
          )}
          <button className="btn-secondary" disabled={!annotator} onClick={() => setEditing(true)}>
            Correct
          </button>
          <button className="btn-ghost" disabled={!annotator || review.isPending} onClick={() => act("forget")}>
            Forget
          </button>
          {!annotator && <span className="text-xs rv-error">set your annotator name first</span>}
          {review.error && <span className="text-xs rv-error">{review.error.message}</span>}
        </div>
      )}
      {memory.status === "forgotten" && (
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

function AddMemoryForm({ personId }: Readonly<{ personId: string }>) {
  const [annotator] = useAnnotator();
  const add = useAddMemory();
  const [text, setText] = useState("");
  const [category, setCategory] = useState<MemoryCategory>("other");
  return (
    <form
      className="card rv-inline-edit"
      onSubmit={(e) => {
        e.preventDefault();
        add.mutate({ personId, annotator, text: text.trim(), category }, { onSuccess: () => setText("") });
      }}
    >
      <input
        className="input"
        value={text}
        maxLength={500}
        placeholder="Add a memory, e.g. “Alice is vegetarian.”"
        aria-label="New memory"
        onChange={(e) => setText(e.target.value)}
      />
      <select
        className="input rv-select"
        value={category}
        aria-label="Category"
        onChange={(e) => setCategory(e.target.value as MemoryCategory)}
      >
        {MEMORY_CATEGORIES.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
      <button className="btn-primary" type="submit" disabled={!annotator || !text.trim() || add.isPending}>
        Add memory
      </button>
      {!annotator && <span className="text-xs rv-error">set your annotator name first</span>}
      {add.error && <span className="text-xs rv-error">{add.error.message}</span>}
    </form>
  );
}

export function MemoriesView({ personId = "" }: Readonly<{ personId?: string }>) {
  const [filter, setFilter] = useState<MemoryFilter>(personId ? "usable" : "open");
  const [person, setPerson] = useState(personId);
  const [origin, setOrigin] = useState<MemoryOrigin | "">("");
  const [page, setPage] = useState(0);
  const people = usePeople(true);
  const pageSize = 20;
  const memories = useMemories(filter, person, page, pageSize, origin);
  const pages = Math.max(1, Math.ceil((memories.data?.total ?? 0) / pageSize));
  const peopleList = people.data?.people ?? [];

  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Memories</h2>
          <p className="text-sm text-muted">
            What the agent remembers about people. Confident memories, and those people ask the agent to remember, are
            used live right away and reviewed here afterwards; the others wait for approval.
          </p>
        </div>
      </div>
      {personId && <AddMemoryForm personId={personId} />}
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
          value={person}
          onChange={(e) => {
            setPerson(e.target.value);
            setPage(0);
          }}
        >
          <option value="">Everyone</option>
          {peopleList.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name}
              {p.archived ? " (archived)" : ""}
            </option>
          ))}
        </select>
        <select
          className="input rv-select"
          value={origin}
          aria-label="Origin"
          onChange={(e) => {
            setOrigin(e.target.value as MemoryOrigin | "");
            setPage(0);
          }}
        >
          <option value="">Any origin</option>
          {ORIGINS.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      <StatusLine loading={memories.isLoading} error={memories.error} />
      {memories.data && memories.data.memories.length === 0 && (
        <p className="text-sm text-muted">
          {filter === "open" ? "Nothing to review." : "No memories."} Memories are extracted by the dream job from
          sessions attributed to a person (pick one in Who&apos;s talking, or on a session page), saved when a person
          asks the agent to remember something, or added on a person&apos;s page.
        </p>
      )}
      {memories.data?.memories.map((m) => (
        <MemoryCard key={`${m.id}-${m.status}-${m.reviewed_at ?? ""}`} memory={m} people={peopleList} />
      ))}
      {pages > 1 && (
        <div className="rv-pager text-sm">
          <button className="btn-ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>
            ← Newer
          </button>
          <span className="text-muted">
            Page {page + 1} / {pages}
          </span>
          <button className="btn-ghost" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>
            Older →
          </button>
        </div>
      )}
    </section>
  );
}
