// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useEffectEvent, useRef, useState } from "react";
import { artifactUrl, useAsrQueue, useSaveReferences, type AsrQueueItem, type AsrQueueSort } from "./api";
import { DiffText, StatusLine } from "./components";
import { formatDate, formatPct, reviewHref, useAnnotator } from "./utils";

type Status = "open" | "reviewed" | "all";

const PAGE_SIZE = 20;

function itemKey(item: AsrQueueItem): string {
  return `${item.session_id}:${item.turn_idx}`;
}

function defaultText(item: AsrQueueItem): string {
  if (item.human_reference) return item.human_reference.text;
  if (item.reference_source && item.models[item.reference_source] !== undefined) {
    return item.models[item.reference_source];
  }
  return item.live.text;
}

/** One turn under review. Keyed by item so the draft resets when the item changes. */
function ReviewCard({
  item,
  position,
  count,
  onSave,
  onNext,
  onPrevious,
  saving,
}: Readonly<{
  item: AsrQueueItem;
  position: number;
  count: number;
  onSave: (text: string) => void;
  onNext: () => void;
  onPrevious: () => void;
  saving: boolean;
}>) {
  const hasPrevious = position > 0;
  const hasNext = position < count - 1;
  const [draft, setDraft] = useState(() => defaultText(item));
  const audioRefs = useRef<HTMLAudioElement[]>([]);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const reference = item.reference_source ? item.models[item.reference_source] : undefined;
  const candidates = Object.entries(item.models).filter(([source]) => source !== item.reference_source);

  const playAll = () => {
    const clips = audioRefs.current.filter(Boolean);
    const first = clips[0];
    if (!first) return;
    if (!first.paused) {
      clips.forEach((clip) => clip.pause());
      return;
    }
    clips.forEach((clip, i) => {
      clip.onended = () => {
        const next = clips[i + 1];
        if (next) void next.play();
      };
    });
    first.currentTime = 0;
    void first.play();
  };

  // Keyboard shortcuts while the text box is not focused; Ctrl/Cmd+Enter works everywhere.
  const onKey = useEffectEvent((event: KeyboardEvent) => {
    const inText = event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement;
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      onSave(draft);
      return;
    }
    if (inText) {
      if (event.key === "Escape") (event.target as HTMLElement).blur();
      return;
    }
    if (event.key === " ") {
      event.preventDefault();
      playAll();
    } else if (event.key === "Enter") {
      event.preventDefault();
      onSave(draft);
    } else if (event.key === "1") {
      setDraft(item.live.text);
    } else if (event.key === "2" && reference !== undefined) {
      setDraft(reference);
    } else if (event.key === "e") {
      event.preventDefault();
      textRef.current?.focus();
    } else if ((event.key === "s" || event.key === "ArrowRight") && hasNext) {
      onNext();
    } else if (event.key === "ArrowLeft" && hasPrevious) {
      onPrevious();
    }
  });
  useEffect(() => {
    const listener = (event: KeyboardEvent) => onKey(event);
    globalThis.addEventListener("keydown", listener);
    return () => globalThis.removeEventListener("keydown", listener);
  }, []);

  return (
    <div className="card rv-card">
      <div className="rv-card-head">
        <span className="text-sm text-muted">
          {position + 1} / {count} · <a href={reviewHref("session", item.session_id)}>{item.session_id}</a> · turn{" "}
          {item.turn_idx} · {item.language ?? "?"} · {formatDate(item.started_at)}
        </span>
        {item.disagreement === null ? (
          <span className="rv-badge rv-badge-muted">awaiting {item.reference_source ?? "reference"}</span>
        ) : (
          <span className={`rv-badge ${item.agree ? "rv-badge-good" : "rv-badge-bad"}`}>
            {item.agree ? "live = reference" : `live vs reference ${formatPct(item.disagreement)}`}
          </span>
        )}
        {item.human_reference && (
          <span className="rv-badge rv-badge-good">reviewed by {item.human_reference.source.slice(6)}</span>
        )}
      </div>

      <div className="rv-audio">
        {item.audio.map((clip, i) => (
          <audio
            key={clip.key}
            ref={(el) => {
              if (el) audioRefs.current[i] = el;
            }}
            controls
            preload="auto"
            src={artifactUrl(clip.key)}
          />
        ))}
      </div>

      <table className="rv-transcripts">
        <tbody>
          <tr>
            <th>
              <kbd>1</kbd> {item.live.source}
            </th>
            <td>
              {reference === undefined ? item.live.text : <DiffText reference={reference} hypothesis={item.live.text} />}
            </td>
          </tr>
          {reference !== undefined && (
            <tr>
              <th>
                <kbd>2</kbd> {item.reference_source}
              </th>
              <td>{reference || <span className="text-muted">(empty)</span>}</td>
            </tr>
          )}
          {candidates.map(([source, text]) => (
            <tr key={source}>
              <th>{source}</th>
              <td>{reference === undefined ? text : <DiffText reference={reference} hypothesis={text} />}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <label className="rv-label" htmlFor="rv-reference">
        Human reference <kbd>e</kbd> to edit
      </label>
      <textarea
        id="rv-reference"
        ref={textRef}
        className="input rv-textarea"
        rows={3}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
      />
      <div className="rv-actions">
        <button className="btn-ghost" onClick={onPrevious} disabled={!hasPrevious} title="←">
          ← Previous
        </button>
        <button className="btn-ghost" onClick={onNext} disabled={!hasNext} title="→ or s">
          Next →
        </button>
        <button className="btn-primary" onClick={() => onSave(draft)} disabled={saving || !draft.trim()} title="Enter">
          {saving ? "Saving…" : "Save reference"}
        </button>
      </div>
      <p className="text-xs text-muted">
        Shortcuts: <kbd>space</kbd> play · <kbd>1</kbd>/<kbd>2</kbd> take live/reference · <kbd>e</kbd> edit ·{" "}
        <kbd>enter</kbd> save (<kbd>ctrl</kbd>+<kbd>enter</kbd> while editing) · <kbd>←</kbd> previous ·{" "}
        <kbd>→</kbd>/<kbd>s</kbd> next
      </p>
    </div>
  );
}

function statusBadge(item: AsrQueueItem) {
  if (item.human_reference) return <span className="rv-badge rv-badge-good">reviewed</span>;
  if (item.disagreement === null) return <span className="rv-badge rv-badge-muted">awaiting reference</span>;
  if (item.agree) return <span className="rv-badge rv-badge-good">agree</span>;
  return <span className="rv-badge rv-badge-bad">{formatPct(item.disagreement)}</span>;
}

/** Paginated list of the turns in the current queue order; the selected turn is highlighted. */
function TurnList({
  items,
  selectedIndex,
  page,
  onPage,
  onSelect,
}: Readonly<{
  items: AsrQueueItem[];
  selectedIndex: number;
  page: number;
  onPage: (page: number) => void;
  onSelect: (index: number) => void;
}>) {
  const pages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
  const first = page * PAGE_SIZE;
  return (
    <div className="rv-turn-list">
      <table className="rv-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Time</th>
            <th>Session · turn</th>
            <th>Status</th>
            <th>Live transcript</th>
          </tr>
        </thead>
        <tbody>
          {items.slice(first, first + PAGE_SIZE).map((item, offset) => {
            const index = first + offset;
            return (
              <tr
                key={itemKey(item)}
                className={index === selectedIndex ? "rv-selected" : ""}
                aria-selected={index === selectedIndex}
                onClick={() => onSelect(index)}
              >
                <td className="text-muted">{index + 1}</td>
                <td className="rv-nowrap">{formatDate(item.started_at)}</td>
                <td className="rv-mono text-xs rv-nowrap">
                  {item.session_id} · {item.turn_idx}
                </td>
                <td>{statusBadge(item)}</td>
                <td className="rv-ellipsis">{item.human_reference?.text ?? item.live.text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {pages > 1 && (
        <div className="rv-pager text-sm">
          <button className="btn-ghost" disabled={page === 0} onClick={() => onPage(page - 1)}>
            ← Newer
          </button>
          <span className="text-muted">
            Page {page + 1} / {pages}
          </span>
          <button className="btn-ghost" disabled={page >= pages - 1} onClick={() => onPage(page + 1)}>
            Older →
          </button>
        </div>
      )}
    </div>
  );
}

export function AsrReview() {
  const [status, setStatus] = useState<Status>("open");
  const [sort, setSort] = useState<AsrQueueSort>("recent");
  // Selection follows the item (by key) across refetches; when it disappears (saved in the
  // open queue) the item now at the same position is selected, i.e. the next one.
  const [selected, setSelected] = useState<{ key: string; index: number }>({ key: "", index: 0 });
  // null = the list page follows the selected turn.
  const [pageOverride, setPageOverride] = useState<number | null>(null);
  const [annotator] = useAnnotator();
  const queue = useAsrQueue(status, sort, 1000);
  const save = useSaveReferences();
  const items = queue.data?.items ?? [];
  const found = items.findIndex((i) => itemKey(i) === selected.key);
  const index = found >= 0 ? found : Math.min(selected.index, Math.max(items.length - 1, 0));
  const item = items[index];
  const page = pageOverride ?? Math.floor(index / PAGE_SIZE);
  const agreeing = items.filter((i) => i.agree && !i.human_reference);

  const select = (next: number) => {
    const target = items[next];
    if (!target) return;
    setSelected({ key: itemKey(target), index: next });
    setPageOverride(null);
  };

  const reset = () => {
    setSelected({ key: "", index: 0 });
    setPageOverride(null);
  };

  const saveText = (target: AsrQueueItem, text: string) => {
    if (!annotator || !text.trim()) return;
    save.mutate(
      { annotator, items: [{ session_id: target.session_id, turn_idx: target.turn_idx, text }] },
      {
        onSuccess: () => {
          // In the open queue the saved turn leaves the list and its successor takes its position.
          if (status === "open") setSelected({ key: "", index });
          else select(Math.min(index + 1, items.length - 1));
        },
      }
    );
  };

  const acceptAgreeing = () => {
    if (!annotator || !agreeing.length) return;
    save.mutate({
      annotator,
      items: agreeing.map((i) => ({ session_id: i.session_id, turn_idx: i.turn_idx, text: i.live.text })),
    });
  };

  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">ASR reference</h2>
          <p className="text-sm text-muted">
            Set what the user actually said. The reference you save becomes the WER ground truth and is rescored
            immediately.
          </p>
        </div>
        <div className="rv-segmented">
          {(["open", "reviewed", "all"] as Status[]).map((s) => (
            <button
              key={s}
              className={`tab-btn ${status === s ? "active" : ""}`}
              onClick={() => {
                setStatus(s);
                reset();
              }}
            >
              {s}
            </button>
          ))}
          <select
            className="input rv-select"
            value={sort}
            aria-label="Order"
            onChange={(event) => {
              setSort(event.target.value as AsrQueueSort);
              reset();
            }}
          >
            <option value="recent">Most recent first</option>
            <option value="disagreement">Most disagreement first</option>
          </select>
        </div>
      </div>

      {queue.data && (
        <div className="rv-stats text-sm">
          <span>
            <strong>{queue.data.open}</strong> open
          </span>
          <span>
            <strong>{queue.data.total - queue.data.open}</strong> reviewed
          </span>
          {status === "open" && agreeing.length > 0 && (
            <button className="btn-secondary" onClick={acceptAgreeing} disabled={!annotator || save.isPending}>
              Accept {agreeing.length} turns where live = reference
            </button>
          )}
        </div>
      )}
      {!annotator && <p className="text-sm rv-error">Set your annotator name (top right) to save references.</p>}
      {save.error && <p className="text-sm rv-error">{save.error.message}</p>}
      <StatusLine loading={queue.isLoading} error={queue.error} />

      {item ? (
        <>
          <ReviewCard
            key={itemKey(item)}
            item={item}
            position={index}
            count={items.length}
            saving={save.isPending}
            onSave={(text) => saveText(item, text)}
            onNext={() => select(index + 1)}
            onPrevious={() => select(index - 1)}
          />
          <TurnList items={items} selectedIndex={index} page={page} onPage={setPageOverride} onSelect={select} />
        </>
      ) : (
        queue.data && <p className="text-sm text-muted">Nothing to review here.</p>
      )}
    </section>
  );
}
