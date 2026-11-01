// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// "Decision reference" activity: set the right decision-tool choice of turns (e.g. the
// robot's emotion). The saved choice is the ground truth of the redecide agreement.

import { useEffect, useEffectEvent, useState, type ReactNode } from "react";
import {
  useDecisionQueue,
  useSaveDecisionReferences,
  type AsrQueueSort,
  type DecisionChoicesMap,
  type DecisionQueueItem,
} from "./api";
import { StatusLine } from "./components";
import { formatDate, reviewHref, useAnnotator } from "./utils";

type Status = "open" | "reviewed" | "all";

const PAGE_SIZE = 20;

function itemKey(item: DecisionQueueItem): string {
  return `${item.session_id}:${item.turn_idx}`;
}

/** The choices a review starts from: the human ones, else the reference model's, else the live ones. */
function defaultChoices(item: DecisionQueueItem): DecisionChoicesMap {
  const reference = item.reference_source ? item.models[item.reference_source] : undefined;
  return { ...item.live.choices, ...reference, ...item.human_reference?.choices };
}

/** The judged exchange, one line per speaker ("User: ...", "Assistant: ..."). */
function Exchange({ state }: Readonly<{ state: string }>) {
  return (
    <div className="rv-transcripts">
      {state.split("\n").map((line, i) => {
        const [speaker, ...rest] = line.split(": ");
        return (
          <p key={`${i}-${speaker}`} className="text-sm">
            <span className="rv-label">{speaker}</span> {rest.join(": ")}
          </p>
        );
      })}
    </div>
  );
}

function statusBadge(item: DecisionQueueItem) {
  if (item.human_reference) return <span className="rv-badge rv-badge-good">reviewed</span>;
  if (item.disagree) return <span className="rv-badge rv-badge-bad">live ≠ reference</span>;
  if (item.agree) return <span className="rv-badge rv-badge-good">agree</span>;
  return <span className="rv-badge rv-badge-muted">awaiting reference</span>;
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
  item: DecisionQueueItem;
  position: number;
  count: number;
  onSave: (choices: DecisionChoicesMap) => void;
  onNext: () => void;
  onPrevious: () => void;
  saving: boolean;
}>) {
  const hasPrevious = position > 0;
  const hasNext = position < count - 1;
  const [draft, setDraft] = useState(() => defaultChoices(item));
  const reference = item.reference_source ? item.models[item.reference_source] : undefined;
  const candidates = Object.entries(item.models).filter(([source]) => source !== item.reference_source);
  const tools = Object.keys(item.options);
  const complete = tools.every((tool) => draft[tool]);

  const onKey = useEffectEvent((event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
    if (event.key === "Enter") {
      event.preventDefault();
      if (complete) onSave(draft);
    } else if (event.key === "1") {
      setDraft({ ...draft, ...item.live.choices });
    } else if (event.key === "2" && reference) {
      setDraft({ ...draft, ...reference });
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

  const sourceRow = (label: ReactNode, choices: DecisionChoicesMap, key: string) => (
    <tr key={key}>
      <th>{label}</th>
      <td>
        {Object.entries(choices).map(([tool, choice]) => (
          <span key={tool} className="rv-badge rv-badge-muted" title={item.options[tool]?.[choice] ?? tool}>
            {tools.length > 1 ? `${tool}: ` : ""}
            {choice}
          </span>
        ))}
      </td>
    </tr>
  );

  return (
    <div className="card rv-card">
      <div className="rv-card-head">
        <span className="text-sm text-muted">
          {position + 1} / {count} · <a href={reviewHref("session", item.session_id)}>{item.session_id}</a> · turn{" "}
          {item.turn_idx} · {item.language ?? "?"} · {formatDate(item.started_at)}
        </span>
        {statusBadge(item)}
        {item.human_reference && (
          <span className="rv-badge rv-badge-good">reviewed by {item.human_reference.source.slice(6)}</span>
        )}
      </div>

      <Exchange state={item.state} />

      <table className="rv-transcripts">
        <tbody>
          {sourceRow(
            <>
              <kbd>1</kbd> {item.live.source}
            </>,
            item.live.choices,
            "live"
          )}
          {reference &&
            sourceRow(
              <>
                <kbd>2</kbd> {item.reference_source}
              </>,
              reference,
              "reference"
            )}
          {candidates.map(([source, choices]) => sourceRow(source, choices, source))}
        </tbody>
      </table>

      {tools.map((tool) => (
        <div key={tool}>
          <span className="rv-label">Human reference · {tool}</span>
          <div className="rv-chips" role="group" aria-label={`${tool} choice`}>
            {Object.entries(item.options[tool]).map(([choice, description]) => (
              <button
                key={choice}
                className={`rv-chip ${draft[tool] === choice ? "active" : ""}`}
                aria-pressed={draft[tool] === choice}
                title={description}
                onClick={() => setDraft({ ...draft, [tool]: choice })}
              >
                {choice}
              </button>
            ))}
          </div>
        </div>
      ))}

      <div className="rv-actions">
        <button className="btn-ghost" onClick={onPrevious} disabled={!hasPrevious} title="←">
          ← Previous
        </button>
        <button className="btn-ghost" onClick={onNext} disabled={!hasNext} title="→ or s">
          Next →
        </button>
        <button className="btn-primary" onClick={() => onSave(draft)} disabled={saving || !complete} title="Enter">
          {saving ? "Saving…" : "Save reference"}
        </button>
      </div>
      <p className="text-xs text-muted">
        Shortcuts: <kbd>1</kbd>/<kbd>2</kbd> take live/reference · <kbd>enter</kbd> save · <kbd>←</kbd> previous ·{" "}
        <kbd>→</kbd>/<kbd>s</kbd> next · hover a choice for its description
      </p>
    </div>
  );
}

/** Paginated list of the turns in the current queue order; the selected turn is highlighted. */
function TurnList({
  items,
  selectedIndex,
  page,
  onPage,
  onSelect,
}: Readonly<{
  items: DecisionQueueItem[];
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
            <th>Choice</th>
            <th>Exchange</th>
          </tr>
        </thead>
        <tbody>
          {items.slice(first, first + PAGE_SIZE).map((item, offset) => {
            const index = first + offset;
            const shown = item.human_reference?.choices ?? item.live.choices;
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
                <td className="rv-nowrap">{Object.values(shown).join(", ")}</td>
                <td className="rv-ellipsis">{item.state.replaceAll("\n", " · ")}</td>
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

export function DecisionReview() {
  const [status, setStatus] = useState<Status>("open");
  const [sort, setSort] = useState<AsrQueueSort>("recent");
  // Selection follows the item (by key) across refetches; when it disappears (saved in the
  // open queue) the item now at the same position is selected, i.e. the next one.
  const [selected, setSelected] = useState<{ key: string; index: number }>({ key: "", index: 0 });
  // null = the list page follows the selected turn.
  const [pageOverride, setPageOverride] = useState<number | null>(null);
  const [annotator] = useAnnotator();
  const queue = useDecisionQueue(status, sort, 1000);
  const save = useSaveDecisionReferences();
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

  const referenceItems = (target: DecisionQueueItem, choices: DecisionChoicesMap) =>
    Object.entries(choices).map(([tool, choice]) => ({
      session_id: target.session_id,
      turn_idx: target.turn_idx,
      tool,
      choice,
    }));

  const saveChoices = (target: DecisionQueueItem, choices: DecisionChoicesMap) => {
    if (!annotator) return;
    save.mutate(
      { annotator, items: referenceItems(target, choices) },
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
    save.mutate({ annotator, items: agreeing.flatMap((i) => referenceItems(i, i.live.choices)) });
  };

  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Decision reference</h2>
          <p className="text-sm text-muted">
            Set what the decision tools should have chosen, for example the robot&apos;s emotion. The choice you save
            becomes the ground truth of the agreement scores and is rescored immediately.
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
            <option value="disagreement">Disagreement first</option>
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
            onSave={(choices) => saveChoices(item, choices)}
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
