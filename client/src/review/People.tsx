// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import {
  useCreatePerson,
  useDuplicatePeople,
  useForgetIdentity,
  useMergePeople,
  usePeople,
  usePersonIdentity,
  useUpdatePerson,
  type DuplicatePair,
  type IdentityModel,
  type Person,
} from "./api";
import { StatusLine } from "./components";
import { MemoriesView } from "./Memories";
import { formatDate, reviewHref, useAnnotator } from "./utils";

const MODALITY_LABELS: Record<string, string> = { voice: "Voice", face: "Face", other: "Other" };

function modelLabel(model: IdentityModel): string {
  return model.modality ? MODALITY_LABELS[model.modality] : model.model;
}

function IdentityCounts({ person }: Readonly<{ person: Person }>) {
  const samples = person.identity_samples ?? {};
  const kinds = Object.entries(samples).filter(([, count]) => count);
  if (!kinds.length) return <span className="text-muted">—</span>;
  return (
    <span className="rv-card-head">
      {kinds.map(([kind, count]) => (
        <span key={kind} className="rv-badge rv-badge-muted" title={`${count} ${kind} sample(s) bound at enrollment`}>
          {MODALITY_LABELS[kind] ?? kind} {count}
        </span>
      ))}
    </span>
  );
}

function VoiceIdTurns({ person }: Readonly<{ person: Person }>) {
  const turns = person.voice_id_turns;
  if (!turns?.turns) return <span className="text-muted">—</span>;
  return (
    <span title="Turns attributed live by voice ID; verified = the speaker's face confirmed the voice">
      {turns.turns}
      {turns.verified > 0 && <span className="text-muted"> ({turns.verified} verified)</span>}
    </span>
  );
}

/** Scores and reasons why two people may be the same person. */
function DuplicateEvidence({ pair }: Readonly<{ pair: DuplicatePair }>) {
  return (
    <span className="rv-card-head">
      {pair.same_name && <span className="rv-badge rv-badge-warn">same name</span>}
      {pair.scores.map((s) => (
        <span
          key={s.model}
          className={`rv-badge ${s.score >= 0.6 ? "rv-badge-warn" : "rv-badge-muted"}`}
          title={`Cosine similarity of the two people's ${s.modality ?? s.model} centroids (model ${s.model})`}
        >
          {s.modality ? MODALITY_LABELS[s.modality] : s.model} {s.score.toFixed(2)}
        </span>
      ))}
    </span>
  );
}

/** "Keep X" buttons: merge the other person of the pair into X. */
function MergeButtons({ pair, onMerged }: Readonly<{ pair: DuplicatePair; onMerged?: (targetId: string) => void }>) {
  const [annotator] = useAnnotator();
  const merge = useMergePeople();
  const [a, b] = pair.people;
  const keep = (target: Person, source: Person) => {
    const question =
      `Merge "${source.name}" into "${target.name}"?\n\n` +
      `Turns, voice and face samples and memories of "${source.name}" move to "${target.name}", ` +
      `and "${source.name}" is deleted. This cannot be undone.`;
    if (!globalThis.confirm(question)) return;
    merge.mutate(
      { targetId: target.id, sourceId: source.id, annotator },
      { onSuccess: () => onMerged?.(target.id) }
    );
  };
  return (
    <span className="rv-nowrap">
      {[
        [a, b],
        [b, a],
      ].map(([target, source]) => (
        <button
          key={target.id}
          className="btn-ghost text-xs"
          disabled={!annotator || target.archived || merge.isPending}
          title={
            target.archived
              ? "Unarchive this person to keep them"
              : `Keep "${target.name}" and merge "${source.name}" into them`
          }
          onClick={() => keep(target, source)}
        >
          Keep {target.name}
        </button>
      ))}
      {merge.error && <span className="text-xs rv-error"> {String(merge.error)}</span>}
    </span>
  );
}

function DuplicatesPanel() {
  const [annotator] = useAnnotator();
  const duplicates = useDuplicatePeople();
  const pairs = duplicates.data?.duplicates ?? [];
  if (!pairs.length) return null;
  return (
    <div className="rv-section">
      <h3 className="font-semibold">Possible duplicates</h3>
      <p className="text-sm text-muted">
        People whose enrolled voices or faces are similar, or who share a name. Merge them when they are the same
        person; the one you keep keeps its name.
        {!annotator && <span className="rv-error"> Set your annotator name first.</span>}
      </p>
      <table className="rv-table rv-table-static">
        <thead>
          <tr>
            <th>People</th>
            <th>Similarity</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {pairs.map((pair) => (
            <tr key={pair.people.map((p) => p.id).join(":")}>
              <td>
                {pair.people.map((p, i) => (
                  <span key={p.id}>
                    {i > 0 && " · "}
                    <a href={reviewHref("people", p.id)}>{p.name}</a>
                    {p.archived && <span className="text-muted"> (archived)</span>}
                  </span>
                ))}
              </td>
              <td>
                <DuplicateEvidence pair={pair} />
              </td>
              <td>
                <MergeButtons pair={pair} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MergeInto({ person, people }: Readonly<{ person: Person; people: Person[] }>) {
  const [annotator] = useAnnotator();
  const merge = useMergePeople();
  const [sourceId, setSourceId] = useState("");
  const candidates = people.filter((p) => p.id !== person.id);
  const source = candidates.find((p) => p.id === sourceId);
  return (
    <form
      className="rv-inline-edit"
      onSubmit={(e) => {
        e.preventDefault();
        if (!source) return;
        const question =
          `Merge "${source.name}" into "${person.name}"?\n\n` +
          `Everything known about "${source.name}" moves to "${person.name}", and "${source.name}" is deleted. ` +
          "This cannot be undone.";
        if (globalThis.confirm(question)) {
          merge.mutate({ targetId: person.id, sourceId: source.id, annotator }, { onSuccess: () => setSourceId("") });
        }
      }}
    >
      <span className="text-sm text-muted">Same person as</span>
      <select className="input" value={sourceId} onChange={(e) => setSourceId(e.target.value)}>
        <option value="">pick someone to merge into {person.name}…</option>
        {candidates.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
            {p.archived ? " (archived)" : ""}
          </option>
        ))}
      </select>
      <button
        className="btn-secondary"
        type="submit"
        disabled={!annotator || !source || person.archived || merge.isPending}
        title={person.archived ? "Unarchive this person before merging others into them" : undefined}
      >
        Merge
      </button>
      {merge.error && <span className="text-xs rv-error">{String(merge.error)}</span>}
    </form>
  );
}

function IdentitySection({ person, people }: Readonly<{ person: Person; people: Person[] }>) {
  const [annotator] = useAnnotator();
  const identity = usePersonIdentity(person.id);
  const forget = useForgetIdentity();
  const models = identity.data?.models ?? [];
  const duplicates = identity.data?.duplicates ?? [];
  const forgetModel = (model: IdentityModel) => {
    const label = modelLabel(model).toLowerCase();
    if (globalThis.confirm(`Delete the ${model.count} ${label} sample(s) of "${person.name}"? They must enroll again.`)) {
      forget.mutate({ personId: person.id, annotator, model: model.model });
    }
  };
  return (
    <section className="rv-page rv-section">
      <h3 className="font-semibold">Voice and face identification</h3>
      <StatusLine loading={identity.isLoading} error={identity.error ?? forget.error} />
      {identity.data && !models.length && (
        <p className="text-sm text-muted">
          No voice or face enrolled. {person.name} is recognised only when picked in Who&apos;s talking or attributed in
          review.
        </p>
      )}
      {models.length > 0 && (
        <table className="rv-table rv-table-static">
          <thead>
            <tr>
              <th>Kind</th>
              <th>Model</th>
              <th>Samples</th>
              <th>Enrolled</th>
              <th>From sessions</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.model}>
                <td>{modelLabel(m)}</td>
                <td className="text-xs rv-mono">{m.model}</td>
                <td>{m.count}</td>
                <td className="text-xs">
                  {formatDate(m.first_at)}
                  {m.last_at !== m.first_at && <> – {formatDate(m.last_at)}</>}
                </td>
                <td className="text-xs">
                  {m.sessions.map((sessionId, i) => (
                    <span key={sessionId}>
                      {i > 0 && ", "}
                      <a className="rv-mono" href={reviewHref("sessions", sessionId)}>
                        {sessionId.slice(0, 8)}
                      </a>
                    </span>
                  ))}
                </td>
                <td>
                  <button
                    className="btn-ghost text-xs"
                    disabled={!annotator || forget.isPending}
                    title="Delete these samples, e.g. when they belong to someone else"
                    onClick={() => forgetModel(m)}
                  >
                    Forget
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {duplicates.length > 0 && (
        <>
          <h4 className="text-sm font-semibold">May be the same person as</h4>
          <table className="rv-table rv-table-static">
            <tbody>
              {duplicates.map((pair) => {
                const other = pair.people.find((p) => p.id !== person.id) ?? pair.people[0];
                return (
                  <tr key={other.id}>
                    <td>
                      <a href={reviewHref("people", other.id)}>{other.name}</a>
                    </td>
                    <td>
                      <DuplicateEvidence pair={pair} />
                    </td>
                    <td>
                      <MergeButtons
                        pair={pair}
                        onMerged={(targetId) => {
                          if (targetId !== person.id) globalThis.location.hash = reviewHref("people", targetId);
                        }}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
      <MergeInto person={person} people={people} />
      {!annotator && <span className="text-xs rv-error">Set your annotator name to merge or forget.</span>}
    </section>
  );
}

function PersonRow({ person }: Readonly<{ person: Person }>) {
  const update = useUpdatePerson();
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(person.name);
  const counts = person.memories ?? {};
  return (
    <tr className={person.archived ? "text-muted" : undefined}>
      <td>
        {editing ? (
          <form
            className="rv-inline-edit"
            onSubmit={(e) => {
              e.preventDefault();
              if (name.trim()) update.mutate({ id: person.id, name }, { onSuccess: () => setEditing(false) });
            }}
          >
            <input className="input" value={name} maxLength={128} onChange={(e) => setName(e.target.value)} />
            <button className="btn-primary" type="submit" disabled={!name.trim() || update.isPending}>
              Save
            </button>
            <button className="btn-ghost" type="button" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </form>
        ) : (
          <a href={reviewHref("people", person.id)}>{person.name}</a>
        )}
      </td>
      <td>{person.sessions ?? 0}</td>
      <td>
        <IdentityCounts person={person} />
      </td>
      <td>
        <VoiceIdTurns person={person} />
      </td>
      <td>{counts.active ?? 0}</td>
      <td>{counts.proposed ?? 0}</td>
      <td>{counts.forgotten ?? 0}</td>
      <td className="text-xs">{formatDate(person.created_at)}</td>
      <td className="rv-nowrap">
        {!editing && (
          <button className="btn-ghost text-xs" onClick={() => setEditing(true)}>
            Rename
          </button>
        )}
        <button
          className="btn-ghost text-xs"
          disabled={update.isPending}
          onClick={() => update.mutate({ id: person.id, archived: !person.archived })}
          title={person.archived ? "Show again in Who's talking" : "Hide from Who's talking (memories are kept)"}
        >
          {person.archived ? "Unarchive" : "Archive"}
        </button>
      </td>
    </tr>
  );
}

export function PeopleView({ personId = "" }: Readonly<{ personId?: string }>) {
  const people = usePeople(true);
  const create = useCreatePerson();
  const [name, setName] = useState("");
  const selected = people.data?.people.find((p) => p.id === personId);

  if (personId) {
    return (
      <>
        <section className="rv-page">
          <a className="text-sm" href={reviewHref("people")}>
            ← People
          </a>
          <h2 className="text-lg font-semibold">{selected?.name ?? personId}</h2>
          <p className="text-sm text-muted">
            {selected?.sessions ?? 0} attributed session(s). Memories below are what the agent knows about{" "}
            {selected?.name ?? "this person"}.
          </p>
        </section>
        {selected && <IdentitySection person={selected} people={people.data?.people ?? []} />}
        {people.data && !selected && (
          <p className="rv-page text-sm text-muted">
            This person no longer exists; they may have been merged into someone else.
          </p>
        )}
        <MemoriesView key={personId} personId={personId} />
      </>
    );
  }

  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">People</h2>
          <p className="text-sm text-muted">
            Who the agent talks to. Pick one in Who&apos;s talking before a call, or attribute recorded sessions and
            turns on the session page.
          </p>
        </div>
        <form
          className="rv-inline-edit"
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) create.mutate(name, { onSuccess: () => setName("") });
          }}
        >
          <input
            className="input"
            placeholder="New person's name"
            value={name}
            maxLength={128}
            onChange={(e) => setName(e.target.value)}
          />
          <button className="btn-primary" type="submit" disabled={!name.trim() || create.isPending}>
            Add
          </button>
        </form>
      </div>
      <StatusLine loading={people.isLoading} error={people.error ?? create.error} />
      <DuplicatesPanel />
      {people.data && people.data.people.length === 0 && <p className="text-sm text-muted">No people yet.</p>}
      {people.data && people.data.people.length > 0 && (
        <table className="rv-table rv-table-static">
          <thead>
            <tr>
              <th>Name</th>
              <th>Sessions</th>
              <th title="Voice and face samples bound at enrollment">Identity</th>
              <th title="Turns attributed live by voice ID">Voice ID turns</th>
              <th>Active memories</th>
              <th>Proposed</th>
              <th>Forgotten</th>
              <th>Added</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {people.data.people.map((p) => (
              <PersonRow key={p.id} person={p} />
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
