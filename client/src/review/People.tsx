// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { useCreatePerson, usePeople, useUpdatePerson, type Person } from "./api";
import { StatusLine } from "./components";
import { MemoriesView } from "./Memories";
import { formatDate, reviewHref } from "./utils";

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
      {people.data && people.data.people.length === 0 && <p className="text-sm text-muted">No people yet.</p>}
      {people.data && people.data.people.length > 0 && (
        <table className="rv-table rv-table-static">
          <thead>
            <tr>
              <th>Name</th>
              <th>Sessions</th>
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
