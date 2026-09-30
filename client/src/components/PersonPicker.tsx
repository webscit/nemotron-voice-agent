// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { PersonOption } from "../hooks/useLivePerson";

export function PersonPicker({
  people,
  value,
  onChange,
  disabled,
}: Readonly<{ people: PersonOption[]; value: string; onChange: (id: string) => void; disabled: boolean }>) {
  return (
    <label className="d-flex items-center gap-2 text-xs text-muted" title="Used for memories about this person">
      Who&apos;s talking
      <select className="input" value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        <option value="">Unknown</option>
        {people.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>
    </label>
  );
}
