// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useSyncExternalStore } from "react";
import { useQuery } from "@tanstack/react-query";

// "Who's talking": the person the next session is attributed to, so the agent
// can use what it remembers about them. People are managed in the review UI
// (#/review/people); the list endpoint only exists when monitoring is enabled.

const PERSON_KEY = "live.person_id";
const listeners = new Set<() => void>();

export type PersonOption = { id: string; name: string };

function readPerson(): string {
  try {
    return globalThis.localStorage?.getItem(PERSON_KEY) ?? "";
  } catch {
    return "";
  }
}

export function useLivePerson(): [string, (id: string) => void] {
  const personId = useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    readPerson
  );
  const setPersonId = useCallback((id: string) => {
    try {
      globalThis.localStorage?.setItem(PERSON_KEY, id);
    } catch {
      // Storage unavailable (private mode): the choice lasts for this page only.
    }
    listeners.forEach((cb) => cb());
  }, []);
  return [personId, setPersonId];
}

export function usePeopleOptions(enabled: boolean) {
  return useQuery({
    queryKey: ["live", "people"],
    queryFn: async () => {
      const res = await fetch("/api/review/people");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return ((await res.json()) as { people: PersonOption[] }).people;
    },
    enabled,
    retry: false,
    staleTime: 30_000,
    refetchOnWindowFocus: true,
  });
}
