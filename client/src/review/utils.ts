// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useCallback, useSyncExternalStore } from "react";

// ---- Hash routing: #/review/<view>/<arg> ----

function subscribeHash(onChange: () => void) {
  globalThis.addEventListener("hashchange", onChange);
  return () => globalThis.removeEventListener("hashchange", onChange);
}

export function useHash(): string {
  return useSyncExternalStore(subscribeHash, () => globalThis.location.hash);
}

export function isReviewHash(hash: string): boolean {
  return hash === "#/review" || hash.startsWith("#/review/");
}

export function reviewRoute(hash: string): { view: string; arg: string } {
  const [, , view = "sessions", ...rest] = hash.replace(/^#/, "").split("/");
  return { view, arg: decodeURIComponent(rest.join("/")) };
}

export function reviewHref(view: string, arg = ""): string {
  return `#/review/${view}${arg ? `/${encodeURIComponent(arg)}` : ""}`;
}

// ---- Annotator identity (no auth: remembered per browser) ----

const ANNOTATOR_KEY = "review.annotator";
const annotatorListeners = new Set<() => void>();

function readAnnotator(): string {
  try {
    return globalThis.localStorage?.getItem(ANNOTATOR_KEY) ?? "";
  } catch {
    return "";
  }
}

export function useAnnotator(): [string, (name: string) => void] {
  const name = useSyncExternalStore(
    (cb) => {
      annotatorListeners.add(cb);
      return () => annotatorListeners.delete(cb);
    },
    readAnnotator
  );
  const setName = useCallback((value: string) => {
    try {
      globalThis.localStorage?.setItem(ANNOTATOR_KEY, value.trim());
    } catch {
      // Storage unavailable (private mode): the name just won't persist.
    }
    annotatorListeners.forEach((cb) => cb());
  }, []);
  return [name, setName];
}

export const ANNOTATOR_PATTERN = /^[\w .@-]{1,64}$/u;

// ---- Formatting ----

export function formatDate(epochSecs: number | null | undefined): string {
  if (!epochSecs) return "–";
  return new Date(epochSecs * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDuration(secs: number | null | undefined): string {
  if (secs === null || secs === undefined) return "–";
  if (secs < 60) return `${secs.toFixed(1)} s`;
  return `${Math.floor(secs / 60)} min ${Math.round(secs % 60)} s`;
}

export function formatPct(value: number | null | undefined): string {
  return value === null || value === undefined ? "–" : `${(value * 100).toFixed(1)}%`;
}

export function formatSecs(value: number | null | undefined): string {
  return value === null || value === undefined ? "–" : `${value.toFixed(2)} s`;
}

// ---- Word diff (LCS over normalized tokens) ----
// Tokenization mirrors the server's WER normalization (src/monitoring/jobs/wer.py):
// hyphens and punctuation separate words, so "Rappelle-moi" matches "rappelle moi".

export type DiffToken = { text: string; sep: string; kind: "same" | "del" | "ins" };

type Token = { text: string; sep: string };

function tokens(text: string): Token[] {
  const out: Token[] = [];
  for (const word of text.split(/\s+/u).filter(Boolean)) {
    const parts = word.split(/(?<=[-‐‑–—])/u);
    parts.forEach((part, i) => out.push({ text: part, sep: i < parts.length - 1 ? "" : " " }));
  }
  return out;
}

function norm(token: string): string {
  return token.toLocaleLowerCase().replaceAll(/[^\p{L}\p{N}']/gu, "");
}

/** Diff ``hypothesis`` against ``reference``: ``del`` = missing from hypothesis, ``ins`` = extra. */
export function wordDiff(reference: string, hypothesis: string): DiffToken[] {
  const a = tokens(reference);
  const b = tokens(hypothesis);
  const lcs: number[][] = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] =
        norm(a[i].text) === norm(b[j].text) ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const out: DiffToken[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (norm(a[i].text) === norm(b[j].text)) {
      out.push({ ...b[j], kind: "same" });
      i++;
      j++;
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      out.push({ ...a[i++], kind: "del" });
    } else {
      out.push({ ...b[j++], kind: "ins" });
    }
  }
  while (i < a.length) out.push({ ...a[i++], kind: "del" });
  while (j < b.length) out.push({ ...b[j++], kind: "ins" });
  return out;
}
