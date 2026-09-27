// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

// ---- Types (mirror src/monitoring/api.py) ----

export interface SessionModels {
  asr?: string | null;
  llm?: string | null;
  tts?: string | null;
  voice?: string | null;
}

export interface SessionSummary {
  id: string;
  example: string;
  started_at: number;
  ended_at: number | null;
  end_reason: string | null;
  language: string | null;
  models: SessionModels;
  user_turns: number;
  audio_turns: number;
  reviewed_turns: number;
  wer: Record<string, number | null>;
  jobs: Record<string, string>;
}

export interface MediaRef {
  id: number;
  key: string;
  mime: string | null;
  duration_secs: number | null;
  source: string | null;
  ts: number;
  width?: number | null;
  height?: number | null;
}

export interface LlmCallBrief {
  id: number;
  started_at: number;
  model: string | null;
  ttfb: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  n_images: number | null;
  interrupted: boolean;
  output_text: string | null;
  function_calls: { name: string; tool_call_id: string; arguments: unknown }[] | null;
}

export interface WerValue {
  wer: number;
  cer: number;
  word_errors: number;
  ref_words: number;
  reference_source: string;
}

export interface HumanReference {
  source: string;
  text: string;
  at: number;
}

export interface TurnDetail {
  idx: number;
  user_text?: string | null;
  bot_text?: string | null;
  user_started_at?: number | null;
  bot_started_at?: number | null;
  interrupted?: boolean;
  language?: string | null;
  audio: { user: MediaRef[]; bot: MediaRef[] };
  images: MediaRef[];
  llm_calls: LlmCallBrief[];
  transcripts: Record<string, string>;
  human_reference: HumanReference | null;
  wer: Record<string, WerValue>;
  user_bot_latency: number | null;
}

export interface JobBrief {
  kind: string;
  status: string;
  error: string | null;
  attempts: number;
  finished_at?: number | null;
}

export interface SessionDetail {
  session: SessionSummary & { config: Record<string, unknown> };
  live_source: string;
  reference_source: string | null;
  conversation_audio: MediaRef | null;
  turns: TurnDetail[];
  wer_summary: Record<string, { wer: number | null; turns: number; reference_sources: string[] }>;
  jobs: JobBrief[];
}

export interface AsrQueueItem {
  session_id: string;
  turn_idx: number;
  started_at: number;
  language: string | null;
  audio: { key: string; duration_secs: number | null }[];
  live: { source: string; text: string };
  models: Record<string, string>;
  reference_source: string | null;
  human_reference: HumanReference | null;
  disagreement: number | null;
  agree: boolean;
}

export interface AsrQueue {
  total: number;
  open: number;
  agreeing: number;
  matching: number;
  items: AsrQueueItem[];
}

export type AsrQueueSort = "recent" | "disagreement";

export interface Activity {
  id: string;
  title: string;
  job: string;
  description: string;
  open: number;
}

export interface DreamerJob {
  id: number;
  kind: string;
  target: string;
  status: string;
  attempts: number;
  error: string | null;
  progress: Record<string, unknown> | null;
}

export interface DreamerWorker {
  state: "starting" | "idle" | "waiting" | "running" | "paused" | string;
  alive: boolean;
  heartbeat_at: number;
  host?: string;
  kind?: string;
  target?: string;
  job_id?: number;
  reason?: string;
  started_services?: string[];
}

export interface DreamerStatus {
  live_sessions: number;
  idle: boolean;
  last_activity: number | null;
  paused: boolean;
  paused_at: number | null;
  worker: DreamerWorker | null;
  services: { name: string; state: string }[];
  job_kinds: string[];
  counts: Record<string, Record<string, number>>;
  running: DreamerJob[];
  failed: DreamerJob[];
  pending: number;
}

export type JobStatus = "pending" | "running" | "done" | "failed" | "cancelled";

export interface JobRow extends DreamerJob {
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface Distribution {
  n: number;
  p10: number | null;
  p50: number | null;
  p90: number | null;
}

export interface MetricsVariant {
  key: Record<string, string>;
  label: string;
  sessions: number;
  turns: number;
  interrupt_rate: number | null;
  latency: Distribution;
  first_speech: Distribution;
  ttfb: Record<string, Distribution>;
  ttft: { text: Distribution; vision: Distribution };
  prompt_tokens_p50: number | null;
  wer: Record<string, { wer: number | null; ref_words: number }>;
}

export interface MetricsResponse {
  group_by: string[];
  totals: { sessions: number; turns: number; latency_p50: number | null; live_wer: number | null };
  variants: MetricsVariant[];
  sessions: { id: string; started_at: number; variant: number; latency_p50: number | null; live_wer: number | null }[];
}

export interface ReferenceItem {
  session_id: string;
  turn_idx: number;
  text: string;
}

// ---- Fetch helpers ----

const BASE = "/api/review";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}${body ? `: ${body.slice(0, 200)}` : ""}`);
  }
  return res.json();
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function artifactUrl(key: string): string {
  return `${BASE}/artifacts/${key.split("/").map(encodeURIComponent).join("/")}`;
}

// ---- Queries ----

const live = { staleTime: 0, refetchOnWindowFocus: true } as const;

export function useActivities() {
  return useQuery({
    queryKey: ["review", "activities"],
    queryFn: () => request<{ activities: Activity[] }>("/activities"),
    ...live,
  });
}

export function useSessions(limit = 100) {
  return useQuery({
    queryKey: ["review", "sessions", limit],
    queryFn: () => request<{ total: number; sessions: SessionSummary[] }>(`/sessions?limit=${limit}`),
    ...live,
  });
}

export function useSessionDetail(sessionId: string) {
  return useQuery({
    queryKey: ["review", "session", sessionId],
    queryFn: () => request<SessionDetail>(`/sessions/${encodeURIComponent(sessionId)}`),
    enabled: Boolean(sessionId),
    ...live,
  });
}

export function useLlmCall(sessionId: string, callId: number | null) {
  return useQuery({
    queryKey: ["review", "llm-call", sessionId, callId],
    queryFn: () =>
      request<{ messages: unknown; output_text: string; tools: string[] }>(
        `/sessions/${encodeURIComponent(sessionId)}/llm-calls/${callId}`
      ),
    enabled: callId !== null,
    staleTime: Infinity,
  });
}

export function useAsrQueue(status: "open" | "reviewed" | "all", sort: AsrQueueSort, limit = 200) {
  return useQuery({
    queryKey: ["review", "asr-queue", status, sort, limit],
    queryFn: () => request<AsrQueue>(`/asr/queue?status=${status}&sort=${sort}&limit=${limit}`),
    ...live,
  });
}

export function useDreamerStatus() {
  return useQuery({
    queryKey: ["review", "dreamer"],
    queryFn: () => request<DreamerStatus>("/dreamer"),
    refetchInterval: 5000,
    staleTime: 0,
  });
}

export function useJobs(status: JobStatus | "", page: number, pageSize = 25) {
  return useQuery({
    queryKey: ["review", "jobs", status, page, pageSize],
    queryFn: () =>
      request<{ total: number; jobs: JobRow[] }>(
        `/jobs?limit=${pageSize}&offset=${page * pageSize}${status ? `&status=${status}` : ""}`
      ),
    refetchInterval: 5000,
    staleTime: 0,
    placeholderData: (previous) => previous,
  });
}

export function useMetrics(sinceDays: number | null, groupBy: string[]) {
  const params = new URLSearchParams({ group_by: groupBy.join(",") });
  if (sinceDays) params.set("since_days", String(sinceDays));
  return useQuery({
    queryKey: ["review", "metrics", sinceDays, groupBy.join(",")],
    queryFn: () => request<MetricsResponse>(`/metrics?${params}`),
    staleTime: 0,
    // Keep the previous render while a new filter loads (no skeleton flash).
    placeholderData: (previous) => previous,
  });
}

// ---- Mutations ----

export function useSaveReferences() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: { annotator: string; items: ReferenceItem[] }) =>
      postJson<{ saved: number; wer_summary: Record<string, Record<string, unknown>> }>(
        "/asr/references",
        payload
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["review"] }),
  });
}

export function useRequeueJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: { kind: string; session_ids: string[] }) =>
      postJson<{ queued: number }>("/jobs", payload),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["review"] }),
  });
}

function useReviewMutation<TIn, TOut>(fn: (input: TIn) => Promise<TOut>) {
  const qc = useQueryClient();
  return useMutation({ mutationFn: fn, onSuccess: () => qc.invalidateQueries({ queryKey: ["review"] }) });
}

export function usePauseDreamer() {
  return useReviewMutation((paused: boolean) => postJson<{ paused: boolean }>("/dreamer/pause", { paused }));
}

export function useCancelJob() {
  return useReviewMutation((jobId: number) => postJson<{ cancelled: number }>(`/jobs/${jobId}/cancel`, {}));
}

export function useEnqueueAll() {
  return useReviewMutation((kind: string) => postJson<{ queued: number }>("/jobs/enqueue-all", { kind }));
}
