// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import {
  useCancelJob,
  useDreamerStatus,
  useEnqueueAll,
  useJobs,
  usePauseDreamer,
  useRequeueJob,
  type DreamerStatus,
  type JobRow,
  type JobStatus,
} from "./api";
import { StatusLine } from "./components";
import { formatDate, reviewHref } from "./utils";

const PAGE_SIZE = 25;
const STATUS_FILTERS: (JobStatus | "")[] = ["", "pending", "running", "failed", "done", "cancelled"];

const JOB_TONE: Record<string, string> = {
  pending: "rv-badge-muted",
  running: "rv-badge-warn",
  done: "rv-badge-good",
  failed: "rv-badge-bad",
  cancelled: "rv-badge-muted",
};

const SERVICE_TONE: Record<string, string> = {
  healthy: "rv-badge-good",
  running: "rv-badge-good",
  starting: "rv-badge-warn",
  unhealthy: "rv-badge-bad",
};

function secondsAgo(epoch: number | null | undefined): string {
  if (!epoch) return "never";
  const secs = Math.max(0, Math.round(Date.now() / 1000 - epoch));
  if (secs < 90) return `${secs} s ago`;
  if (secs < 5400) return `${Math.round(secs / 60)} min ago`;
  return formatDate(epoch);
}

/** One-line summary of what the dreamer is doing and why. */
function WorkerLine({ status }: Readonly<{ status: DreamerStatus }>) {
  const worker = status.worker;
  if (!worker) {
    return (
      <p className="text-sm text-muted">
        No dreamer has reported yet. Start it with <code>docker compose --profile dreamer up -d</code>.
      </p>
    );
  }
  if (!worker.alive) {
    return (
      <p className="text-sm rv-error">
        Dreamer not responding (last heartbeat {secondsAgo(worker.heartbeat_at)}). Jobs stay queued until it is back.
      </p>
    );
  }
  let doing = "Idle: waiting for jobs.";
  if (worker.state === "running") {
    doing = `Running ${worker.kind} on ${worker.target}.`;
  } else if (worker.state === "paused") {
    doing = "Paused: no job starts, a running job stops at its next checkpoint.";
  } else if (worker.state === "waiting") {
    doing =
      status.live_sessions > 0
        ? `Waiting: ${status.live_sessions} live session(s).`
        : "Waiting for the idle grace period after the last session.";
  }
  return (
    <p className="text-sm">
      {doing}{" "}
      <span className="text-muted">
        · {worker.host} · heartbeat {secondsAgo(worker.heartbeat_at)}
      </span>
    </p>
  );
}

function StatTile({ label, value, tone }: Readonly<{ label: string; value: number; tone?: string }>) {
  return (
    <div className="card rv-stat">
      <span className="rv-label">{label}</span>
      <span className={`rv-stat-value ${tone ?? ""}`}>{value}</span>
    </div>
  );
}

function JobActions({ job }: Readonly<{ job: JobRow }>) {
  const cancel = useCancelJob();
  const requeue = useRequeueJob();
  if (job.status === "pending") {
    return (
      <button className="btn-ghost text-xs" disabled={cancel.isPending} onClick={() => cancel.mutate(job.id)}>
        Cancel
      </button>
    );
  }
  if (job.status === "running") return null;
  return (
    <button
      className="btn-ghost text-xs"
      disabled={requeue.isPending}
      onClick={() => requeue.mutate({ kind: job.kind, session_ids: [job.target] })}
      title="Queue again from scratch"
    >
      {job.status === "done" ? "Re-run" : "Retry"}
    </button>
  );
}

function progressText(job: JobRow): string {
  if (job.error) return job.error;
  const done = (job.progress as { done?: unknown[] } | null)?.done;
  return Array.isArray(done) && done.length ? `${done.length} step(s) done` : "";
}

function JobQueue() {
  const [status, setStatus] = useState<JobStatus | "">("");
  const [page, setPage] = useState(0);
  const jobs = useJobs(status, page, PAGE_SIZE);
  const total = jobs.data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  return (
    <div className="rv-section">
      <div className="rv-page-head">
        <h3 className="text-sm font-semibold">Queue</h3>
        <div className="rv-segmented">
          {STATUS_FILTERS.map((s) => (
            <button
              key={s || "all"}
              className={`tab-btn ${status === s ? "active" : ""}`}
              onClick={() => {
                setStatus(s);
                setPage(0);
              }}
            >
              {s || "all"}
            </button>
          ))}
        </div>
      </div>
      <StatusLine loading={jobs.isLoading} error={jobs.error} />
      {jobs.data && jobs.data.jobs.length === 0 && <p className="text-sm text-muted">No jobs.</p>}
      {jobs.data && jobs.data.jobs.length > 0 && (
        <table className="rv-table rv-table-static">
          <thead>
            <tr>
              <th>#</th>
              <th>Job</th>
              <th>Session</th>
              <th>Status</th>
              <th>Attempts</th>
              <th>Queued</th>
              <th>Details</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {jobs.data.jobs.map((job) => (
              <tr key={job.id}>
                <td className="text-muted">{job.id}</td>
                <td>{job.kind}</td>
                <td className="rv-mono text-xs">
                  <a href={reviewHref("session", job.target)}>{job.target}</a>
                </td>
                <td>
                  <span className={`rv-badge ${JOB_TONE[job.status] ?? "rv-badge-muted"}`}>{job.status}</span>
                </td>
                <td>{job.attempts}</td>
                <td className="rv-nowrap">{formatDate(job.created_at)}</td>
                <td className={`rv-ellipsis text-xs ${job.error ? "rv-error" : "text-muted"}`} title={job.error ?? ""}>
                  {progressText(job)}
                </td>
                <td>
                  <JobActions job={job} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
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
    </div>
  );
}

export function DreamerStatusView() {
  const status = useDreamerStatus();
  const pause = usePauseDreamer();
  const enqueueAll = useEnqueueAll();
  const data = status.data;
  const totals = { pending: 0, running: 0, done: 0, failed: 0 };
  for (const counts of Object.values(data?.counts ?? {})) {
    for (const key of Object.keys(totals) as (keyof typeof totals)[]) totals[key] += counts[key] ?? 0;
  }
  let stateLabel = "";
  let stateTone = "rv-badge-muted";
  if (data?.paused) {
    stateLabel = "paused";
    stateTone = "rv-badge-warn";
  } else if (data && data.live_sessions > 0) {
    stateLabel = `${data.live_sessions} live session(s)`;
    stateTone = "rv-badge-warn";
  } else if (data) {
    stateLabel = data.idle ? "idle: jobs may run" : "cooling down";
    stateTone = "rv-badge-good";
  }

  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Dreamer</h2>
          <p className="text-sm text-muted">
            Post-processing runs only while nobody is talking to the assistant. Last session activity:{" "}
            {secondsAgo(data?.last_activity)}.
          </p>
        </div>
        <div className="rv-actions">
          {stateLabel && <span className={`rv-badge ${stateTone}`}>{stateLabel}</span>}
          {data && (
            <button
              className={data.paused ? "btn-primary" : "btn-secondary"}
              disabled={pause.isPending}
              onClick={() => pause.mutate(!data.paused)}
              title={data.paused ? "Let jobs run again when idle" : "Stop starting jobs; a running job stops at its next step"}
            >
              {data.paused ? "Resume" : "Pause"}
            </button>
          )}
        </div>
      </div>
      <StatusLine loading={status.isLoading} error={status.error} />
      {data && (
        <>
          <WorkerLine status={data} />
          <div className="rv-stats-grid">
            <StatTile label="Pending" value={totals.pending} />
            <StatTile label="Running" value={totals.running} />
            <StatTile label="Done" value={totals.done} />
            <StatTile label="Failed" value={totals.failed} tone={totals.failed ? "rv-error" : ""} />
          </div>

          <div className="rv-section">
            <h3 className="text-sm font-semibold">Jobs</h3>
            <table className="rv-table rv-table-static">
              <thead>
                <tr>
                  <th>Kind</th>
                  <th>Pending</th>
                  <th>Running</th>
                  <th>Done</th>
                  <th>Failed</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {data.job_kinds.map((kind) => {
                  const counts = data.counts[kind] ?? {};
                  return (
                    <tr key={kind}>
                      <td>{kind}</td>
                      <td>{counts.pending ?? 0}</td>
                      <td>{counts.running ?? 0}</td>
                      <td>{counts.done ?? 0}</td>
                      <td>{counts.failed ?? 0}</td>
                      <td>
                        <button
                          className="btn-ghost text-xs"
                          disabled={enqueueAll.isPending}
                          onClick={() => enqueueAll.mutate(kind)}
                          title="Queue for every ended session that has never had this job"
                        >
                          Queue for all sessions
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {enqueueAll.data && (
              <p className="text-xs text-muted">Queued {enqueueAll.data.queued} new job(s).</p>
            )}
          </div>

          {data.services.length > 0 && (
            <div className="rv-section">
              <h3 className="text-sm font-semibold">On-demand models</h3>
              <p className="text-xs text-muted">
                Started by the dreamer only while idle and stopped before a conversation needs the GPU.
              </p>
              <div className="rv-card-head">
                {data.services.map((svc) => (
                  <span key={svc.name} className={`rv-badge ${SERVICE_TONE[svc.state] ?? "rv-badge-muted"}`}>
                    {svc.name}: {svc.state}
                  </span>
                ))}
              </div>
            </div>
          )}

          <JobQueue />
        </>
      )}
    </section>
  );
}
