// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useDreamerStatus, useRequeueJob } from "./api";
import { StatusLine } from "./components";
import { formatDate, reviewHref } from "./utils";

export function DreamerStatusView() {
  const status = useDreamerStatus();
  const requeue = useRequeueJob();
  const data = status.data;
  let state = "–";
  if (data) {
    if (data.live_sessions) state = `busy · ${data.live_sessions} live session(s)`;
    else state = data.idle ? "idle · jobs may run" : "cooling down after the last session";
  }
  return (
    <section className="rv-page">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Dreamer</h2>
          <p className="text-sm text-muted">
            Post-processing runs only while nobody is talking to the assistant. Last activity:{" "}
            {formatDate(data?.last_activity)}.
          </p>
        </div>
        <span className={`rv-badge ${data?.live_sessions ? "rv-badge-warn" : "rv-badge-good"}`}>{state}</span>
      </div>
      <StatusLine loading={status.isLoading} error={status.error} />
      {data && Object.keys(data.counts).length === 0 && (
        <p className="text-sm text-muted">
          No jobs yet. Jobs are queued automatically when a recorded session ends (run the dreamer with{" "}
          <code>--profile dreamer</code>).
        </p>
      )}
      {data && Object.keys(data.counts).length > 0 && (
        <>
          <table className="rv-table">
            <thead>
              <tr>
                <th>Job</th>
                <th>Pending</th>
                <th>Running</th>
                <th>Done</th>
                <th>Failed</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.counts).map(([kind, counts]) => (
                <tr key={kind}>
                  <td>{kind}</td>
                  <td>{counts.pending ?? 0}</td>
                  <td>{counts.running ?? 0}</td>
                  <td>{counts.done ?? 0}</td>
                  <td>{counts.failed ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.running.map((job) => (
            <p key={job.id} className="text-sm">
              Running {job.kind} on <a href={reviewHref("session", job.target)}>{job.target}</a>
              {job.progress && Object.keys(job.progress).length > 0 && (
                <span className="text-muted"> · progress {JSON.stringify(job.progress).slice(0, 80)}</span>
              )}
            </p>
          ))}
          {data.failed.length > 0 && (
            <>
              <h3 className="text-sm font-semibold">Failures</h3>
              {data.failed.map((job) => (
                <div key={job.id} className="card rv-card">
                  <div className="rv-card-head">
                    <span className="text-sm">
                      {job.kind} · <a href={reviewHref("session", job.target)}>{job.target}</a> · {job.status} after{" "}
                      {job.attempts} attempt(s)
                    </span>
                    <button
                      className="btn-secondary"
                      disabled={requeue.isPending}
                      onClick={() => requeue.mutate({ kind: job.kind, session_ids: [job.target] })}
                    >
                      Retry
                    </button>
                  </div>
                  <pre className="rv-pre">{job.error}</pre>
                </div>
              ))}
            </>
          )}
        </>
      )}
    </section>
  );
}
