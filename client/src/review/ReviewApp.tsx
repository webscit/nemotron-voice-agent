// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "../api";
import { useActivities } from "./api";
import { AsrReview } from "./AsrReview";
import { DreamerStatusView } from "./Dreamer";
import { MetricsView } from "./Metrics";
import { SessionDetailView, SessionsList } from "./Sessions";
import { ANNOTATOR_PATTERN, reviewHref, reviewRoute, useAnnotator, useHash } from "./utils";
import "./review.scss";

function AnnotatorField() {
  const [annotator, setAnnotator] = useAnnotator();
  const [draft, setDraft] = useState(annotator);
  const valid = ANNOTATOR_PATTERN.test(draft.trim());
  return (
    <form
      className="rv-annotator"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid) setAnnotator(draft);
      }}
    >
      <label className="text-xs text-muted" htmlFor="rv-annotator">
        Annotator
      </label>
      <input
        id="rv-annotator"
        className="input"
        placeholder="your name"
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => valid && setAnnotator(draft)}
      />
    </form>
  );
}

function Nav({ view }: Readonly<{ view: string }>) {
  const activities = useActivities();
  const link = (id: string, label: string, badge?: number) => (
    <a key={id} href={reviewHref(id)} className={`rv-nav-link ${view === id ? "active" : ""}`}>
      {label}
      {badge ? <span className="rv-count">{badge}</span> : null}
    </a>
  );
  return (
    <nav className="rv-nav">
      <span className="rv-nav-section">Browse</span>
      {link("sessions", "Sessions")}
      {link("metrics", "Metrics")}
      <span className="rv-nav-section">Activities</span>
      {(activities.data?.activities ?? []).map((a) => link(a.id, a.title, a.open))}
      {activities.error && <span className="text-xs rv-error">review API unavailable</span>}
      <span className="rv-nav-section">Post-processing</span>
      {link("dreamer", "Dreamer")}
    </nav>
  );
}

function Content({ view, arg }: Readonly<{ view: string; arg: string }>) {
  switch (view) {
    case "session":
      return <SessionDetailView sessionId={arg} />;
    case "asr":
      return <AsrReview />;
    case "dreamer":
      return <DreamerStatusView />;
    case "metrics":
      return <MetricsView />;
    default:
      return <SessionsList />;
  }
}

export function ReviewApp() {
  const { view, arg } = reviewRoute(useHash());
  return (
    <QueryClientProvider client={queryClient}>
      <div className="h-screen d-flex flex-col overflow-hidden rv-root">
        <header className="px-4 py-3 border-b d-flex justify-between items-center rv-header">
          <h1 className="text-lg font-semibold">
            <span style={{ color: "#76b900", fontWeight: 700, letterSpacing: "0.08em" }}>Nemotron</span> Voice Agent ·
            Review
          </h1>
          <div className="d-flex items-center gap-3">
            <AnnotatorField />
            <a className="btn-ghost" href="#/">
              Live console
            </a>
          </div>
        </header>
        <div className="flex-1 d-flex overflow-hidden">
          <Nav view={view === "session" ? "sessions" : view} />
          <main className="flex-1 overflow-y-auto">
            <Content view={view} arg={arg} />
          </main>
        </div>
      </div>
    </QueryClientProvider>
  );
}
