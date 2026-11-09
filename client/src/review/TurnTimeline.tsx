// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Per-turn waterfall of the session view: one row per latency stage, the bot speech,
// the decision model and each tool call on a shared time axis that runs from the end
// of user speech to the bot stopping, with the first perceivable response marked and
// the host load during the turn underneath.

import type { SystemSample, ToolCall, TurnDetail, TurnMetrics } from "./api";
import { INK, KIND_LABELS, SERIES, secondsTick, stageInfo } from "./chartTheme";
import { formatSecs, triggerLabel } from "./utils";

const OUTCOME_TONE: Record<ToolCall["outcome"], string> = {
  ok: "rv-badge-muted",
  error: "rv-badge-bad",
  timeout: "rv-badge-bad",
  cancelled: "rv-badge-warn",
};
const OUTCOME_FILL: Record<ToolCall["outcome"], string> = {
  ok: stageInfo("tool").color,
  error: "#f87171",
  timeout: "#f87171",
  cancelled: "#fbbf24",
};
const BOT_SPEECH_FILL = "rgba(255, 255, 255, 0.28)";
const TICK_STEPS = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 30, 60, 120];
// Load sparkline viewBox: x follows the axis in per-mille, y is 0-100 % load.
const LOAD_WIDTH = 1000;
const LOAD_HEIGHT = 28;

interface Bar {
  start: number;
  /** Null: the span has no recorded end (a tool call without result). */
  end: number | null;
  color: string;
  hollow?: boolean;
  title: string;
}

interface Row {
  key: string;
  label: string;
  detail: string;
  badge?: { text: string; tone: string };
  bars: Bar[];
  /** Shown in the track when the row has no bar to draw. */
  note?: string;
}

/** Decision model span: the recorded start, else estimated from its call (older rows), else unknown. */
function decisionSpan(turn: TurnDetail, metrics: TurnMetrics): { start: number; estimated: boolean } | null {
  const secs = metrics.decision_secs;
  if (metrics.decision_started_at != null) return { start: metrics.decision_started_at, estimated: false };
  if (secs === null) return null;
  // The decision call is sent as soon as the model answers.
  const sends = turn.tool_calls.filter((call) => call.trigger === "decision").map((call) => call.sent_at);
  if (sends.length) return { start: Math.min(...sends) - secs, estimated: true };
  if (turn.bot_speech_started_at != null && metrics.decision_lag_secs !== null) {
    return { start: turn.bot_speech_started_at + metrics.decision_lag_secs - secs, estimated: true };
  }
  return null;
}

const pct = (value: number) => `${value.toFixed(3)}%`;

function niceTicks(from: number, to: number): number[] {
  const step = TICK_STEPS.find((candidate) => (to - from) / candidate <= 5) ?? TICK_STEPS[TICK_STEPS.length - 1];
  const ticks: number[] = [];
  for (let value = Math.ceil(from / step - 1e-9) * step; value <= to + 1e-9; value += step) {
    ticks.push(Math.abs(value) < 1e-9 ? 0 : value);
  }
  return ticks;
}

/** Legend entry of the decision model for the turn's first answer: latency and when its call left. */
function DecisionLegend({ metrics }: Readonly<{ metrics: TurnMetrics }>) {
  if (!metrics.decision_outcome) return null;
  const lag = metrics.decision_lag_secs;
  return (
    <>
      <span className="rv-stage-legend-item" title="Runs alongside TTS, so it is not a latency stage">
        <span className="rv-swatch rv-swatch-hollow" />
        Decision model {formatSecs(metrics.decision_secs)}
      </span>
      {lag !== null && (
        <span className="rv-stage-legend-item">
          sent {formatSecs(Math.abs(lag))} {lag < 0 ? "before" : "after"} the first bot audio
        </span>
      )}
      {metrics.decision_outcome !== "called" && (
        <span className={`rv-badge ${metrics.decision_outcome === "error" ? "rv-badge-bad" : "rv-badge-muted"}`}>
          {metrics.decision_outcome}
        </span>
      )}
    </>
  );
}

/** GPU and CPU load (both percent, one axis), stretched to the waterfall axis; the band is the wait for the first response. */
function LoadTrack({
  samples,
  x,
  band,
}: Readonly<{ samples: SystemSample[]; x: (ts: number) => number; band: [number, number] | null }>) {
  const px = (ts: number) => (x(ts) / 100) * LOAD_WIDTH;
  const y = (value: number) => LOAD_HEIGHT - 1 - (Math.min(Math.max(value, 0), 100) / 100) * (LOAD_HEIGHT - 2);
  const line = (key: "gpu_load" | "cpu_load") =>
    samples
      .filter((sample) => sample[key] !== null)
      .map((sample) => `${px(sample.ts).toFixed(1)},${y(sample[key] as number).toFixed(1)}`)
      .join(" ");
  const gpu = line("gpu_load");
  const cpu = line("cpu_load");
  if (!gpu && !cpu) return null;
  return (
    <svg
      className="rv-wf-load"
      viewBox={`0 0 ${LOAD_WIDTH} ${LOAD_HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="GPU and CPU load during the turn, 0 to 100 percent"
    >
      <title>Host load during the turn, 0–100%. Shaded: waiting for the first response.</title>
      {band && (
        <rect
          x={px(band[0])}
          y={0}
          width={Math.max(px(band[1]) - px(band[0]), 1)}
          height={LOAD_HEIGHT}
          fill="rgba(255,255,255,0.08)"
        />
      )}
      <line
        x1={0}
        y1={LOAD_HEIGHT - 0.5}
        x2={LOAD_WIDTH}
        y2={LOAD_HEIGHT - 0.5}
        stroke={INK.axis}
        vectorEffect="non-scaling-stroke"
      />
      {cpu && (
        <polyline
          points={cpu}
          fill="none"
          stroke={SERIES[1]}
          strokeWidth={2}
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      )}
      {gpu && (
        <polyline
          points={gpu}
          fill="none"
          stroke={SERIES[0]}
          strokeWidth={2}
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}

/** Tool calls as badges, for turns that cannot be placed on a time axis. */
function ToolBadges({ calls }: Readonly<{ calls: ToolCall[] }>) {
  if (!calls.length) return null;
  return (
    <div className="rv-stage-legend">
      {calls.map((call) => (
        <span
          key={call.call_id}
          className={`rv-badge ${OUTCOME_TONE[call.outcome]}`}
          title={`${triggerLabel(call.trigger)} → ${call.target}${call.perceivable ? " · perceivable" : ""}`}
        >
          {call.name} · {formatSecs(call.duration_secs)} · {call.outcome}
        </span>
      ))}
    </div>
  );
}

export function TurnTimeline({ turn, samples }: Readonly<{ turn: TurnDetail; samples: SystemSample[] }>) {
  const metrics = turn.metrics;
  if (!metrics) return null;
  const segments = metrics.segments ?? [];
  // Time zero: the user stopped speaking (``derive`` patches this column, the first stage starts there).
  const origin = turn.user_speech_stopped_at ?? (segments.length ? segments[0].start : null);
  const at = (secs: number | null) => (origin !== null && secs !== null ? origin + secs : null);
  const responseAt = at(metrics.response_latency);
  const criticalEnd = at(metrics.total_secs);
  const audioAt = turn.bot_speech_started_at ?? at(metrics.voice_latency);
  const decision = decisionSpan(turn, metrics);
  const rel = (ts: number) => (origin === null ? "" : `+${formatSecs(ts - origin)}`);

  // ---- rows
  const rows: Row[] = [];
  const stageKeys = [...new Set(segments.map((segment) => segment.stage))];
  for (const key of stageKeys) {
    const info = stageInfo(key);
    const own = segments.filter((segment) => segment.stage === key);
    const secs = own.reduce((sum, segment) => sum + segment.end - segment.start, 0);
    rows.push({
      key: `stage-${key}`,
      label: info.label,
      detail: formatSecs(secs),
      bars: own.map((segment) => ({
        start: segment.start,
        end: segment.end,
        color: info.color,
        title: `${info.label}: ${formatSecs(segment.end - segment.start)}, from ${rel(segment.start)}`,
      })),
    });
  }
  if (audioAt !== null && turn.bot_stopped_at != null && turn.bot_stopped_at > audioAt) {
    const secs = turn.bot_stopped_at - audioAt;
    rows.push({
      key: "bot-speech",
      label: "Bot speech",
      detail: formatSecs(secs),
      bars: [
        {
          start: audioAt,
          end: turn.bot_stopped_at,
          color: BOT_SPEECH_FILL,
          title: `Bot speech: first audio ${rel(audioAt)} → bot stopped ${rel(turn.bot_stopped_at)} (${formatSecs(secs)})${turn.interrupted ? ", interrupted" : ""}`,
        },
      ],
    });
  }
  if (metrics.decision_outcome || metrics.decision_secs !== null) {
    const what = [
      metrics.decision_outcome,
      metrics.decision_choice && `→ ${metrics.decision_choice}`,
      metrics.decision_confidence !== null && `${Math.round(metrics.decision_confidence * 100)}% confidence`,
    ]
      .filter(Boolean)
      .join(" ");
    rows.push({
      key: "decision",
      label: "Decision model",
      detail: formatSecs(metrics.decision_secs),
      bars:
        decision && metrics.decision_secs !== null
          ? [
              {
                start: decision.start,
                end: decision.start + metrics.decision_secs,
                color: INK.secondary,
                hollow: true,
                title:
                  `Decision model: ${formatSecs(metrics.decision_secs)}, from ${rel(decision.start)}` +
                  (what ? ` · ${what}` : "") +
                  (decision.estimated ? " · start estimated from its call (recomputed metrics record it)" : ""),
              },
            ]
          : [],
      note: decision ? undefined : "start not recorded",
    });
  }
  turn.tool_calls.forEach((call) => {
    const end = call.duration_secs === null ? null : call.sent_at + call.duration_secs;
    rows.push({
      key: `tool-${call.call_id}`,
      label: call.name,
      detail: formatSecs(call.duration_secs),
      badge: call.outcome === "ok" ? undefined : { text: call.outcome, tone: OUTCOME_TONE[call.outcome] },
      bars: [
        {
          start: call.sent_at,
          end,
          color: OUTCOME_FILL[call.outcome],
          title:
            `${call.name}: ${triggerLabel(call.trigger)} → ${call.target}, sent ${rel(call.sent_at)}, ` +
            `${end === null ? "no result" : formatSecs(call.duration_secs)}, ${call.outcome}` +
            (call.perceivable ? " · perceivable" : ""),
        },
      ],
    });
  });

  // ---- axis: user stop -> bot stop, widened to every span
  const points = [origin, responseAt, criticalEnd, audioAt, turn.bot_stopped_at ?? null];
  for (const row of rows) for (const bar of row.bars) points.push(bar.start, bar.end);
  const known = points.filter((value): value is number => value !== null && Number.isFinite(value));
  const axis =
    origin !== null && known.length && rows.some((row) => row.bars.length)
      ? { origin, start: Math.min(...known), end: Math.max(...known) }
      : null;
  const placeable = axis !== null;
  const span = axis ? Math.max(axis.end - axis.start, 1e-3) : 1;
  const x = (ts: number) => (axis ? ((ts - axis.start) / span) * 100 : 0);

  const turnSamples = axis ? samples.filter((sample) => sample.ts >= axis.start - 1 && sample.ts <= axis.end + 1) : [];
  const ticks = axis ? niceTicks(axis.start - axis.origin, axis.end - axis.origin) : [];
  const marker = axis && responseAt !== null ? x(responseAt) : null;
  const responseLabel = metrics.response_via === "tool" ? "first response (action)" : "first response";
  const unexplained = metrics.unexplained_secs ?? 0;

  const markerLine = marker !== null && <span className="rv-wf-marker" style={{ left: pct(marker) }} aria-hidden />;

  return (
    <div className="rv-timeline">
      <div className="rv-timeline-head">
        <span className="rv-badge rv-badge-muted">{KIND_LABELS[metrics.kind] ?? metrics.kind}</span>
        {metrics.response_latency !== null && (
          <span
            className="rv-badge rv-badge-good"
            title={
              metrics.response_via === "tool"
                ? "user stops → a perceivable tool call is sent"
                : "user stops → first perceivable response (the first bot audio)"
            }
          >
            response {formatSecs(metrics.response_latency)}
            {metrics.response_via === "tool" ? " (action)" : ""}
          </span>
        )}
        {metrics.voice_latency !== null && (
          <span className="rv-badge rv-badge-muted" title="user stops → first bot audio">
            voice {formatSecs(metrics.voice_latency)}
          </span>
        )}
        {metrics.barge_in && (
          <span className="rv-badge rv-badge-warn" title="The user started speaking while the bot was speaking">
            barge-in
          </span>
        )}
        <span className="text-xs text-muted">
          {metrics.n_llm_calls ?? 0} LLM call{metrics.n_llm_calls === 1 ? "" : "s"}
          {metrics.prompt_tokens !== null && ` · ${metrics.prompt_tokens}→${metrics.completion_tokens ?? "?"} tok`}
          {metrics.gpu_load_mean !== null &&
            ` · GPU ${Math.round(metrics.gpu_load_mean)}% mean, ${Math.round(metrics.gpu_load_peak ?? 0)}% peak`}
        </span>
      </div>
      {axis && (
        <div
          className="rv-wf"
          role="group"
          aria-label={`Turn waterfall: seconds after the user stopped speaking, up to ${formatSecs(axis.end - axis.origin)}${
            responseAt !== null ? `; ${responseLabel} at ${formatSecs(responseAt - axis.origin)}` : ""
          }`}
        >
          {rows.map((row) => (
            <div key={row.key} className="rv-wf-row">
              <span className="rv-wf-label" title={`${row.label} ${row.detail}`}>
                <span className="rv-wf-name">{row.label}</span>
                {row.badge && <span className={`rv-badge ${row.badge.tone}`}>{row.badge.text}</span>}
                <span className="rv-wf-secs">{row.detail}</span>
              </span>
              <span className="rv-wf-track">
                {markerLine}
                {row.bars.map((bar, i) => (
                  <span
                    key={i}
                    role="img"
                    aria-label={bar.title}
                    title={bar.title}
                    className={`rv-wf-bar${bar.hollow ? " rv-wf-bar-hollow" : ""}${bar.end === null ? " rv-wf-bar-open" : ""}`}
                    style={{
                      left: pct(x(bar.start)),
                      width: bar.end === null ? undefined : pct(x(bar.end) - x(bar.start)),
                      ...(bar.hollow ? { borderColor: bar.color } : { background: bar.color }),
                    }}
                  />
                ))}
                {row.note && <span className="rv-wf-note">{row.note}</span>}
              </span>
            </div>
          ))}
          {turnSamples.length > 0 && (
            <div className="rv-wf-row">
              <span className="rv-wf-label">
                <span className="rv-wf-name">Host load</span>
              </span>
              <span className="rv-wf-track rv-wf-track-load">
                {markerLine}
                <LoadTrack
                  samples={turnSamples}
                  x={x}
                  band={responseAt !== null ? [axis.origin, responseAt] : null}
                />
              </span>
            </div>
          )}
          <div className="rv-wf-row rv-wf-axis" aria-hidden>
            <span className="rv-wf-label" />
            <span className="rv-wf-track">
              {ticks.map((tick) => {
                const left = x(axis.origin + tick);
                // Keep the edge labels inside the track.
                const shift = left < 6 ? "none" : left > 94 ? "translateX(-100%)" : "translateX(-50%)";
                return (
                  <span key={tick} className="rv-wf-tick" style={{ left: pct(left), transform: shift }}>
                    {secondsTick(tick)}
                  </span>
                );
              })}
            </span>
          </div>
        </div>
      )}
      {(stageKeys.length > 0 || metrics.decision_outcome || turnSamples.length > 0) && (
        <div className="rv-stage-legend">
          {stageKeys.map((key) => {
            const info = stageInfo(key);
            const secs = segments
              .filter((segment) => segment.stage === key)
              .reduce((sum, segment) => sum + segment.end - segment.start, 0);
            return (
              <span key={key} className="rv-stage-legend-item">
                <span className="rv-swatch" style={{ background: info.color }} />
                {info.label} {formatSecs(secs)}
              </span>
            );
          })}
          {unexplained > 0.0005 && (
            <span
              className="rv-stage-legend-item"
              title="Time up to the first response that no stage owns: the gaps between stage bars"
            >
              <span className="rv-swatch" style={{ background: stageInfo("unexplained").color }} />
              {stageInfo("unexplained").label} {formatSecs(unexplained)}
            </span>
          )}
          {placeable && marker !== null && (
            <span className="rv-stage-legend-item">
              <span className="rv-swatch rv-swatch-marker" />
              {responseLabel}
            </span>
          )}
          {turnSamples.length > 0 && (
            <>
              <span className="rv-stage-legend-item">
                <span className="rv-swatch" style={{ background: SERIES[0] }} />
                GPU load
              </span>
              <span className="rv-stage-legend-item">
                <span className="rv-swatch" style={{ background: SERIES[1] }} />
                CPU load
              </span>
            </>
          )}
          <DecisionLegend metrics={metrics} />
        </div>
      )}
      {!placeable && <ToolBadges calls={turn.tool_calls} />}
    </div>
  );
}
