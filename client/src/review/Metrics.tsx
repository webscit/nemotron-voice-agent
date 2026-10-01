// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { ScatterPointItem } from "recharts/types/cartesian/Scatter";
import { useMetrics, type Distribution, type MetricsResponse, type MetricsVariant } from "./api";
import { StatusLine } from "./components";
import { formatDate, formatPct, formatSecs, reviewHref } from "./utils";

// Dark-mode categorical steps of the dataviz reference palette, validated on the
// card surface #1a1a1a (all checks pass). Slot order is the CVD-safety mechanism.
const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const OVERFLOW = "#898781"; // past 8 variants: muted, never a generated hue
const INK = { primary: "#ffffff", secondary: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835" };

// Color follows the variant, not its rank: a label keeps its slot for the whole
// page lifetime even when a filter removes or reorders variants.
const slotByLabel = new Map<string, number>();
function variantColor(label: string): string {
  if (!slotByLabel.has(label)) slotByLabel.set(label, slotByLabel.size);
  const slot = slotByLabel.get(label) ?? SERIES.length;
  return SERIES[slot] ?? OVERFLOW;
}

const RANGES: { label: string; days: number | null }[] = [
  { label: "24 h", days: 1 },
  { label: "7 days", days: 7 },
  { label: "30 days", days: 30 },
  { label: "All", days: null },
];

const GROUP_FIELDS: { field: string; label: string }[] = [
  { field: "llm.model", label: "LLM" },
  { field: "asr.model", label: "ASR" },
  { field: "tts.model", label: "TTS" },
  { field: "tts.voice", label: "Voice" },
  { field: "language", label: "Language" },
  { field: "turn_detection.silero_vad_only", label: "VAD-only turns" },
  { field: "transport", label: "Transport" },
];
const DEFAULT_GROUP = ["llm.model", "asr.model", "tts.model", "language"];

const tooltipStyle = {
  background: "#252525",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 6,
  color: INK.primary,
  fontSize: 12,
};
const axisTick = { fill: INK.muted, fontSize: 11 };

/** Seconds axis ticks: milliseconds below 1 s so sub-second ticks don't round together. */
function secondsTick(value: number): string {
  if (value === 0) return "0";
  return Math.abs(value) < 1 ? `${Math.round(value * 1000)} ms` : `${Number(value.toFixed(2))} s`;
}

function short(index: number): string {
  return `V${index + 1}`;
}

function ChartCard({
  title,
  subtitle,
  children,
}: Readonly<{ title: string; subtitle?: string; children: ReactNode }>) {
  return (
    <figure className="card rv-chart">
      <figcaption>
        <span className="text-sm font-semibold">{title}</span>
        {subtitle && <span className="text-xs text-muted"> {subtitle}</span>}
      </figcaption>
      {children}
    </figure>
  );
}

function Empty({ what }: Readonly<{ what: string }>) {
  return <p className="text-xs text-muted rv-chart-empty">No {what} recorded in this range.</p>;
}

/** Tooltip body shared by the grouped bar charts. */
function VariantTooltip({
  active,
  payload,
  variants,
  format,
}: Readonly<{
  active?: boolean;
  payload?: { dataKey?: string | number; value?: number; payload?: Record<string, unknown> }[];
  variants: MetricsVariant[];
  format: (value: number) => string;
}>) {
  if (!active || !payload?.length) return null;
  const category = String(payload[0].payload?.category ?? "");
  return (
    <div style={tooltipStyle} className="rv-tooltip">
      <strong>{category}</strong>
      {payload.map((entry) => {
        const index = Number(String(entry.dataKey).slice(1)) - 1;
        const variant = variants[index];
        if (!variant || entry.value === undefined || entry.value === null) return null;
        const n = entry.payload?.[`${String(entry.dataKey)}_n`];
        return (
          <div key={String(entry.dataKey)} className="rv-tooltip-row">
            <span className="rv-swatch" style={{ background: variantColor(variant.label) }} />
            <span>{short(index)}</span>
            <span className="rv-tooltip-value">{format(entry.value)}</span>
            {typeof n === "number" && <span className="text-muted">n={n}</span>}
          </div>
        );
      })}
    </div>
  );
}

/** Grouped bars: one category per x tick, one bar per variant (color = variant). */
function GroupedBars({
  variants,
  categories,
  value,
  format,
  unit,
}: Readonly<{
  variants: MetricsVariant[];
  categories: string[];
  value: (variant: MetricsVariant, category: string) => { value: number | null; n?: number };
  format: (value: number) => string;
  unit: "s" | "%";
}>) {
  const rows = categories.map((category) => {
    const row: Record<string, number | string | null> = { category };
    variants.forEach((variant, i) => {
      const cell = value(variant, category);
      row[short(i)] = cell.value;
      if (cell.n !== undefined) row[`${short(i)}_n`] = cell.n;
    });
    return row;
  });
  const barSize = Math.max(6, Math.min(18, Math.floor(160 / Math.max(1, variants.length))));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} barGap={2} barCategoryGap="24%">
        <CartesianGrid stroke={INK.grid} vertical={false} />
        <XAxis dataKey="category" tick={axisTick} tickLine={false} axisLine={{ stroke: INK.axis }} />
        <YAxis
          tick={axisTick}
          tickLine={false}
          axisLine={false}
          width={56}
          tickFormatter={(v: number) => (unit === "%" ? `${Math.round(v * 100)}%` : secondsTick(v))}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          content={<VariantTooltip variants={variants} format={format} />}
        />
        {variants.map((variant, i) => (
          <Bar
            key={variant.label}
            dataKey={short(i)}
            fill={variantColor(variant.label)}
            barSize={barSize}
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Median bar with a p10–p90 whisker per variant (horizontal). */
function LatencyRange({ variants }: Readonly<{ variants: MetricsVariant[] }>) {
  const rows = variants
    .map((variant, i) => ({ variant, i }))
    .filter(({ variant }) => variant.latency.p50 !== null)
    .map(({ variant, i }) => {
      const d = variant.latency;
      return {
        name: short(i),
        label: variant.label,
        p50: d.p50,
        range: [(d.p50 ?? 0) - (d.p10 ?? d.p50 ?? 0), (d.p90 ?? d.p50 ?? 0) - (d.p50 ?? 0)],
        dist: d,
      };
    });
  if (!rows.length) return <Empty what="user→bot latency" />;
  return (
    <ResponsiveContainer width="100%" height={Math.max(240, rows.length * 34 + 40)}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={INK.grid} horizontal={false} />
        <XAxis
          type="number"
          tick={axisTick}
          tickLine={false}
          axisLine={{ stroke: INK.axis }}
          tickFormatter={secondsTick}
        />
        <YAxis type="category" dataKey="name" tick={axisTick} tickLine={false} axisLine={false} width={36} />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          content={({ active, payload }) => {
            const row = payload?.[0]?.payload as (typeof rows)[number] | undefined;
            if (!active || !row) return null;
            return (
              <div style={tooltipStyle} className="rv-tooltip">
                <strong>
                  {row.name} · {row.label}
                </strong>
                <div>
                  median {formatSecs(row.dist.p50)} · p10 {formatSecs(row.dist.p10)} · p90 {formatSecs(row.dist.p90)}
                </div>
                <div className="text-muted">{row.dist.n} turns</div>
              </div>
            );
          }}
        />
        <Bar dataKey="p50" barSize={14} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {rows.map((row) => (
            <Cell key={row.name} fill={variantColor(row.label)} />
          ))}
          <ErrorBar dataKey="range" direction="x" width={6} stroke={INK.secondary} strokeWidth={1.5} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Per-session median latency over time; one variant emphasized, the rest muted. */
function LatencyTrend({ data, highlight }: Readonly<{ data: MetricsResponse; highlight: number }>) {
  const points = data.sessions
    .filter((s) => s.latency_p50 !== null)
    .map((s) => ({ x: s.started_at, y: s.latency_p50, id: s.id, variant: s.variant }));
  if (points.length < 2) return <Empty what="sessions with latency" />;
  const focus = points.filter((p) => p.variant === highlight);
  const rest = points.filter((p) => p.variant !== highlight);
  const color = data.variants[highlight] ? variantColor(data.variants[highlight].label) : SERIES[0];
  const open = (item: ScatterPointItem) => {
    const sessionId = (item.payload as { id?: string } | undefined)?.id;
    if (sessionId) globalThis.location.hash = reviewHref("session", sessionId);
  };
  return (
    <ResponsiveContainer width="100%" height={240}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={INK.grid} />
        <XAxis
          type="number"
          dataKey="x"
          domain={["dataMin", "dataMax"]}
          tick={axisTick}
          tickLine={false}
          axisLine={{ stroke: INK.axis }}
          tickFormatter={(v: number) => formatDate(v)}
          scale="time"
        />
        <YAxis
          type="number"
          dataKey="y"
          tick={axisTick}
          tickLine={false}
          axisLine={false}
          width={56}
          tickFormatter={secondsTick}
        />
        <ZAxis range={[64, 64]} />
        <Tooltip
          cursor={{ stroke: INK.axis }}
          content={({ active, payload }) => {
            const p = payload?.[0]?.payload as (typeof points)[number] | undefined;
            if (!active || !p) return null;
            return (
              <div style={tooltipStyle} className="rv-tooltip">
                <strong>{p.id}</strong>
                <div>
                  {short(p.variant)} · median user→bot {formatSecs(p.y)}
                </div>
                <div className="text-muted">{formatDate(p.x)} · click to open</div>
              </div>
            );
          }}
        />
        <Scatter data={rest} fill="#5a5955" isAnimationActive={false} onClick={open} cursor="pointer" />
        <Scatter
          data={focus}
          fill={color}
          stroke="#1a1a1a"
          strokeWidth={2}
          isAnimationActive={false}
          onClick={open}
          cursor="pointer"
        />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function Legend({
  variants,
  highlight,
  onHighlight,
}: Readonly<{ variants: MetricsVariant[]; highlight: number; onHighlight: (i: number) => void }>) {
  return (
    <div className="rv-legend" role="list">
      {variants.map((variant, i) => (
        <button
          key={variant.label}
          role="listitem"
          className={`rv-legend-item ${i === highlight ? "active" : ""}`}
          onClick={() => onHighlight(i)}
          title="Highlight in the trend chart"
        >
          <span className="rv-swatch" style={{ background: variantColor(variant.label) }} />
          <strong>{short(i)}</strong>
          <span className="rv-legend-label">{variant.label}</span>
          <span className="text-muted">
            {variant.sessions} session{variant.sessions === 1 ? "" : "s"}
          </span>
        </button>
      ))}
    </div>
  );
}

function Tile({ label, value, hint }: Readonly<{ label: string; value: string; hint?: string }>) {
  return (
    <div className="card rv-stat">
      <span className="rv-label">{label}</span>
      <span className="rv-stat-value">{value}</span>
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </div>
  );
}

function dist(d: Distribution | undefined): { value: number | null; n: number } {
  return { value: d?.p50 ?? null, n: d?.n ?? 0 };
}

function SummaryTable({ variants }: Readonly<{ variants: MetricsVariant[] }>) {
  const services = [...new Set(variants.flatMap((v) => Object.keys(v.ttfb)))].sort();
  const sources = [...new Set(variants.flatMap((v) => Object.keys(v.wer)))].sort();
  return (
    <div className="rv-table-scroll">
      <table className="rv-table rv-table-static rv-num">
        <thead>
          <tr>
            <th>Variant</th>
            <th>Sessions</th>
            <th>Turns</th>
            <th>Interrupted</th>
            <th>User→bot p50</th>
            <th>p90</th>
            <th>First speech p50</th>
            {services.map((s) => (
              <th key={s}>{s} TTFB p50</th>
            ))}
            <th>LLM TTFT text</th>
            <th>with images</th>
            <th>Prompt tok p50</th>
            {sources.map((s) => (
              <th key={s}>WER {s}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {variants.map((v, i) => (
            <tr key={v.label}>
              <td title={v.label}>
                <span className="rv-swatch" style={{ background: variantColor(v.label) }} /> {short(i)}
              </td>
              <td>{v.sessions}</td>
              <td>{v.turns}</td>
              <td>{formatPct(v.interrupt_rate)}</td>
              <td>{formatSecs(v.latency.p50)}</td>
              <td>{formatSecs(v.latency.p90)}</td>
              <td>{formatSecs(v.first_speech.p50)}</td>
              {services.map((s) => (
                <td key={s}>{formatSecs(v.ttfb[s]?.p50)}</td>
              ))}
              <td>{formatSecs(v.ttft.text.p50)}</td>
              <td>{formatSecs(v.ttft.vision.p50)}</td>
              <td>{v.prompt_tokens_p50 === null ? "–" : Math.round(v.prompt_tokens_p50)}</td>
              {sources.map((s) => (
                <td key={s}>{formatPct(v.wer[s]?.wer)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function MetricsView() {
  const [days, setDays] = useState<number | null>(7);
  const [groupBy, setGroupBy] = useState<string[]>(DEFAULT_GROUP);
  const [highlight, setHighlight] = useState(0);
  const metrics = useMetrics(days, groupBy);
  const data = metrics.data;
  const variants = data?.variants ?? [];
  const services = [...new Set(variants.flatMap((v) => Object.keys(v.ttfb)))].sort();
  const sources = [...new Set(variants.flatMap((v) => Object.keys(v.wer)))].sort();
  const hasTtft = variants.some((v) => v.ttft.text.n + v.ttft.vision.n > 0);
  const focus = Math.min(highlight, Math.max(variants.length - 1, 0));

  const toggleField = (field: string) => {
    setGroupBy((current) =>
      current.includes(field) ? current.filter((f) => f !== field) : [...current, field].slice(0, 8)
    );
    setHighlight(0);
  };

  return (
    <section className="rv-page rv-page-wide">
      <div className="rv-page-head">
        <div>
          <h2 className="text-lg font-semibold">Metrics</h2>
          <p className="text-sm text-muted">
            Compare pipeline variants. A variant is the combination of the fields selected below, taken from each
            session&apos;s recorded configuration.
          </p>
        </div>
      </div>

      <div className="rv-filters">
        <div className="rv-segmented" role="group" aria-label="Time range">
          {RANGES.map((range) => (
            <button
              key={range.label}
              className={`tab-btn ${days === range.days ? "active" : ""}`}
              onClick={() => setDays(range.days)}
            >
              {range.label}
            </button>
          ))}
        </div>
        <div className="rv-chips" role="group" aria-label="Compare by">
          <span className="text-xs text-muted">Compare by</span>
          {GROUP_FIELDS.map(({ field, label }) => (
            <button
              key={field}
              className={`rv-chip ${groupBy.includes(field) ? "active" : ""}`}
              aria-pressed={groupBy.includes(field)}
              onClick={() => toggleField(field)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <StatusLine loading={metrics.isLoading} error={metrics.error} />
      {data && data.totals.sessions === 0 && <p className="text-sm text-muted">No recorded sessions in this range.</p>}
      {data && data.totals.sessions > 0 && (
        <div className={metrics.isPlaceholderData ? "rv-refetching" : ""}>
          <div className="rv-stats-grid">
            <Tile label="Sessions" value={String(data.totals.sessions)} />
            <Tile label="User turns" value={String(data.totals.turns)} />
            <Tile label="Median user→bot" value={formatSecs(data.totals.latency_p50)} hint="user stops → bot speaks" />
            <Tile label="Live ASR WER" value={formatPct(data.totals.live_wer)} hint="vs human or reference" />
          </div>

          <Legend variants={variants} highlight={focus} onHighlight={setHighlight} />
          {variants.length > SERIES.length && (
            <p className="text-xs text-muted">
              Variants past {SERIES.length} share a gray: narrow the comparison fields to tell them apart.
            </p>
          )}

          <div className="rv-chart-grid">
            <ChartCard title="User → bot latency" subtitle="median, whisker p10–p90">
              <LatencyRange variants={variants} />
            </ChartCard>
            <ChartCard title="Latency per session" subtitle={`median; ${short(focus)} highlighted (pick in legend)`}>
              <LatencyTrend data={data} highlight={focus} />
            </ChartCard>
            <ChartCard title="Time to first byte by service" subtitle="median">
              {services.length ? (
                <GroupedBars
                  variants={variants}
                  categories={services}
                  value={(v, s) => dist(v.ttfb[s])}
                  format={formatSecs}
                  unit="s"
                />
              ) : (
                <Empty what="service TTFB" />
              )}
            </ChartCard>
            <ChartCard title="LLM time to first token" subtitle="median, text-only vs with images">
              {hasTtft ? (
                <GroupedBars
                  variants={variants}
                  categories={["Text only", "With images"]}
                  value={(v, c) => dist(c === "Text only" ? v.ttft.text : v.ttft.vision)}
                  format={formatSecs}
                  unit="s"
                />
              ) : (
                <Empty what="LLM calls" />
              )}
            </ChartCard>
            <ChartCard title="ASR word error rate" subtitle="by transcript source, vs human or reference">
              {sources.length ? (
                <GroupedBars
                  variants={variants}
                  categories={sources}
                  value={(v, s) => ({ value: v.wer[s]?.wer ?? null, n: v.wer[s]?.ref_words })}
                  format={(x) => formatPct(x)}
                  unit="%"
                />
              ) : (
                <Empty what="WER (run reasr or set human references)" />
              )}
            </ChartCard>
          </div>

          <h3 className="text-sm font-semibold">All numbers</h3>
          <SummaryTable variants={variants} />
        </div>
      )}
    </section>
  );
}
