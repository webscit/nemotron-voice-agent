// SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-2-Clause

// Evaluates arithmetic expressions inside a sandboxed Web Worker.
//
// Two independent layers keep this safe for LLM-authored input:
//  1. Every network-capable global (fetch, XMLHttpRequest, WebSocket,
//     EventSource, importScripts, RTCPeerConnection, navigator.sendBeacon)
//     is disabled in the worker before anything is evaluated.
//  2. The evaluator itself only accepts a whitelisted character set and a
//     whitelisted set of `Math` member names, so nothing outside plain
//     arithmetic can execute even if a caller found a way around (1).
//
// The worker runs as a classic (non-module) worker built from a Blob so this
// stays a single self-contained file with no bundler-specific worker syntax.

const WORKER_SOURCE = `
"use strict";

// --- 1. Disable every network-capable API before evaluating anything. ---
self.fetch = undefined;
self.XMLHttpRequest = undefined;
self.WebSocket = undefined;
self.EventSource = undefined;
self.importScripts = undefined;
self.RTCPeerConnection = undefined;
if (self.navigator) {
  self.navigator.sendBeacon = undefined;
}

// --- 2. Restrict evaluation to plain arithmetic. ---
var ALLOWED_CHARS = /^[0-9+\\-*/%^().,\\sA-Za-z]*$/;
var ALLOWED_IDENTIFIERS = [
  "Math", "PI", "E",
  "abs", "ceil", "floor", "round", "trunc", "sign",
  "sqrt", "cbrt", "pow", "exp", "log", "log2", "log10",
  "min", "max", "hypot",
  "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
  "sinh", "cosh", "tanh",
];

function evaluateExpression(expression) {
  if (typeof expression !== "string" || !expression.trim()) {
    throw new Error("Expression must be a non-empty string");
  }
  if (expression.length > 200) {
    throw new Error("Expression is too long");
  }
  if (!ALLOWED_CHARS.test(expression)) {
    throw new Error("Expression contains characters that are not allowed");
  }
  var identifiers = expression.match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
  for (var i = 0; i < identifiers.length; i++) {
    if (ALLOWED_IDENTIFIERS.indexOf(identifiers[i]) === -1) {
      throw new Error("Unknown identifier: " + identifiers[i]);
    }
  }
  // '^' reads naturally as "power" but is bitwise XOR in JS; translate it.
  var normalized = expression.replace(/\\^/g, "**");

  var fn = new Function(
    "Math", "PI", "E",
    "abs", "ceil", "floor", "round", "trunc", "sign",
    "sqrt", "cbrt", "pow", "exp", "log", "log2", "log10",
    "min", "max", "hypot",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "return (" + normalized + ");"
  );
  var result = fn(
    Math, Math.PI, Math.E,
    Math.abs, Math.ceil, Math.floor, Math.round, Math.trunc, Math.sign,
    Math.sqrt, Math.cbrt, Math.pow, Math.exp, Math.log, Math.log2, Math.log10,
    Math.min, Math.max, Math.hypot,
    Math.sin, Math.cos, Math.tan, Math.asin, Math.acos, Math.atan, Math.atan2,
    Math.sinh, Math.cosh, Math.tanh
  );
  if (typeof result !== "number" || !isFinite(result)) {
    throw new Error("Expression did not evaluate to a finite number");
  }
  return result;
}

self.onmessage = function (event) {
  var id = event.data && event.data.id;
  var expression = event.data && event.data.expression;
  try {
    self.postMessage({ id: id, ok: true, result: evaluateExpression(expression) });
  } catch (err) {
    self.postMessage({ id: id, ok: false, error: err && err.message ? err.message : String(err) });
  }
};
`;

type WorkerResponse = { id: number; ok: true; result: number } | { id: number; ok: false; error: string };

const EVAL_TIMEOUT_MS = 2000;

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<
  number,
  { resolve: (value: number) => void; reject: (error: Error) => void; timeoutId: number }
>();

function failAllPending(error: Error): void {
  for (const [id, entry] of pending) {
    pending.delete(id);
    window.clearTimeout(entry.timeoutId);
    entry.reject(error);
  }
}

function getWorker(): Worker {
  if (worker) return worker;

  const blobUrl = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: "application/javascript" }));
  const instance = new Worker(blobUrl);

  instance.onmessage = (event: MessageEvent<WorkerResponse>) => {
    const entry = pending.get(event.data.id);
    if (!entry) return;
    pending.delete(event.data.id);
    window.clearTimeout(entry.timeoutId);
    if (event.data.ok) {
      entry.resolve(event.data.result);
    } else {
      entry.reject(new Error(event.data.error));
    }
  };
  instance.onerror = (event: ErrorEvent) => {
    failAllPending(new Error(event.message || "Math worker crashed"));
  };

  worker = instance;
  return instance;
}

/**
 * Recover the `expression` argument from a tool call.
 *
 * Some LLM/inference-stack combinations (seen with the Nemotron models this
 * pipeline uses through vLLM) occasionally wrap tool arguments as a
 * JSON-encoded string under `original_args` instead of top-level fields.
 */
export function resolveExpressionArgument(args: Record<string, unknown>): string {
  if (typeof args.expression === "string" && args.expression.trim()) {
    return args.expression;
  }
  if (typeof args.original_args === "string") {
    try {
      const decoded = JSON.parse(args.original_args) as { expression?: unknown };
      if (typeof decoded.expression === "string") {
        return decoded.expression;
      }
    } catch {
      // fall through
    }
  }
  return "";
}

/** Evaluate a plain arithmetic expression in the sandboxed math worker. */
export function evaluateMathExpression(expression: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const activeWorker = getWorker();
    const id = nextId++;
    const timeoutId = window.setTimeout(() => {
      pending.delete(id);
      reject(new Error("Math evaluation timed out"));
    }, EVAL_TIMEOUT_MS);
    pending.set(id, { resolve, reject, timeoutId });
    activeWorker.postMessage({ id, expression });
  });
}
