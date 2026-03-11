/**
 * Shared error diagnostic utilities for extracting structured error info
 * with source context (snippets). Used by both the CLI and the web worker.
 */

import {
  RuntimeError,
  offsetToColumn,
  offsetToLine,
  type CallFrame,
} from "../runtime/error.js";
import { SyntaxError } from "../parser/errors.js";
import { SemanticError } from "../lowering/errors.js";
import type { WorkspaceFile } from "../workspace/index.js";
import { CompilationErrors } from "./errors.js";

export interface DiagnosticInfo {
  message: string;
  errorType: "syntax" | "semantic" | "runtime" | "unknown";
  file: string | null;
  line: number | null;
  snippet: string | null;
  callStack?: CallFrame[] | null;
}

/** Look up source text for a given filename. */
export function getSourceForFile(
  file: string,
  mainFileName: string,
  mainSource: string,
  wsFiles: WorkspaceFile[] | undefined
): string | null {
  if (file === mainFileName) return mainSource;
  return wsFiles?.find(f => f.name === file)?.source ?? null;
}

/** Extract a code snippet (with context lines and pointer) around a 1-based line number. */
export function extractSnippetByLine(
  source: string,
  lineNumber: number,
  contextLines = 2,
  column?: number
): string | null {
  if (lineNumber < 1) return null;
  const lines = source.split("\n");
  if (lineNumber > lines.length) return null;

  const startLine = Math.max(1, lineNumber - contextLines);
  const endLine = Math.min(lines.length, lineNumber + contextLines);

  const gutterWidth = 6; // "  NNNN | " prefix width
  const result: string[] = [];
  for (let i = startLine; i <= endLine; i++) {
    const num = i.toString().padStart(4, " ");
    const marker = i === lineNumber ? ">" : " ";
    result.push(`${marker}${num} | ${lines[i - 1]}`);
    if (i === lineNumber && column && column >= 1) {
      result.push(" ".repeat(gutterWidth) + " ".repeat(column - 1) + "^");
    }
  }
  return result.join("\n");
}

/**
 * Extract structured diagnostic info from any error, returning an array.
 * For CompilationErrors, returns one DiagnosticInfo per inner error.
 * For all other errors, returns a single-element array.
 */
export function diagnoseErrors(
  error: unknown,
  mainSource: string,
  mainFileName: string,
  wsFiles?: WorkspaceFile[]
): DiagnosticInfo[] {
  if (error instanceof CompilationErrors) {
    return error.errors.map(e =>
      diagnoseError(e, mainSource, mainFileName, wsFiles)
    );
  }
  return [diagnoseError(error, mainSource, mainFileName, wsFiles)];
}

/** Extract structured diagnostic info from a single error thrown during compilation/execution. */
export function diagnoseError(
  error: unknown,
  mainSource: string,
  mainFileName: string,
  wsFiles?: WorkspaceFile[]
): DiagnosticInfo {
  // ── RuntimeError ─────────────────────────────────────────────────────────
  if (error instanceof RuntimeError) {
    let snippet: string | null = null;
    if (error.line !== null) {
      const file = error.file ?? mainFileName;
      const src = getSourceForFile(file, mainFileName, mainSource, wsFiles);
      snippet = src
        ? extractSnippetByLine(src, error.line, 2, error.column ?? undefined)
        : null;
    }
    return {
      message: error.message,
      errorType: "runtime",
      file: error.file,
      line: error.line,
      snippet,
      callStack: error.callStack,
    };
  }

  // ── SyntaxError ──────────────────────────────────────────────────────────
  if (error instanceof SyntaxError) {
    const src = error.file
      ? getSourceForFile(error.file, mainFileName, mainSource, wsFiles)
      : null;
    const col =
      error.column ?? (src ? offsetToColumn(src, error.position) : undefined);
    const snippet =
      src && error.line !== null
        ? extractSnippetByLine(src, error.line, 2, col)
        : null;
    return {
      message: error.message,
      errorType: "syntax",
      file: error.file,
      line: error.line,
      snippet,
      callStack: null,
    };
  }

  // ── SemanticError (with span) ─────────────────────────────────────────────
  if (error instanceof SemanticError && error.span !== null) {
    // Resolve file/line/column from span if not already set
    const file = error.file ?? error.span.file;
    const src = file
      ? getSourceForFile(file, mainFileName, mainSource, wsFiles)
      : null;
    const line =
      error.line ?? (src ? offsetToLine(src, error.span.start) : null);
    const column =
      error.column ?? (src ? offsetToColumn(src, error.span.start) : undefined);
    const snippet =
      src && line !== null ? extractSnippetByLine(src, line, 2, column) : null;
    return {
      message: error.message,
      errorType: "semantic",
      file,
      line,
      snippet,
      callStack: null,
    };
  }

  // ── SemanticError (without span) ─────────────────────────────────────────
  if (error instanceof SemanticError) {
    return {
      message: error.message,
      errorType: "semantic",
      file: null,
      line: null,
      snippet: null,
      callStack: null,
    };
  }

  // ── Unexpected errors ─────────────────────────────────────────────────────
  const message = error instanceof Error ? error.message : String(error);
  return {
    message,
    errorType: "unknown",
    file: null,
    line: null,
    snippet: null,
    callStack: null,
  };
}

/** Format multiple DiagnosticInfo as a human-readable string for console output. */
export function formatDiagnostics(infos: DiagnosticInfo[]): string {
  return infos.map(formatDiagnostic).join("\n\n");
}

/** Format a DiagnosticInfo as a human-readable string for console output. */
export function formatDiagnostic(info: DiagnosticInfo): string {
  const labels: Record<DiagnosticInfo["errorType"], string> = {
    syntax: "SyntaxError",
    semantic: "SemanticError",
    runtime: "RuntimeError",
    unknown: "Error",
  };

  let result = labels[info.errorType];

  if (info.file && info.line !== null) {
    result += ` at ${info.file}:${info.line}`;
  } else if (info.line !== null) {
    result += ` at line ${info.line}`;
  }

  result += `: ${info.message}`;

  if (info.snippet) {
    result += `\n${info.snippet}`;
  }

  if (info.callStack != null && info.callStack.length > 0) {
    result += `\nCall stack (most recent call first):`;
    const stack = info.callStack;
    const N = stack.length;
    for (let i = N - 1; i >= 0; i--) {
      const name = stack[i].name;
      let loc: string;
      if (i === N - 1) {
        if (info.file && info.line !== null) {
          loc = `${info.file}:${info.line}`;
        } else if (info.line !== null) {
          loc = `line ${info.line}`;
        } else {
          loc = "unknown";
        }
      } else {
        const callerFrame = stack[i + 1];
        if (callerFrame.callerFile && callerFrame.callerLine > 0) {
          loc = `${callerFrame.callerFile}:${callerFrame.callerLine}`;
        } else if (callerFrame.callerLine > 0) {
          loc = `line ${callerFrame.callerLine}`;
        } else {
          loc = "unknown";
        }
      }
      result += `\n  at ${name} (${loc})`;
    }
  }

  return result;
}
