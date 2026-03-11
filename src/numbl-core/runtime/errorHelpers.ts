/**
 * Error helper utilities for creating context-aware RuntimeErrors.
 */

import { RuntimeError } from "./error.js";
import type { Span } from "../parser/types.js";
import type { IRExpr, IRStmt } from "../lowering/nodes.js";

/**
 * Create a RuntimeError with optional span context.
 */
export function runtimeError(
  message: string,
  span?: Span | null
): RuntimeError {
  return new RuntimeError(message, span ?? undefined);
}

/**
 * Create a RuntimeError from an IR expression (auto-extracts span).
 */
export function errorFromExpr(message: string, expr: IRExpr): RuntimeError {
  return new RuntimeError(message, expr.span ?? undefined);
}

/**
 * Create a RuntimeError from an IR statement (auto-extracts span).
 */
export function errorFromStmt(message: string, stmt: IRStmt): RuntimeError {
  return new RuntimeError(message, stmt.span ?? undefined);
}

/**
 * Format a RuntimeError for display with file:line and snippet.
 * This is a convenience wrapper around RuntimeError.toString().
 */
export function formatError(
  err: RuntimeError,
  fileSources?: Map<string, string>
): string {
  // If fileSources is provided and error has no context yet, enrich it
  if (fileSources && err.span && !err.snippet) {
    err.withContext(fileSources);
  }
  return err.toString();
}
