/**
 * Semantic Errors
 */

import type { Span } from "../parser";

export class SemanticError extends Error {
  span: Span | null;
  identifier: string | null;
  /** Resolved source file name (set by prepare.ts) */
  file: string | null;
  /** Resolved 1-based line number (set by prepare.ts) */
  line: number | null;
  /** Resolved 1-based column number (set by prepare.ts) */
  column: number | undefined;

  constructor(
    message: string,
    span: Span | null = null,
    identifier: string | null = null
  ) {
    super(message);
    this.name = "SemanticError";
    this.span = span;
    this.identifier = identifier;
    this.file = null;
    this.line = null;
    this.column = undefined;
  }

  withSpan(span: Span): SemanticError {
    this.span = span;
    return this;
  }

  withIdentifier(identifier: string): SemanticError {
    this.identifier = identifier;
    return this;
  }
}
