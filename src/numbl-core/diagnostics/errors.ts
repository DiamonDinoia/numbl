/**
 * Aggregate compilation error — collects multiple errors from parsing/lowering
 * across workspace files so that all errors can be reported at once.
 */
export class CompilationErrors extends Error {
  errors: Error[];
  constructor(errors: Error[]) {
    const count = errors.length;
    super(`${count} compilation error${count === 1 ? "" : "s"}`);
    this.name = "CompilationErrors";
    this.errors = errors;
  }
}
