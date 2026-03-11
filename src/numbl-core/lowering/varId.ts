/**
 * Variable Identifier
 */

/** Stable variable identifier after name binding */
export class VarId {
  constructor(public readonly id: string) {}

  toString(): string {
    return `VarId(${this.id})`;
  }
}
