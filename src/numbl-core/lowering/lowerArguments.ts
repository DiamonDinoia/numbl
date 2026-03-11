import type { ArgumentsBlock, Expr } from "../parser/index.js";
import type { IRExpr } from "./nodes.js";
import { IRArgumentsBlock, IRArgumentEntry } from "./nodes.js";

/**
 * Lower a list of arguments blocks.
 *
 * @param blocks  - AST-level arguments blocks
 * @param lowerExpr - callback to lower an AST Expr to a IRExpr
 */
export function lowerArgumentsBlocks(
  blocks: ArgumentsBlock[],
  lowerExpr: (expr: Expr) => IRExpr
): IRArgumentsBlock[] {
  return blocks.map(block => ({
    kind: block.kind,
    entries: block.entries.map(
      (entry): IRArgumentEntry => ({
        name: entry.name,
        dimensions: entry.dimensions,
        className: entry.className,
        validators: entry.validators,
        defaultValue: entry.defaultValue ? lowerExpr(entry.defaultValue) : null,
      })
    ),
  }));
}
