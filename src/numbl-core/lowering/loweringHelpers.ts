/**
 * Helper utilities for lowering.
 *
 * Pre-defines variables and collects assigned names for control flow analysis.
 */

import { type Stmt as AstStmt } from "../parser/index.js";
import { type LoweringContext } from "./loweringContext.js";

/**
 * Pre-define variables assigned in a body so they're visible across iterations.
 * This is needed for loops where a variable assigned in later iterations
 * should be accessible in earlier iterations' conditionals.
 */
export function preDefineBodyVars(
  ctx: LoweringContext,
  stmts: AstStmt[],
  alreadyDefined: Set<string>
): void {
  for (const s of stmts) {
    collectAssignedNames(s, alreadyDefined).forEach(name => {
      if (!alreadyDefined.has(name)) {
        alreadyDefined.add(name);
        if (!ctx.lookup(name)) {
          ctx.defineVariable(name, undefined);
        }
      }
    });
  }
}

/**
 * Collect variable names assigned in a statement (not into nested functions).
 * This recursively traverses control flow structures but stops at function boundaries.
 */
export function collectAssignedNames(
  stmt: AstStmt,
  skip: Set<string>
): string[] {
  const names: string[] = [];
  switch (stmt.type) {
    case "Assign":
      names.push(stmt.name);
      break;
    case "For":
      names.push(stmt.varName);
      for (const s of stmt.body) names.push(...collectAssignedNames(s, skip));
      break;
    case "If":
      for (const s of stmt.thenBody)
        names.push(...collectAssignedNames(s, skip));
      for (const b of stmt.elseifBlocks)
        for (const s of b.body) names.push(...collectAssignedNames(s, skip));
      if (stmt.elseBody)
        for (const s of stmt.elseBody)
          names.push(...collectAssignedNames(s, skip));
      break;
    case "While":
      for (const s of stmt.body) names.push(...collectAssignedNames(s, skip));
      break;
    case "TryCatch":
      for (const s of stmt.tryBody)
        names.push(...collectAssignedNames(s, skip));
      for (const s of stmt.catchBody)
        names.push(...collectAssignedNames(s, skip));
      break;
    case "Switch":
      for (const c of stmt.cases)
        for (const s of c.body) names.push(...collectAssignedNames(s, skip));
      if (stmt.otherwise)
        for (const s of stmt.otherwise)
          names.push(...collectAssignedNames(s, skip));
      break;
    case "MultiAssign":
      for (const lv of stmt.lvalues) {
        if (lv && lv.type === "Var") names.push(lv.name);
      }
      break;
    case "AssignLValue":
      if (stmt.lvalue.type === "Var") names.push(stmt.lvalue.name);
      break;
    // Function: don't recurse into nested functions
  }
  return names;
}
