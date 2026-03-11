/**
 * Codegen helper utilities.
 *
 * Standalone functions used by the codegen pipeline: type suffix encoding,
 * JS reserved words, variable ID collection, and expression analysis.
 */

import {
  type IRExpr,
  type IRLValue,
  type IRStmt,
  walkExpr,
} from "../lowering/index.js";
import type { LoweringContext } from "../lowering/loweringContext.js";
import type { Expr } from "../parser/index.js";
import { lowerExpr } from "../lowering/lowerExpr.js";
import type { Codegen } from "./codegen.js";
// Re-export specialization key utilities from shared module
export { computeSpecKey, hashForJsId } from "../lowering/specKey.js";

// ── JS reserved words ───────────────────────────────────────────────────

export const JS_RESERVED = new Set([
  "break",
  "case",
  "catch",
  "class",
  "const",
  "continue",
  "debugger",
  "default",
  "delete",
  "do",
  "else",
  "export",
  "extends",
  "finally",
  "for",
  "function",
  "if",
  "import",
  "in",
  "instanceof",
  "let",
  "new",
  "of",
  "return",
  "super",
  "switch",
  "this",
  "throw",
  "try",
  "typeof",
  "var",
  "void",
  "while",
  "with",
  "yield",
  "implements",
  "interface",
  "package",
  "private",
  "protected",
  "public",
  "static",
  "undefined",
  "NaN",
  "Infinity",
  "arguments",
  "eval",
  "Math",
  "Array",
  "Object",
  "String",
  "Number",
  "Boolean",
  "Error",
  "Map",
  "Set",
  "JSON",
]);

// ── Expression analysis ─────────────────────────────────────────────────

/**
 * Like exprContainsEnd but does NOT recurse into the indices of inner
 * Index/IndexCell expressions.  `end` inside a nested indexing like
 * `ii(end:-1:3)` is scoped to that inner indexing operation and will be
 * resolved by the inner genIndexArg, so the outer level should not wrap
 * it in a deferred-end callback.
 */
export function exprContainsEndShallow(expr: IRExpr): boolean {
  let found = false;
  function walk(e: IRExpr, insideInner: boolean): void {
    const k = e.kind;
    if (k.type === "End") {
      if (!insideInner) found = true;
      return;
    }
    switch (k.type) {
      case "Unary":
        walk(k.operand, insideInner);
        break;
      case "Binary":
        walk(k.left, insideInner);
        walk(k.right, insideInner);
        break;
      case "Tensor":
      case "Cell":
        for (const row of k.rows) for (const el of row) walk(el, insideInner);
        break;
      case "Range":
        walk(k.start, insideInner);
        if (k.step) walk(k.step, insideInner);
        walk(k.end, insideInner);
        break;
      case "Member":
        walk(k.base, insideInner);
        break;
      case "MemberDynamic":
        walk(k.base, insideInner);
        walk(k.nameExpr, insideInner);
        break;
      case "Index":
      case "IndexCell":
        // Walk base at the same level, but indices are inside an inner
        // indexing context — `end` there is resolved by that inner operation.
        walk(k.base, insideInner);
        for (const idx of k.indices) walk(idx, true);
        break;
      case "MethodCall":
        walk(k.base, insideInner);
        for (const a of k.args) walk(a, insideInner);
        break;
      case "SuperConstructorCall":
      case "FuncCall":
        if (k.type === "FuncCall" && k.instanceBase)
          walk(k.instanceBase, insideInner);
        for (const a of k.args) walk(a, insideInner);
        break;
      case "ClassInstantiation":
        for (const a of k.args) walk(a, insideInner);
        break;
      case "AnonFunc":
        walk(k.body, insideInner);
        break;
    }
  }
  walk(expr, false);
  return found;
}

// ── Variable ID collection ──────────────────────────────────────────────

/** Collect VarIds referenced in IR statements (non-recursive into nested Functions). */
export function collectVarIds(stmts: IRStmt[], out: Set<string>): void {
  for (const stmt of stmts) {
    if (stmt.type === "Function") continue; // Don't recurse into nested functions
    collectVarIdsFromStmt(stmt, out);
  }
}

function collectVarIdsFromStmt(stmt: IRStmt, out: Set<string>): void {
  switch (stmt.type) {
    case "Assign":
      out.add(stmt.variable.id.id);
      collectVarIdsFromExpr(stmt.expr, out);
      break;
    case "ExprStmt":
      collectVarIdsFromExpr(stmt.expr, out);
      break;
    case "MultiAssign":
      collectVarIdsFromExpr(stmt.expr, out);
      for (const lv of stmt.lvalues) {
        if (lv) collectVarIdsFromLValue(lv, out);
      }
      break;
    case "AssignLValue":
      collectVarIdsFromLValue(stmt.lvalue, out);
      collectVarIdsFromExpr(stmt.expr, out);
      break;
    case "If":
      collectVarIdsFromExpr(stmt.cond, out);
      collectVarIds(stmt.thenBody, out);
      for (const b of stmt.elseifBlocks) {
        collectVarIdsFromExpr(b.cond, out);
        collectVarIds(b.body, out);
      }
      if (stmt.elseBody) collectVarIds(stmt.elseBody, out);
      break;
    case "While":
      collectVarIdsFromExpr(stmt.cond, out);
      collectVarIds(stmt.body, out);
      break;
    case "For":
      out.add(stmt.variable.id.id);
      collectVarIdsFromExpr(stmt.expr, out);
      collectVarIds(stmt.body, out);
      break;
    case "Switch":
      collectVarIdsFromExpr(stmt.expr, out);
      for (const c of stmt.cases) {
        collectVarIdsFromExpr(c.value, out);
        collectVarIds(c.body, out);
      }
      if (stmt.otherwise) collectVarIds(stmt.otherwise, out);
      break;
    case "TryCatch":
      collectVarIds(stmt.tryBody, out);
      if (stmt.catchVar) out.add(stmt.catchVar.id.id);
      collectVarIds(stmt.catchBody, out);
      break;
    case "Global":
    case "Persistent":
      for (const v of stmt.vars) out.add(v.variable.id.id);
      break;
    case "Return":
    case "Break":
    case "Continue":
      break;
    // Function: handled at top level of collectVarIds
  }
}

function collectVarIdsFromExpr(expr: IRExpr, out: Set<string>): void {
  walkExpr(expr, e => {
    if (e.kind.type === "Var") out.add(e.kind.variable.id.id);
  });
}

// ── Class property helpers ────────────────────────────────────────────────

/**
 * Collect all property names and defaults from a class and its ancestors.
 * Walks the inheritance chain (child first, then parents).
 */
export function collectClassProperties(
  ctx: LoweringContext,
  className: string
): { propertyNames: string[]; propertyDefaults: Map<string, Expr> } {
  const classInfo = ctx.getClassInfo(className);
  if (!classInfo) return { propertyNames: [], propertyDefaults: new Map() };

  const propertyNames: string[] = [...classInfo.propertyNames];
  const propertyDefaults = new Map(classInfo.propertyDefaults);
  let parentName = classInfo.superClass;
  while (parentName) {
    const parentInfo = ctx.getClassInfo(parentName);
    if (!parentInfo) break;
    for (const propName of parentInfo.propertyNames) {
      if (!propertyNames.includes(propName)) {
        propertyNames.push(propName);
        const defaultExpr = parentInfo.propertyDefaults.get(propName);
        if (defaultExpr && !propertyDefaults.has(propName)) {
          propertyDefaults.set(propName, defaultExpr);
        }
      }
    }
    parentName = parentInfo.superClass;
  }
  return { propertyNames, propertyDefaults };
}

/**
 * Generate the JS defaults argument for createClassInstance.
 * Returns "undefined" if no defaults, or a JS object literal string.
 */
export function genPropertyDefaults(
  cg: Codegen,
  className: string,
  defaults: Map<string, Expr>
): string {
  if (defaults.size === 0) return "undefined";
  const ctx =
    cg.loweringCtx.getOrCreateClassFileContext(className) ?? cg.loweringCtx;
  const entries: string[] = [];
  for (const [propName, astExpr] of defaults) {
    const irExpr = lowerExpr(ctx, astExpr);
    const jsExpr = cg.genExpr(irExpr);
    entries.push(`${JSON.stringify(propName)}: ${jsExpr}`);
  }
  return `{${entries.join(", ")}}`;
}

function collectVarIdsFromLValue(lv: IRLValue, out: Set<string>): void {
  if (lv.type === "Var") {
    out.add(lv.variable.id.id);
  } else if (lv.type === "Member" || lv.type === "MemberDynamic") {
    collectVarIdsFromExpr(lv.base, out);
    if (lv.type === "MemberDynamic") collectVarIdsFromExpr(lv.nameExpr, out);
  } else if (lv.type === "Index" || lv.type === "IndexCell") {
    collectVarIdsFromExpr(lv.base, out);
    for (const i of lv.indices) collectVarIdsFromExpr(i, out);
  }
}
