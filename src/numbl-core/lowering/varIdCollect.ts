/**
 * Helpers that collect all VarIds referenced in IR nodes.
 * Used by lowering (to compute script-level var sets) and by codegen
 * (to generate var-save/restore frames around function calls).
 */

import { IRExpr, IRLValue, IRStmt } from "./nodes.js";
import { walkExpr } from "./nodeUtils.js";

function collectStmtVarIds(stmts: IRStmt[], out: Set<string>): void {
  for (const s of stmts) {
    collectStmtVarIdsOne(s, out);
  }
}

export function collectStmtVarIdsOne(s: IRStmt, out: Set<string>): void {
  switch (s.type) {
    case "Assign":
      out.add(s.variable.id.id);
      collectExprVarIds(s.expr, out);
      break;
    case "MultiAssign":
      for (const lv of s.lvalues) {
        if (lv) collectLValueVarIds(lv, out);
      }
      collectExprVarIds(s.expr, out);
      break;
    case "ExprStmt":
      collectExprVarIds(s.expr, out);
      break;
    case "AssignLValue":
      collectExprVarIds(s.expr, out);
      collectLValueVarIds(s.lvalue, out);
      break;
    case "If":
      collectExprVarIds(s.cond, out);
      collectStmtVarIds(s.thenBody, out);
      for (const b of s.elseifBlocks) {
        collectExprVarIds(b.cond, out);
        collectStmtVarIds(b.body, out);
      }
      if (s.elseBody) collectStmtVarIds(s.elseBody, out);
      break;
    case "While":
      collectExprVarIds(s.cond, out);
      collectStmtVarIds(s.body, out);
      break;
    case "For":
      out.add(s.variable.id.id);
      collectExprVarIds(s.expr, out);
      collectStmtVarIds(s.body, out);
      break;
    case "Switch":
      collectExprVarIds(s.expr, out);
      for (const c of s.cases) {
        collectExprVarIds(c.value, out);
        collectStmtVarIds(c.body, out);
      }
      if (s.otherwise) collectStmtVarIds(s.otherwise, out);
      break;
    case "TryCatch":
      collectStmtVarIds(s.tryBody, out);
      if (s.catchVar) out.add(s.catchVar.id.id);
      collectStmtVarIds(s.catchBody, out);
      break;
    case "Function":
      // Don't recurse into nested function definitions
      break;
    case "Global": {
      const vars = s.vars;
      vars.forEach(v => out.add(v.variable.id.id)); // is this the right thing to do? I think we don't handle globals yet
      break;
    }
    case "Return":
    case "Break":
    case "Continue":
    case "Persistent":
      break;
    default:
      throw new Error(`Unhandled statement type: ${(s as IRStmt).type}`);
  }
}

function collectExprVarIds(e: IRExpr, out: Set<string>): void {
  walkExpr(e, sub => {
    if (sub.kind.type === "Var") out.add(sub.kind.variable.id.id);
    else if (sub.kind.type === "AnonFunc") {
      for (const p of sub.kind.params) out.add(p.id.id);
    }
  });
}

function collectLValueVarIds(lv: IRLValue, out: Set<string>): void {
  switch (lv.type) {
    case "Var":
      out.add(lv.variable.id.id);
      break;
    case "Member":
    case "MemberDynamic":
      collectExprVarIds(lv.base, out);
      break;
    case "Index":
    case "IndexCell":
      collectExprVarIds(lv.base, out);
      for (const idx of lv.indices) collectExprVarIds(idx, out);
      break;
  }
}
