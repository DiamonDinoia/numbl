/**
 * Shared types for the AST-to-IR lowering pass.
 */

import { VarId } from "./varId.js";
import { ItemType } from "./itemTypes.js";
import type { IRProgram, IRStmt } from "./nodes.js";

export interface IRVariable {
  id: VarId;
  name: string;
  // undefined means it hasn't been assigned yet, whereas
  // unknown means it has been assigned and is unknown
  ty: ItemType | undefined;
  isTopLevel?: boolean;
}

export interface LoweringResult {
  irProgram: IRProgram;
  variables: Map<string, IRVariable>;
  functions: Map<string, IRStmt & { type: "Function" }>; // by function ID
  classes: Map<string, IRStmt & { type: "ClassDef" }>;
  /** VarIds referenced in top-level (script-scope) statements. */
  scriptVarIds: Set<string>;
}
