import { RuntimeValue, toNumber, RuntimeError } from "../runtime/index.js";
import {
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
} from "../runtime/types.js";
import { builtinSingle } from "./registry.js";
import { register } from "./registry.js";

// ── Helpers ─────────────────────────────────────────────────────────────

function isNumeric(v: RuntimeValue): boolean {
  return isRuntimeNumber(v) || isRuntimeTensor(v) || isRuntimeLogical(v);
}

function isInteger(n: number): boolean {
  return Number.isFinite(n) && Math.floor(n) === n;
}

function numericElements(v: RuntimeValue): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeTensor(v)) return Array.from(v.data);
  return [];
}

function numel(v: RuntimeValue): number {
  if (isRuntimeNumber(v) || isRuntimeLogical(v)) return 1;
  if (isRuntimeTensor(v)) return v.data.length;
  if (isRuntimeChar(v)) return v.value.length;
  if (isRuntimeString(v)) return 1;
  if (isRuntimeCell(v)) return v.data.length;
  return 0;
}

// ── Validators ──────────────────────────────────────────────────────────

/** Register a validator that checks each numeric element against a predicate */
function makeElementValidator(
  name: string,
  predicate: (n: number) => boolean,
  message: string
): void {
  register(
    name,
    builtinSingle(args => {
      const v = args[0];
      if (!v) throw new RuntimeError(`${name}: missing argument`);
      if (!isNumeric(v)) throw new RuntimeError(message);
      for (const n of numericElements(v)) {
        if (!predicate(n)) throw new RuntimeError(message);
      }
      return 0 as RuntimeValue;
    })
  );
}

export function registerValidatorFunctions(): void {
  register(
    "mustBeNumeric",
    builtinSingle(args => {
      const v = args[0];
      if (!v) throw new RuntimeError("mustBeNumeric: missing argument");
      if (!isNumeric(v)) throw new RuntimeError("Value must be numeric.");
      return 0 as RuntimeValue;
    })
  );

  makeElementValidator("mustBeInteger", isInteger, "Value must be integer.");
  makeElementValidator("mustBePositive", n => n > 0, "Value must be positive.");
  makeElementValidator(
    "mustBeNonnegative",
    n => n >= 0,
    "Value must be nonnegative."
  );
  makeElementValidator("mustBeNonzero", n => n !== 0, "Value must be nonzero.");
  makeElementValidator(
    "mustBeFinite",
    n => Number.isFinite(n),
    "Value must be finite."
  );

  // mustBeNonempty: value must not be empty
  register(
    "mustBeNonempty",
    builtinSingle(args => {
      const v = args[0];
      if (!v) throw new RuntimeError("mustBeNonempty: missing argument");
      if (numel(v) === 0) throw new RuntimeError("Value must be nonempty.");
      return 0 as RuntimeValue;
    })
  );

  // mustBeScalarOrEmpty: value must be scalar (1 element) or empty (0 elements)
  register(
    "mustBeScalarOrEmpty",
    builtinSingle(args => {
      const v = args[0];
      if (!v) throw new RuntimeError("mustBeScalarOrEmpty: missing argument");
      const n = numel(v);
      if (n !== 0 && n !== 1) {
        throw new RuntimeError("Value must be scalar or empty.");
      }
      return 0 as RuntimeValue;
    })
  );

  // mustBeVector: value must be a vector (1-D or row/column)
  register(
    "mustBeVector",
    builtinSingle(args => {
      const v = args[0];
      if (!v) throw new RuntimeError("mustBeVector: missing argument");
      if (isRuntimeTensor(v)) {
        const isVec =
          v.shape.length <= 2 &&
          (v.shape[0] === 1 || v.shape[1] === 1 || v.data.length === 0);
        if (!isVec) throw new RuntimeError("Value must be a vector.");
      } else if (!isRuntimeNumber(v) && !isRuntimeLogical(v)) {
        throw new RuntimeError("Value must be a vector.");
      }
      return 0 as RuntimeValue;
    })
  );

  // mustBeMember: value must be a member of a set
  register(
    "mustBeMember",
    builtinSingle(args => {
      const v = args[0];
      const set = args[1];
      if (!v || !set)
        throw new RuntimeError("mustBeMember: requires 2 arguments");
      const vElems = isRuntimeString(v) ? [v] : numericElements(v);
      const setElems = isRuntimeCell(set)
        ? set.data
        : isRuntimeString(set)
          ? [set]
          : numericElements(set);
      for (const elem of vElems) {
        const found = (setElems as unknown[]).some(s =>
          typeof s === "object" &&
          s !== null &&
          "kind" in s &&
          isRuntimeString(s as RuntimeValue)
            ? (s as { kind: string; value: string }).value === String(elem)
            : s === elem
        );
        if (!found)
          throw new RuntimeError(
            "Value must be a member of the specified set."
          );
      }
      return 0 as RuntimeValue;
    })
  );

  // mustBeInRange: value must be within [lower, upper]
  register(
    "mustBeInRange",
    builtinSingle(args => {
      const v = args[0];
      const lower = args[1] ? toNumber(args[1]) : -Infinity;
      const upper = args[2] ? toNumber(args[2]) : Infinity;
      if (!v) throw new RuntimeError("mustBeInRange: missing argument");
      if (!isNumeric(v)) throw new RuntimeError("Value must be in range.");
      for (const n of numericElements(v)) {
        if (n < lower || n > upper) {
          throw new RuntimeError(
            `Value must be in range [${lower}, ${upper}].`
          );
        }
      }
      return 0 as RuntimeValue;
    })
  );
}
