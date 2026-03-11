/**
 * Helper to call a registered builtin from within another builtin's apply().
 */

import { RuntimeError, RuntimeValue } from "../../runtime/index.js";
import { getBuiltin } from "../registry.js";

export function applyBuiltin(
  caller: string,
  name: string,
  args: RuntimeValue[],
  nargout: number
): RuntimeValue {
  const branches = getBuiltin(name);
  if (!branches)
    throw new RuntimeError(`${caller}: builtin '${name}' not found`);
  for (const branch of branches) {
    const result = branch.apply(args, nargout);
    if (result !== undefined) {
      if (result instanceof Promise)
        throw new RuntimeError(
          `${caller}: builtin '${name}' returned async result`
        );
      if (Array.isArray(result)) return result[0];
      return result;
    }
  }
  throw new RuntimeError(`${caller}: builtin '${name}' returned no result`);
}
