/**
 * Shared shape/size argument parsing for array constructors
 * (zeros, ones, rand, randn, true, false, etc.)
 */

import { type RuntimeValue, toNumber } from "../runtime/index.js";
import { isRuntimeTensor } from "../runtime/types.js";

/** Parse shape arguments: zeros(2,3) or zeros([2,3]) -> [2, 3]
 *  Negative dimensions are clamped to 0 */
export function parseShapeArgs(args: RuntimeValue[]): number[] {
  if (args.length === 1 && isRuntimeTensor(args[0])) {
    const t = args[0];
    const shape: number[] = [];
    for (let i = 0; i < t.data.length; i++)
      shape.push(Math.max(0, Math.round(t.data[i])));
    return shape;
  }
  return args.map(a => Math.max(0, Math.round(toNumber(a))));
}
