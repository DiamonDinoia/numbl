// Translated from SRC/dlapy2.f
// DLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary
// overflow and unnecessary underflow.

import { dlamch } from "./dlamch.js";
import { MACH_RMAX } from "../utils/constants.js";

export function dlapy2(x: number, y: number): number {
  const xIsNan = Number.isNaN(x);
  const yIsNan = Number.isNaN(y);
  if (xIsNan) return x;
  if (yIsNan) return y;

  const hugeval = dlamch(MACH_RMAX); // overflow threshold

  const xabs = Math.abs(x);
  const yabs = Math.abs(y);
  const w = Math.max(xabs, yabs);
  const z = Math.min(xabs, yabs);

  if (z === 0.0 || w > hugeval) {
    return w;
  } else {
    return w * Math.sqrt(1.0 + (z / w) * (z / w));
  }
}
