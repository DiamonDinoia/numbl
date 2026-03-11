// Translated from SRC/dlartg.f90
// DLARTG generates a plane rotation (Givens rotation) so that
//
//    [  C  S  ]  .  [ F ]  =  [ R ]
//    [ -S  C  ]     [ G ]     [ 0 ]
//
// where C**2 + S**2 = 1.
//
// The mathematical formulas used for C and S are:
//    R = sign(F) * sqrt(F**2 + G**2)
//    C = F / R
//    S = G / R
// Hence C >= 0. The algorithm incorporates scaling to avoid overflow
// or underflow in computing the square root of the sum of squares.
//
// This version is discontinuous in R at F = 0 but returns the same
// C and S as ZLARTG for complex inputs (F,0) and (G,0).
//
// Reference: Anderson E. (2017), Algorithm 978: Safe Scaling in the
// Level 1 BLAS, ACM Trans Math Softw 44:1--28

import { dlamch } from "./dlamch.js";
import { MACH_SFMIN } from "../utils/constants.js";

// Fortran SIGN(a,b) intrinsic: returns |a| * sign(b).
// sign(b) is +1 if b >= +0, -1 if b < 0 (including -0).
function fsign(a: number, b: number): number {
  return b >= 0 && !Object.is(b, -0) ? Math.abs(a) : -Math.abs(a);
}

export function dlartg(
  f: number,
  g: number
): { cs: number; sn: number; r: number } {
  // safmin = dlamch('S'), safmax = 1/safmin
  const safmin = dlamch(MACH_SFMIN);
  const safmax = 1.0 / safmin;
  const rtmin = Math.sqrt(safmin);
  const rtmax = Math.sqrt(safmax / 2.0);

  const f1 = Math.abs(f);
  const g1 = Math.abs(g);

  let cs: number;
  let sn: number;
  let r: number;

  if (g === 0.0) {
    cs = 1.0;
    sn = 0.0;
    r = f;
  } else if (f === 0.0) {
    cs = 0.0;
    sn = fsign(1.0, g); // sign(one, g)
    r = g1;
  } else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
    // Both f and g are in safe range; no scaling needed
    const d = Math.sqrt(f * f + g * g);
    cs = f1 / d;
    r = fsign(d, f); // sign(d, f)
    sn = g / r;
  } else {
    // Need scaling
    const u = Math.min(safmax, Math.max(safmin, f1, g1));
    const fs = f / u;
    const gs = g / u;
    const d = Math.sqrt(fs * fs + gs * gs);
    cs = Math.abs(fs) / d;
    r = fsign(d, f); // sign(d, f)
    sn = gs / r;
    r = r * u;
  }

  return { cs, sn, r };
}
