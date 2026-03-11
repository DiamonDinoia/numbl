// Translated from SRC/dladiv.f
// DLADIV performs complex division in real arithmetic, avoiding unnecessary
// overflow.
//
//                    a + i*b
//         p + i*q = ---------
//                    c + i*d
//
// The algorithm is due to Michael Baudin and Robert L. Smith
// and can be found in the paper "A Robust Complex Division in Scilab".

import { dlamch } from "./dlamch.js";
import { MACH_RMAX, MACH_SFMIN, MACH_EPS } from "../utils/constants.js";

export function dladiv(
  a: number,
  b: number,
  c: number,
  d: number
): { p: number; q: number } {
  const BS = 2.0;
  const HALF = 0.5;
  const TWO = 2.0;

  let aa = a;
  let bb = b;
  let cc = c;
  let dd = d;
  const ab = Math.max(Math.abs(a), Math.abs(b));
  const cd = Math.max(Math.abs(c), Math.abs(d));
  let s = 1.0;

  const ov = dlamch(MACH_RMAX);
  const un = dlamch(MACH_SFMIN);
  const eps = dlamch(MACH_EPS);
  const be = BS / (eps * eps);

  if (ab >= HALF * ov) {
    aa = HALF * aa;
    bb = HALF * bb;
    s = TWO * s;
  }
  if (cd >= HALF * ov) {
    cc = HALF * cc;
    dd = HALF * dd;
    s = HALF * s;
  }
  if (ab <= (un * BS) / eps) {
    aa = aa * be;
    bb = bb * be;
    s = s / be;
  }
  if (cd <= (un * BS) / eps) {
    cc = cc * be;
    dd = dd * be;
    s = s * be;
  }

  let p: number;
  let q: number;
  if (Math.abs(dd) <= Math.abs(cc)) {
    ({ p, q } = dladiv1(aa, bb, cc, dd));
  } else {
    ({ p, q } = dladiv1(bb, aa, dd, cc));
    q = -q;
  }
  p = p * s;
  q = q * s;

  return { p, q };
}

export function dladiv1(
  a: number,
  b: number,
  c: number,
  d: number
): { p: number; q: number } {
  const ONE = 1.0;

  const r = d / c;
  const t = ONE / (c + d * r);
  const p = dladiv2(a, b, c, d, r, t);
  const q = dladiv2(b, -a, c, d, r, t);

  return { p, q };
}

export function dladiv2(
  a: number,
  b: number,
  c: number,
  d: number,
  r: number,
  t: number
): number {
  const ZERO = 0.0;

  if (r !== ZERO) {
    const br = b * r;
    if (br !== ZERO) {
      return (a + br) * t;
    } else {
      return a * t + b * t * r;
    }
  } else {
    return (a + d * (b / c)) * t;
  }
}
