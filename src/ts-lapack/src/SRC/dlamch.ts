// Translated from INSTALL/dlamch.f
// DLAMCH determines double precision machine parameters.
//
// cmach codes:
//   'E'  eps     - relative machine precision (unit roundoff)
//   'S'  sfmin   - safe minimum: 1/sfmin does not overflow
//   'B'  base    - base of the machine (2 for IEEE 754)
//   'P'  prec    - eps * base
//   'N'  t       - number of mantissa digits (53 for float64)
//   'R'  rnd     - 1.0 (rounding mode; we always assume round-to-nearest)
//   'M'  emin    - minimum exponent before underflow (-1021)
//   'U'  rmin    - underflow threshold (smallest positive normal)
//   'L'  emax    - largest exponent before overflow (1024)
//   'O'  rmax    - overflow threshold

import {
  MACH_EPS,
  MACH_SFMIN,
  MACH_BASE,
  MACH_PREC,
  MACH_T,
  MACH_RND,
  MACH_EMIN,
  MACH_RMIN,
  MACH_EMAX,
  MACH_RMAX,
} from "../utils/constants.js";

// Smallest positive normalised float64 = 2^-1022
const SFMIN = 2.2250738585072014e-308;

export function dlamch(cmach: number): number {
  // eps = epsilon / 2 (assuming rounding, not chopping)
  const eps = Number.EPSILON * 0.5;

  if (cmach === MACH_EPS) {
    return eps;
  } else if (cmach === MACH_SFMIN) {
    const small = 1.0 / Number.MAX_VALUE;
    return small >= SFMIN ? small * (1.0 + eps) : SFMIN;
  } else if (cmach === MACH_BASE) {
    return 2.0;
  } else if (cmach === MACH_PREC) {
    return eps * 2.0;
  } else if (cmach === MACH_T) {
    return 53.0;
  } else if (cmach === MACH_RND) {
    return 1.0;
  } else if (cmach === MACH_EMIN) {
    return -1021.0;
  } else if (cmach === MACH_RMIN) {
    return SFMIN;
  } else if (cmach === MACH_EMAX) {
    return 1024.0;
  } else if (cmach === MACH_RMAX) {
    return Number.MAX_VALUE;
  }
  return 0.0;
}
