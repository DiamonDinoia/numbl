// Translated from SRC/dlanv2.f
// DLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric
// matrix in standard form:
//
//      [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ]
//      [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ]
//
// where either
// 1) CC = 0 so that AA and DD are real eigenvalues of the matrix, or
// 2) AA = DD and BB*CC < 0, so that AA +/- sqrt(BB*CC) are complex
//    conjugate eigenvalues.

import { dlamch } from "./dlamch.js";
import { dlapy2 } from "./dlapy2.js";
import { MACH_SFMIN, MACH_PREC, MACH_BASE } from "../utils/constants.js";

/** Fortran SIGN(A,B): returns |A| with the sign of B. Treats B=0 as positive. */
function fsign(a: number, b: number): number {
  return b >= 0 ? Math.abs(a) : -Math.abs(a);
}

export function dlanv2(
  a: number,
  b: number,
  c: number,
  d: number
): {
  a: number;
  b: number;
  c: number;
  d: number;
  rt1r: number;
  rt1i: number;
  rt2r: number;
  rt2i: number;
  cs: number;
  sn: number;
} {
  const ZERO = 0.0;
  const HALF = 0.5;
  const ONE = 1.0;
  const TWO = 2.0;
  const MULTPL = 4.0;

  const safmin = dlamch(MACH_SFMIN);
  const eps = dlamch(MACH_PREC);
  const base = dlamch(MACH_BASE);
  const safmn2 = Math.pow(
    base,
    Math.trunc(Math.log(safmin / eps) / Math.log(base) / TWO)
  );
  const safmx2 = ONE / safmn2;

  let cs: number;
  let sn: number;
  let rt1i: number;
  let rt2i: number;
  let temp: number;
  let p: number;
  let tau: number;

  if (c === ZERO) {
    cs = ONE;
    sn = ZERO;
  } else if (b === ZERO) {
    // Swap rows and columns
    cs = ZERO;
    sn = ONE;
    temp = d;
    d = a;
    a = temp;
    b = -c;
    c = ZERO;
  } else if (a - d === ZERO && Math.sign(b) !== Math.sign(c)) {
    // Note: Fortran SIGN(ONE, X) returns |ONE| with sign of X.
    // Math.sign returns -1,0,+1 so comparing signs works equivalently.
    cs = ONE;
    sn = ZERO;
  } else {
    temp = a - d;
    p = HALF * temp;
    const bcmax = Math.max(Math.abs(b), Math.abs(c));
    const bcmis =
      Math.min(Math.abs(b), Math.abs(c)) * Math.sign(b) * Math.sign(c);
    const scale = Math.max(Math.abs(p), bcmax);
    let z = (p / scale) * p + (bcmax / scale) * bcmis;

    // If Z is of the order of the machine accuracy, postpone the
    // decision on the nature of eigenvalues
    if (z >= MULTPL * eps) {
      // Real eigenvalues. Compute A and D.
      z = p + fsign(Math.sqrt(scale) * Math.sqrt(z), p);
      a = d + z;
      d = d - (bcmax / z) * bcmis;

      // Compute B and the rotation matrix
      tau = dlapy2(c, z);
      cs = z / tau;
      sn = c / tau;
      b = b - c;
      c = ZERO;
    } else {
      // Complex eigenvalues, or real (almost) equal eigenvalues.
      // Make diagonal elements equal.
      let count = 0;
      let sigma = b + c;

      // Scaling loop (translates Fortran GOTO 10)
      let keepScaling = true;
      while (keepScaling) {
        keepScaling = false;
        count = count + 1;
        const loopScale = Math.max(Math.abs(temp), Math.abs(sigma));
        if (loopScale >= safmx2) {
          sigma = sigma * safmn2;
          temp = temp * safmn2;
          if (count <= 20) {
            keepScaling = true;
            continue;
          }
        }
        if (loopScale <= safmn2) {
          sigma = sigma * safmx2;
          temp = temp * safmx2;
          if (count <= 20) {
            keepScaling = true;
            continue;
          }
        }
      }

      p = HALF * temp;
      tau = dlapy2(sigma, temp);
      cs = Math.sqrt(HALF * (ONE + Math.abs(sigma) / tau));
      sn = -(p / (tau * cs)) * fsign(ONE, sigma);

      // Compute [ AA  BB ] = [ A  B ] [ CS -SN ]
      //         [ CC  DD ]   [ C  D ] [ SN  CS ]
      const aa = a * cs + b * sn;
      const bb = -a * sn + b * cs;
      const cc = c * cs + d * sn;
      const dd = -c * sn + d * cs;

      // Compute [ A  B ] = [ CS  SN ] [ AA  BB ]
      //         [ C  D ]   [-SN  CS ] [ CC  DD ]
      a = aa * cs + cc * sn;
      b = bb * cs + dd * sn;
      c = -(aa * sn) + cc * cs;
      d = -bb * sn + dd * cs;

      temp = HALF * (a + d);
      a = temp;
      d = temp;

      if (c !== ZERO) {
        if (b !== ZERO) {
          if (Math.sign(b) === Math.sign(c)) {
            // Real eigenvalues: reduce to upper triangular form
            const sab = Math.sqrt(Math.abs(b));
            const sac = Math.sqrt(Math.abs(c));
            p = fsign(sab * sac, c);
            tau = ONE / Math.sqrt(Math.abs(b + c));
            a = temp + p;
            d = temp - p;
            b = b - c;
            c = ZERO;
            const cs1 = sab * tau;
            const sn1 = sac * tau;
            temp = cs * cs1 - sn * sn1;
            sn = cs * sn1 + sn * cs1;
            cs = temp;
          }
        } else {
          b = -c;
          c = ZERO;
          temp = cs;
          cs = -sn;
          sn = temp;
        }
      }
    }
  }

  // Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I).
  const rt1r = a;
  const rt2r = d;
  if (c === ZERO) {
    rt1i = ZERO;
    rt2i = ZERO;
  } else {
    rt1i = Math.sqrt(Math.abs(b)) * Math.sqrt(Math.abs(c));
    rt2i = -rt1i;
  }

  return { a, b, c, d, rt1r, rt1i, rt2r, rt2i, cs, sn };
}
