// Translated from LAPACK/SRC/dlasv2.f
// DLASV2 computes the singular value decomposition of a 2-by-2
// triangular matrix
//    [  F   G  ]
//    [  0   H  ].
// On return, abs(SSMAX) is the larger singular value, abs(SSMIN) is the
// smaller singular value, and (CSL,SNL) and (CSR,SNR) are the left and
// right singular vectors for abs(SSMAX), giving the decomposition
//
//    [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
//    [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ].

import { dlamch } from "./dlamch.js";
import { MACH_EPS } from "../utils/constants.js";

/** Fortran SIGN(A,B): returns |A| with the sign of B. Treats B=0 as positive. */
function fsign(a: number, b: number): number {
  return b >= 0 ? Math.abs(a) : -Math.abs(a);
}

export function dlasv2(
  f: number,
  g: number,
  h: number
): {
  ssmin: number;
  ssmax: number;
  snr: number;
  csr: number;
  snl: number;
  csl: number;
} {
  const ZERO = 0.0;
  const HALF = 0.5;
  const ONE = 1.0;
  const TWO = 2.0;
  const FOUR = 4.0;

  let ft = f;
  let fa = Math.abs(ft);
  let ht = h;
  let ha = Math.abs(h);

  // PMAX points to the maximum absolute element of matrix
  //   PMAX = 1 if F largest in absolute values
  //   PMAX = 2 if G largest in absolute values
  //   PMAX = 3 if H largest in absolute values
  let pmax = 1;
  const swap = ha > fa;
  if (swap) {
    pmax = 3;
    let temp = ft;
    ft = ht;
    ht = temp;
    temp = fa;
    fa = ha;
    ha = temp;

    // Now FA >= HA
  }

  const gt = g;
  const ga = Math.abs(gt);

  let clt: number;
  let slt: number;
  let crt: number;
  let srt: number;
  let ssmin: number;
  let ssmax: number;

  if (ga === ZERO) {
    // Diagonal matrix
    ssmin = ha;
    ssmax = fa;
    clt = ONE;
    crt = ONE;
    slt = ZERO;
    srt = ZERO;
  } else {
    let gasmal = true;
    if (ga > fa) {
      pmax = 2;
      if (fa / ga < dlamch(MACH_EPS)) {
        // Case of very large GA
        gasmal = false;
        ssmax = ga;
        if (ha > ONE) {
          ssmin = fa / (ga / ha);
        } else {
          ssmin = (fa / ga) * ha;
        }
        clt = ONE;
        slt = ht / gt;
        srt = ONE;
        crt = ft / gt;
      }
    }
    if (gasmal) {
      // Normal case
      const d = fa - ha;
      let l: number;
      if (d === fa) {
        // Copes with infinite F or H
        l = ONE;
      } else {
        l = d / fa;
      }

      // Note that 0 <= L <= 1
      const m = gt / ft;

      // Note that abs(M) <= 1/macheps
      let t = TWO - l;

      // Note that T >= 1
      const mm = m * m;
      const tt = t * t;
      const s = Math.sqrt(tt + mm);

      // Note that 1 <= S <= 1 + 1/macheps
      let r: number;
      if (l === ZERO) {
        r = Math.abs(m);
      } else {
        r = Math.sqrt(l * l + mm);
      }

      // Note that 0 <= R <= 1 + 1/macheps
      const a = HALF * (s + r);

      // Note that 1 <= A <= 1 + abs(M)
      ssmin = ha / a;
      ssmax = fa * a;

      if (mm === ZERO) {
        // Note that M is very tiny
        if (l === ZERO) {
          t = fsign(TWO, ft) * fsign(ONE, gt);
        } else {
          t = gt / fsign(d, ft) + m / t;
        }
      } else {
        t = (m / (s + t) + m / (r + l)) * (ONE + a);
      }
      l = Math.sqrt(t * t + FOUR);
      crt = TWO / l;
      srt = t / l;
      clt = (crt + srt * m) / a;
      slt = ((ht / ft) * srt) / a;
    }
  }

  let csl: number;
  let snl: number;
  let csr: number;
  let snr: number;

  if (swap) {
    csl = srt!;
    snl = crt!;
    csr = slt!;
    snr = clt!;
  } else {
    csl = clt!;
    snl = slt!;
    csr = crt!;
    snr = srt!;
  }

  // Correct signs of SSMAX and SSMIN
  let tsign: number = 0;
  if (pmax === 1) {
    tsign = fsign(ONE, csr) * fsign(ONE, csl) * fsign(ONE, f);
  }
  if (pmax === 2) {
    tsign = fsign(ONE, snr) * fsign(ONE, csl) * fsign(ONE, g);
  }
  if (pmax === 3) {
    tsign = fsign(ONE, snr) * fsign(ONE, snl) * fsign(ONE, h);
  }
  ssmax = fsign(ssmax!, tsign);
  ssmin = fsign(ssmin!, tsign * fsign(ONE, f) * fsign(ONE, h));

  return { ssmin, ssmax, snr, csr, snl, csl };
}
