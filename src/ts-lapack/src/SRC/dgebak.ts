// Translated from SRC/dgebak.f
// DGEBAK forms the right or left eigenvectors of a real general matrix
// by backward transformation on the computed eigenvectors of the
// balanced matrix output by DGEBAL.
//
// JOB parameter (integer encoding):
//   0 = 'N' (none), 1 = 'P' (permute only),
//   2 = 'S' (scale only), 3 = 'B' (both)
//
// SIDE parameter (integer encoding, from constants.ts):
//   LEFT = 0 ('L'), RIGHT = 1 ('R')
//
// Indexing convention (matching Fortran column-major):
//   SCALE(I)  => scale[scaleOff + (I-1)]          (1-based I)
//   V(I,J)    => v[vOff + (I-1) + (J-1)*ldv]      (1-based I,J)

import { LEFT, RIGHT } from "../utils/constants.js";
import { dscal } from "../BLAS/dscal.js";
import { dswap } from "../BLAS/dswap.js";

const JOB_NONE = 0;
const JOB_PERMUTE = 1;
const JOB_SCALE = 2;
const JOB_BOTH = 3;

export function dgebak(
  job: number,
  side: number,
  n: number,
  ilo: number,
  ihi: number,
  scale: Float64Array,
  scaleOff: number,
  m: number,
  v: Float64Array,
  vOff: number,
  ldv: number
): number {
  // Decode and test the input parameters
  const rightv = side === RIGHT;
  const leftv = side === LEFT;

  let info = 0;
  if (job < 0 || job > 3) {
    info = -1;
  } else if (!rightv && !leftv) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (ilo < 1 || ilo > Math.max(1, n)) {
    info = -4;
  } else if (ihi < Math.min(ilo, n) || ihi > n) {
    info = -5;
  } else if (m < 0) {
    info = -7;
  } else if (ldv < Math.max(1, n)) {
    info = -9;
  }
  if (info !== 0) {
    return info;
  }

  // Quick return if possible
  if (n === 0) return 0;
  if (m === 0) return 0;
  if (job === JOB_NONE) return 0;

  // Backward balance (scaling)
  if (ilo !== ihi) {
    if (job === JOB_SCALE || job === JOB_BOTH) {
      if (rightv) {
        for (let i = ilo; i <= ihi; i++) {
          const s = scale[scaleOff + (i - 1)];
          // DSCAL(M, S, V(I,1), LDV) — scale row I across all M columns
          dscal(m, s, v, vOff + (i - 1), ldv);
        }
      }

      if (leftv) {
        for (let i = ilo; i <= ihi; i++) {
          const s = 1.0 / scale[scaleOff + (i - 1)];
          // DSCAL(M, S, V(I,1), LDV) — scale row I across all M columns
          dscal(m, s, v, vOff + (i - 1), ldv);
        }
      }
    }
  }

  // Backward permutation
  // For I = ILO-1 step -1 until 1,
  //         IHI+1 step 1 until N do --
  if (job === JOB_PERMUTE || job === JOB_BOTH) {
    if (rightv) {
      for (let ii = 1; ii <= n; ii++) {
        let i = ii;
        if (i >= ilo && i <= ihi) continue;
        if (i < ilo) i = ilo - ii;
        const k = Math.trunc(scale[scaleOff + (i - 1)]);
        if (k === i) continue;
        // DSWAP(M, V(I,1), LDV, V(K,1), LDV) — swap rows I and K
        dswap(m, v, vOff + (i - 1), ldv, v, vOff + (k - 1), ldv);
      }
    }

    if (leftv) {
      for (let ii = 1; ii <= n; ii++) {
        let i = ii;
        if (i >= ilo && i <= ihi) continue;
        if (i < ilo) i = ilo - ii;
        const k = Math.trunc(scale[scaleOff + (i - 1)]);
        if (k === i) continue;
        // DSWAP(M, V(I,1), LDV, V(K,1), LDV) — swap rows I and K
        dswap(m, v, vOff + (i - 1), ldv, v, vOff + (k - 1), ldv);
      }
    }
  }

  return 0;
}
