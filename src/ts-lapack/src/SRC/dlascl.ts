// Translated from SRC/dlascl.f
// DLASCL multiplies the M by N real matrix A by the real scalar CTO/CFROM.
// This is done without over/underflow as long as the final result
// CTO*A(I,J)/CFROM does not over/underflow.
//
// type codes:
//   0 = 'G' — general (full) matrix
//   1 = 'L' — lower triangular matrix
//   2 = 'U' — upper triangular matrix
//   3 = 'H' — upper Hessenberg matrix
//   4 = 'B' — lower half of symmetric band matrix (lower bandwidth KL)
//   5 = 'Q' — upper half of symmetric band matrix (upper bandwidth KU)
//   6 = 'Z' — band matrix (lower bandwidth KL, upper bandwidth KU)
//
// Array indexing convention (column-major):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//
// Returns INFO: 0 = success, <0 = -i means the i-th argument had an illegal value.

import { dlamch } from "./dlamch.js";
import { MACH_SFMIN } from "../utils/constants.js";

// type constants
const TYPE_G = 0; // 'G' general
const TYPE_L = 1; // 'L' lower triangular
const TYPE_U = 2; // 'U' upper triangular
const TYPE_H = 3; // 'H' upper Hessenberg
const TYPE_B = 4; // 'B' lower half of symmetric band
const TYPE_Q = 5; // 'Q' upper half of symmetric band
const TYPE_Z = 6; // 'Z' band matrix

export function dlascl(
  type: number,
  kl: number,
  ku: number,
  cfrom: number,
  cto: number,
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  let info = 0;

  // Validate the type argument
  const itype = type;
  if (
    itype !== TYPE_G &&
    itype !== TYPE_L &&
    itype !== TYPE_U &&
    itype !== TYPE_H &&
    itype !== TYPE_B &&
    itype !== TYPE_Q &&
    itype !== TYPE_Z
  ) {
    info = -1;
  } else if (cfrom === 0.0 || isNaN(cfrom)) {
    info = -4;
  } else if (isNaN(cto)) {
    info = -5;
  } else if (m < 0) {
    info = -6;
  } else if (
    n < 0 ||
    (itype === TYPE_B && n !== m) ||
    (itype === TYPE_Q && n !== m)
  ) {
    info = -7;
  } else if (itype <= TYPE_H && lda < Math.max(1, m)) {
    info = -9;
  } else if (itype >= TYPE_B) {
    if (kl < 0 || kl > Math.max(m - 1, 0)) {
      info = -2;
    } else if (
      ku < 0 ||
      ku > Math.max(n - 1, 0) ||
      ((itype === TYPE_B || itype === TYPE_Q) && kl !== ku)
    ) {
      info = -3;
    } else if (
      (itype === TYPE_B && lda < kl + 1) ||
      (itype === TYPE_Q && lda < ku + 1) ||
      (itype === TYPE_Z && lda < 2 * kl + ku + 1)
    ) {
      info = -9;
    }
  }

  if (info !== 0) {
    return info;
  }

  // Quick return if possible
  if (n === 0 || m === 0) return 0;

  // Get machine parameters
  const smlnum = dlamch(MACH_SFMIN);
  const bignum = 1.0 / smlnum;

  let cfromc = cfrom;
  let ctoc = cto;

  let done = false;
  while (!done) {
    const cfrom1 = cfromc * smlnum;
    let mul: number;
    let cto1: number;

    if (cfrom1 === cfromc) {
      // CFROMC is an inf. Multiply by a correctly signed zero for
      // finite CTOC, or a NaN if CTOC is infinite.
      mul = ctoc / cfromc;
      done = true;
      cto1 = ctoc;
    } else {
      cto1 = ctoc / bignum;
      if (cto1 === ctoc) {
        // CTOC is either 0 or an inf. In both cases, CTOC itself
        // serves as the correct multiplication factor.
        mul = ctoc;
        done = true;
        cfromc = 1.0;
      } else if (Math.abs(cfrom1) > Math.abs(ctoc) && ctoc !== 0.0) {
        mul = smlnum;
        done = false;
        cfromc = cfrom1;
      } else if (Math.abs(cto1) > Math.abs(cfromc)) {
        mul = bignum;
        done = false;
        ctoc = cto1;
      } else {
        mul = ctoc / cfromc;
        done = true;
        if (mul === 1.0) return 0;
      }
    }

    if (itype === TYPE_G) {
      // Full matrix
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= m; i++) {
          a[aOff + (i - 1) + (j - 1) * lda] *= mul;
        }
      }
    } else if (itype === TYPE_L) {
      // Lower triangular matrix
      for (let j = 1; j <= n; j++) {
        for (let i = j; i <= m; i++) {
          a[aOff + (i - 1) + (j - 1) * lda] *= mul;
        }
      }
    } else if (itype === TYPE_U) {
      // Upper triangular matrix
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= Math.min(j, m); i++) {
          a[aOff + (i - 1) + (j - 1) * lda] *= mul;
        }
      }
    } else if (itype === TYPE_H) {
      // Upper Hessenberg matrix
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= Math.min(j + 1, m); i++) {
          a[aOff + (i - 1) + (j - 1) * lda] *= mul;
        }
      }
    } else if (itype === TYPE_B) {
      // Lower half of a symmetric band matrix
      const k3 = kl + 1;
      const k4 = n + 1;
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= Math.min(k3, k4 - j); i++) {
          a[aOff + (i - 1) + (j - 1) * lda] *= mul;
        }
      }
    } else if (itype === TYPE_Q) {
      // Upper half of a symmetric band matrix
      const k1 = ku + 2;
      const k3 = ku + 1;
      for (let j = 1; j <= n; j++) {
        for (let i = Math.max(k1 - j, 1); i <= k3; i++) {
          a[aOff + (i - 1) + (j - 1) * lda] *= mul;
        }
      }
    } else if (itype === TYPE_Z) {
      // Band matrix
      const k1 = kl + ku + 2;
      const k2 = kl + 1;
      const k3 = 2 * kl + ku + 1;
      const k4 = kl + ku + 1 + m;
      for (let j = 1; j <= n; j++) {
        for (let i = Math.max(k1 - j, k2); i <= Math.min(k3, k4 - j); i++) {
          a[aOff + (i - 1) + (j - 1) * lda] *= mul;
        }
      }
    }
  }

  return 0;
}
