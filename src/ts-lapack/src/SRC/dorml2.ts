// Translated from LAPACK/SRC/dorml2.f
// DORML2 overwrites the general real m by n matrix C with
//
//       Q * C  if SIDE = LEFT and TRANS = NOTRANS, or
//
//       Q**T * C  if SIDE = LEFT and TRANS = TRANS, or
//
//       C * Q  if SIDE = RIGHT and TRANS = NOTRANS, or
//
//       C * Q**T if SIDE = RIGHT and TRANS = TRANS,
//
// where Q is a real orthogonal matrix defined as the product of k
// elementary reflectors
//
//       Q = H(k) . . . H(2) H(1)
//
// as returned by DGELQF. Q is of order m if SIDE = LEFT and of order n
// if SIDE = RIGHT.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]
//   C(I,J) => c[cOff + (I-1) + (J-1)*ldc]
//   TAU(I) => tau[tauOff + (I-1)]
//
// Parameters:
//   side    - LEFT (0) or RIGHT (1)
//   trans   - NOTRANS (0) or TRANS (1)
//   m       - number of rows of C (m >= 0)
//   n       - number of columns of C (n >= 0)
//   k       - number of elementary reflectors (0 <= k <= nq)
//   a       - contains the reflector vectors as returned by dgelq2/dgelqf
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of A
//   tau     - scalar factors of the reflectors
//   tauOff  - offset into tau for TAU(1)
//   c       - the m-by-n matrix C; overwritten on exit
//   cOff    - offset into c for C(1,1)
//   ldc     - leading dimension of C
//   work    - workspace array
//   workOff - offset into work
//
// Returns INFO:
//   = 0: successful exit
//   < 0: if INFO = -i, the i-th argument had an illegal value

import { LEFT, RIGHT, NOTRANS, TRANS } from "../utils/constants.js";
import { dlarf1f } from "./dlarf1f.js";

export function dorml2(
  side: number,
  trans: number,
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number,
  c: Float64Array,
  cOff: number,
  ldc: number,
  work: Float64Array,
  workOff: number
): number {
  // Test the input arguments
  let info = 0;
  const left = side === LEFT;
  const notran = trans === NOTRANS;

  // NQ is the order of Q
  const nq = left ? m : n;

  if (side !== LEFT && side !== RIGHT) {
    info = -1;
  } else if (trans !== NOTRANS && trans !== TRANS) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Math.max(1, k)) {
    info = -7;
  } else if (ldc < Math.max(1, m)) {
    info = -10;
  }
  if (info !== 0) {
    return info;
  }

  // Quick return if possible
  if (m === 0 || n === 0 || k === 0) {
    return 0;
  }

  // Determine loop direction
  let i1: number, i2: number, i3: number;
  if ((left && notran) || (!left && !notran)) {
    i1 = 1;
    i2 = k;
    i3 = 1;
  } else {
    i1 = k;
    i2 = 1;
    i3 = -1;
  }

  let ni: number, mi: number, ic: number, jc: number;
  if (left) {
    ni = n;
    jc = 1;
    mi = 0; // will be set in loop
    ic = 0; // will be set in loop
  } else {
    mi = m;
    ic = 1;
    ni = 0; // will be set in loop
    jc = 0; // will be set in loop
  }

  for (let i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
    if (left) {
      // H(i) is applied to C(i:m,1:n)
      mi = m - i + 1;
      ic = i;
    } else {
      // H(i) is applied to C(1:m,i:n)
      ni = n - i + 1;
      jc = i;
    }

    // Apply H(i)
    // A(I,I) => a[aOff + (i-1) + (i-1)*lda]  -- reflector in row, stride LDA
    // TAU(I) => tau[tauOff + (i-1)]
    // C(IC,JC) => c[cOff + (ic-1) + (jc-1)*ldc]
    dlarf1f(
      side,
      mi,
      ni,
      a,
      aOff + (i - 1) + (i - 1) * lda,
      lda,
      tau[tauOff + (i - 1)],
      c,
      cOff + (ic - 1) + (jc - 1) * ldc,
      ldc,
      work,
      workOff
    );
  }

  return 0;
}
