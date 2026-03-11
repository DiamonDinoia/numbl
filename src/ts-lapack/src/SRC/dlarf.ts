// Translated from SRC/dlarf.f
// DLARF applies a real elementary reflector H to a real m by n matrix
// C, from either the left or the right. H is represented in the form
//
//       H = I - tau * v * v**T
//
// where tau is a real scalar and v is a real vector.
//
// If tau = 0, then H is taken to be the unit matrix.
//
// Array indexing convention (column-major, matching Fortran):
//   C(I,J)  =>  c[cOff + (I-1) + (J-1)*ldc]   (I,J are 1-based)
//   V(I)    =>  v[vOff + (I-1)*incv]            (I is 1-based)
//   WORK(I) =>  work[workOff + (I-1)]           (I is 1-based)

import { LEFT } from "../utils/constants.js";
import { TRANS, NOTRANS } from "../utils/constants.js";
import { dgemv } from "../BLAS/dgemv.js";
import { dger } from "../BLAS/dger.js";

/**
 * DLARF applies a real elementary reflector H to a real m by n matrix C,
 * from either the left or the right.
 *
 * @param side   - LEFT (0) to form H*C, RIGHT (1) to form C*H
 * @param m      - Number of rows of C
 * @param n      - Number of columns of C
 * @param v      - The vector v in H = I - tau*v*v'
 * @param vOff   - Offset into v for V(1)
 * @param incv   - Increment between elements of v (incv <> 0)
 * @param tau    - The scalar tau
 * @param c      - The m-by-n matrix C (column-major)
 * @param cOff   - Offset into c for C(1,1)
 * @param ldc    - Leading dimension of c (ldc >= max(1,m))
 * @param work   - Workspace of dimension n (if side=LEFT) or m (if side=RIGHT)
 * @param workOff - Offset into work
 */
export function dlarf(
  side: number,
  m: number,
  n: number,
  v: Float64Array,
  vOff: number,
  incv: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number,
  work: Float64Array,
  workOff: number
): void {
  const applyleft = side === LEFT;
  let lastv = 0;
  let lastc = 0;

  if (tau !== 0.0) {
    // Set up variables for scanning V. LASTV begins pointing to the end of V.
    if (applyleft) {
      lastv = m;
    } else {
      lastv = n;
    }
    let i: number;
    if (incv > 0) {
      i = vOff + (lastv - 1) * incv;
    } else {
      i = vOff;
    }
    // Look for the last non-zero row in V.
    while (lastv > 0 && v[i] === 0.0) {
      lastv--;
      i -= incv;
    }
    if (applyleft) {
      // Scan for the last non-zero column in C(1:lastv,:).
      lastc = iladlc(lastv, n, c, cOff, ldc);
    } else {
      // Scan for the last non-zero row in C(:,1:lastv).
      lastc = iladlr(m, lastv, c, cOff, ldc);
    }
  }

  // Note that lastc==0 renders the BLAS operations null; no special
  // case is needed at this level.
  if (applyleft) {
    // Form  H * C
    if (lastv > 0) {
      // w(1:lastc,1) := C(1:lastv,1:lastc)**T * v(1:lastv,1)
      dgemv(
        TRANS,
        lastv,
        lastc,
        1.0,
        c,
        cOff,
        ldc,
        v,
        vOff,
        incv,
        0.0,
        work,
        workOff,
        1
      );

      // C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**T
      dger(lastv, lastc, -tau, v, vOff, incv, work, workOff, 1, c, cOff, ldc);
    }
  } else {
    // Form  C * H
    if (lastv > 0) {
      // w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
      dgemv(
        NOTRANS,
        lastc,
        lastv,
        1.0,
        c,
        cOff,
        ldc,
        v,
        vOff,
        incv,
        0.0,
        work,
        workOff,
        1
      );

      // C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**T
      dger(lastc, lastv, -tau, work, workOff, 1, v, vOff, incv, c, cOff, ldc);
    }
  }
}

// ILADLC scans matrix A for its last non-zero column.
// Returns the column index (1-based), or 0 if the matrix is all zero.
function iladlc(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  // Quick test for the common case where one corner is non-zero.
  if (n === 0) return 0;
  // A(1, N)
  if (
    a[aOff + (n - 1) * lda] !== 0.0 ||
    a[aOff + (m - 1) + (n - 1) * lda] !== 0.0
  ) {
    return n;
  }
  // Now scan each column from the right, looking for the first non-zero.
  for (let j = n; j >= 1; j--) {
    for (let i = 1; i <= m; i++) {
      if (a[aOff + (i - 1) + (j - 1) * lda] !== 0.0) {
        return j;
      }
    }
  }
  return 0;
}

// ILADLR scans matrix A for its last non-zero row.
// Returns the row index (1-based), or 0 if the matrix is all zero.
function iladlr(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  // Quick test for the common case where one corner is non-zero.
  if (m === 0) return 0;
  if (a[aOff + (m - 1)] !== 0.0 || a[aOff + (m - 1) + (n - 1) * lda] !== 0.0) {
    return m;
  }
  // Scan each row from the bottom, looking for the first non-zero.
  for (let i = m; i >= 1; i--) {
    for (let j = 1; j <= n; j++) {
      if (a[aOff + (i - 1) + (j - 1) * lda] !== 0.0) {
        return i;
      }
    }
  }
  return 0;
}
