// Translated from SRC/dlange.f
// DLANGE returns the value of the 1-norm, Frobenius norm, infinity-norm,
// or the largest absolute value of any element of a general rectangular matrix.
//
// norm codes:
//   0 = 'M' — max(abs(A(i,j)))
//   1 = 'O' or '1' — one-norm (maximum column sum)
//   2 = 'I' — infinity-norm (maximum row sum)
//   3 = 'F' or 'E' — Frobenius norm (sqrt of sum of squares)
//
// Array indexing convention (column-major):
//   A(I,J)   => a[aOff + (I-1) + (J-1)*lda]     (I,J are 1-based)
//   WORK(I)  => work[workOff + (I-1)]             (I is 1-based)
//
// WORK is only referenced when norm = 2 ('I'), and must have length >= M.

import { dlassq } from "./dlassq.js";

// norm constants
const NORM_MAX = 0; // 'M'
const NORM_ONE = 1; // 'O' or '1'
const NORM_INF = 2; // 'I'
const NORM_FRO = 3; // 'F' or 'E'

export function dlange(
  norm: number,
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  work: Float64Array,
  workOff: number
): number {
  let value: number;

  if (Math.min(m, n) === 0) {
    value = 0.0;
  } else if (norm === NORM_MAX) {
    // Find max(abs(A(i,j))).
    value = 0.0;
    for (let j = 1; j <= n; j++) {
      for (let i = 1; i <= m; i++) {
        const temp = Math.abs(a[aOff + (i - 1) + (j - 1) * lda]);
        if (value < temp || isNaN(temp)) value = temp;
      }
    }
  } else if (norm === NORM_ONE) {
    // Find norm1(A) — maximum column sum.
    value = 0.0;
    for (let j = 1; j <= n; j++) {
      let sum = 0.0;
      for (let i = 1; i <= m; i++) {
        sum = sum + Math.abs(a[aOff + (i - 1) + (j - 1) * lda]);
      }
      if (value < sum || isNaN(sum)) value = sum;
    }
  } else if (norm === NORM_INF) {
    // Find normI(A) — maximum row sum.
    for (let i = 1; i <= m; i++) {
      work[workOff + (i - 1)] = 0.0;
    }
    for (let j = 1; j <= n; j++) {
      for (let i = 1; i <= m; i++) {
        work[workOff + (i - 1)] =
          work[workOff + (i - 1)] + Math.abs(a[aOff + (i - 1) + (j - 1) * lda]);
      }
    }
    value = 0.0;
    for (let i = 1; i <= m; i++) {
      const temp = work[workOff + (i - 1)];
      if (value < temp || isNaN(temp)) value = temp;
    }
  } else if (norm === NORM_FRO) {
    // Find normF(A) — Frobenius norm.
    const scl = { val: 0.0 };
    const sumsq = { val: 1.0 };
    for (let j = 1; j <= n; j++) {
      // DLASSQ( M, A(1,J), 1, SCALE, SUM )
      // A(1,J) starts at aOff + (1-1) + (j-1)*lda = aOff + (j-1)*lda
      const colOff = aOff + (j - 1) * lda;
      dlassq(m, a, colOff, 1, scl, sumsq);
    }
    value = scl.val * Math.sqrt(sumsq.val);
  } else {
    value = 0.0;
  }

  return value;
}
