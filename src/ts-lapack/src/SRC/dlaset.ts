// Translated from SRC/dlaset.f
// DLASET initializes an m-by-n matrix A to BETA on the diagonal and
// ALPHA on the offdiagonals.

import { UPPER, LOWER } from "../utils/constants.js";

/**
 * DLASET initializes an m-by-n matrix A to BETA on the diagonal and
 * ALPHA on the offdiagonals.
 *
 * @param uplo - Specifies the part of matrix A to be set:
 *               UPPER (0) = upper triangular part is set; the strictly
 *                           lower triangular part is not changed.
 *               LOWER (1) = lower triangular part is set; the strictly
 *                           upper triangular part is not changed.
 *               anything else = all of the matrix A is set.
 * @param m - Number of rows of the matrix A. m >= 0.
 * @param n - Number of columns of the matrix A. n >= 0.
 * @param alpha - The constant to which the offdiagonal elements are set.
 * @param beta - The constant to which the diagonal elements are set.
 * @param a - Matrix, dimension (lda, n).
 * @param aOff - Offset into array a.
 * @param lda - Leading dimension of a. lda >= max(1, m).
 */
export function dlaset(
  uplo: number,
  m: number,
  n: number,
  alpha: number,
  beta: number,
  a: Float64Array,
  aOff: number,
  lda: number
): void {
  if (uplo === UPPER) {
    // Set the strictly upper triangular or trapezoidal part of the
    // array to ALPHA.
    for (let j = 2; j <= n; j++) {
      for (let i = 1; i <= Math.min(j - 1, m); i++) {
        // A(I,J) = ALPHA
        a[aOff + (i - 1) + (j - 1) * lda] = alpha;
      }
    }
  } else if (uplo === LOWER) {
    // Set the strictly lower triangular or trapezoidal part of the
    // array to ALPHA.
    for (let j = 1; j <= Math.min(m, n); j++) {
      for (let i = j + 1; i <= m; i++) {
        // A(I,J) = ALPHA
        a[aOff + (i - 1) + (j - 1) * lda] = alpha;
      }
    }
  } else {
    // Set the leading m-by-n submatrix to ALPHA.
    for (let j = 1; j <= n; j++) {
      for (let i = 1; i <= m; i++) {
        // A(I,J) = ALPHA
        a[aOff + (i - 1) + (j - 1) * lda] = alpha;
      }
    }
  }

  // Set the first min(M,N) diagonal elements to BETA.
  for (let i = 1; i <= Math.min(m, n); i++) {
    // A(I,I) = BETA
    a[aOff + (i - 1) + (i - 1) * lda] = beta;
  }
}
