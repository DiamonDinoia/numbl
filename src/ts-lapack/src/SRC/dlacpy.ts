// Translated from SRC/dlacpy.f
// DLACPY copies all or part of a two-dimensional matrix A to another
// matrix B.

import { UPPER, LOWER } from "../utils/constants.js";

/**
 * DLACPY copies all or part of a two-dimensional matrix A to another
 * matrix B.
 *
 * @param uplo - Specifies the part of matrix A to copy:
 *               UPPER (0) = copy upper triangular part,
 *               LOWER (1) = copy lower triangular part,
 *               anything else = copy the entire matrix.
 * @param m - Number of rows of the matrix A. m >= 0.
 * @param n - Number of columns of the matrix A. n >= 0.
 * @param a - Source matrix, dimension (lda, n).
 * @param aOff - Offset into array a.
 * @param lda - Leading dimension of a. lda >= max(1, m).
 * @param b - Destination matrix, dimension (ldb, n).
 * @param bOff - Offset into array b.
 * @param ldb - Leading dimension of b. ldb >= max(1, m).
 */
export function dlacpy(
  uplo: number,
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number
): void {
  if (uplo === UPPER) {
    // Copy upper triangular part
    for (let j = 1; j <= n; j++) {
      for (let i = 1; i <= Math.min(j, m); i++) {
        // B(I,J) = A(I,J)
        b[bOff + (i - 1) + (j - 1) * ldb] = a[aOff + (i - 1) + (j - 1) * lda];
      }
    }
  } else if (uplo === LOWER) {
    // Copy lower triangular part
    for (let j = 1; j <= n; j++) {
      for (let i = j; i <= m; i++) {
        // B(I,J) = A(I,J)
        b[bOff + (i - 1) + (j - 1) * ldb] = a[aOff + (i - 1) + (j - 1) * lda];
      }
    }
  } else {
    // Copy the entire matrix
    for (let j = 1; j <= n; j++) {
      for (let i = 1; i <= m; i++) {
        // B(I,J) = A(I,J)
        b[bOff + (i - 1) + (j - 1) * ldb] = a[aOff + (i - 1) + (j - 1) * lda];
      }
    }
  }
}
