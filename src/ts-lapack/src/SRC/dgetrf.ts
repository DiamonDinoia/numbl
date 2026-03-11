// Translated from SRC/dgetrf.f
// DGETRF computes an LU factorization of a general M-by-N matrix A
// using partial pivoting with row interchanges (right-looking Level 3 BLAS version).
//
// The factorization has the form  A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements (lower trapezoidal if m > n), and U is upper
// triangular (upper trapezoidal if m < n).
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)   =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   IPIV(I)  =>  ipiv[ipivOff + (I-1)]           (I is 1-based)
//
// Parameters:
//   m    - number of rows    (>= 0)
//   n    - number of columns (>= 0)
//   a    - Float64Array of length >= lda*n, stored column-major (modified in place)
//   lda  - leading dimension of a (>= max(1,m))
//   ipiv - Int32Array of length >= min(m,n); output pivot indices (1-based)
//
// Returns INFO:
//   0   => successful exit
//   < 0 => if INFO = -i, the i-th argument had an illegal value (thrown as error)
//   > 0 => U(INFO,INFO) is exactly zero (factorization complete but singular)

import { dgemm } from "../BLAS/dgemm.js";
import { dtrsm } from "../BLAS/dtrsm.js";
import { dlaswp } from "./dlaswp.js";
import { dgetrf2 } from "./dgetrf2.js";
import { xerbla } from "../utils/xerbla.js";
import { ilaenv } from "../utils/ilaenv.js";
import { LEFT, LOWER, NOTRANS, UNIT } from "../utils/constants.js";

export function dgetrf(
  m: number,
  n: number,
  a: Float64Array,
  lda: number,
  ipiv: Int32Array
): number {
  // The internal offset is always 0 for the top-level call.
  // This matches the Fortran: A is passed at its beginning.
  const aOff = 0;
  const ipivOff = 0;

  // Test the input parameters
  let info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, m)) {
    info = -4;
  }
  if (info !== 0) {
    xerbla("DGETRF", -info);
    return info;
  }

  // Quick return if possible
  if (m === 0 || n === 0) return 0;

  // Determine the block size for this environment
  const nb = ilaenv(1, "DGETRF", " ", m, n, -1, -1);

  if (nb <= 1 || nb >= Math.min(m, n)) {
    // Use unblocked code
    return dgetrf2(m, n, a, aOff, lda, ipiv, ipivOff);
  }

  // Use blocked code
  const mn = Math.min(m, n);
  for (let j = 1; j <= mn; j += nb) {
    const jb = Math.min(mn - j + 1, nb);

    // Factor diagonal and subdiagonal blocks and test for exact singularity.
    // DGETRF2(M-J+1, JB, A(J,J), LDA, IPIV(J), IINFO)
    // A(J,J) => aOff + (J-1) + (J-1)*lda
    const iinfo = dgetrf2(
      m - j + 1,
      jb,
      a,
      aOff + (j - 1) + (j - 1) * lda,
      lda,
      ipiv,
      ipivOff + (j - 1)
    );

    // Adjust INFO and the pivot indices
    if (info === 0 && iinfo > 0) info = iinfo + j - 1;
    for (let i = j; i <= Math.min(m, j + jb - 1); i++) {
      ipiv[ipivOff + (i - 1)] = j - 1 + ipiv[ipivOff + (i - 1)];
    }

    // Apply interchanges to columns 1:J-1
    // DLASWP(J-1, A, LDA, J, J+JB-1, IPIV, 1)
    dlaswp(j - 1, a, aOff, lda, j, j + jb - 1, ipiv, ipivOff, 1);

    if (j + jb <= n) {
      // Apply interchanges to columns J+JB:N
      // DLASWP(N-J-JB+1, A(1,J+JB), LDA, J, J+JB-1, IPIV, 1)
      // A(1, J+JB) => aOff + 0 + (J+JB-1)*lda
      dlaswp(
        n - j - jb + 1,
        a,
        aOff + (j + jb - 1) * lda,
        lda,
        j,
        j + jb - 1,
        ipiv,
        ipivOff,
        1
      );

      // Compute block row of U.
      // DTRSM('Left','Lower','No transpose','Unit', JB, N-J-JB+1, ONE,
      //        A(J,J), LDA, A(J,J+JB), LDA)
      // A(J, J)    => aOff + (J-1) + (J-1)*lda
      // A(J, J+JB) => aOff + (J-1) + (J+JB-1)*lda
      dtrsm(
        LEFT,
        LOWER,
        NOTRANS,
        UNIT,
        jb,
        n - j - jb + 1,
        1.0,
        a,
        aOff + (j - 1) + (j - 1) * lda,
        lda,
        a,
        aOff + (j - 1) + (j + jb - 1) * lda,
        lda
      );

      if (j + jb <= m) {
        // Update trailing submatrix.
        // DGEMM('No transpose','No transpose', M-J-JB+1, N-J-JB+1, JB,
        //        -ONE, A(J+JB,J), LDA, A(J,J+JB), LDA, ONE, A(J+JB,J+JB), LDA)
        // A(J+JB, J)    => aOff + (J+JB-1) + (J-1)*lda
        // A(J,    J+JB) => aOff + (J-1)    + (J+JB-1)*lda
        // A(J+JB, J+JB) => aOff + (J+JB-1) + (J+JB-1)*lda
        dgemm(
          NOTRANS,
          NOTRANS,
          m - j - jb + 1,
          n - j - jb + 1,
          jb,
          -1.0,
          a,
          aOff + (j + jb - 1) + (j - 1) * lda,
          lda,
          a,
          aOff + (j - 1) + (j + jb - 1) * lda,
          lda,
          1.0,
          a,
          aOff + (j + jb - 1) + (j + jb - 1) * lda,
          lda
        );
      }
    }
  }

  return info;
}
