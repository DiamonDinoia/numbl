// Translated from SRC/dgetrf2.f
// DGETRF2 computes an LU factorization of a general M-by-N matrix A
// using partial pivoting with row interchanges (recursive version).
//
// The factorization has the form  A = P * L * U
// where P is a permutation matrix, L is lower triangular with unit
// diagonal elements, and U is upper triangular.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   IPIV(I)   =>  ipiv[ipivOff + (I-1)]           (I is 1-based)
//
// Returns INFO:
//   0  => successful
//   >0 => U(INFO,INFO) is exactly zero (singular)

import { dgemm } from "../BLAS/dgemm.js";
import { dscal } from "../BLAS/dscal.js";
import { idamax } from "../BLAS/idamax.js";
import { dlamch } from "./dlamch.js";
import { dlaswp } from "./dlaswp.js";
import { dtrsm } from "../BLAS/dtrsm.js";
import { xerbla } from "../utils/xerbla.js";
import { LEFT, LOWER, NOTRANS, UNIT, MACH_SFMIN } from "../utils/constants.js";

export function dgetrf2(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  ipiv: Int32Array,
  ipivOff: number
): number {
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
    xerbla("DGETRF2", -info);
    return info;
  }

  // Quick return if possible
  if (m === 0 || n === 0) return 0;

  if (m === 1) {
    // Use unblocked code for one row case
    ipiv[ipivOff] = 1; // IPIV(1) = 1
    if (a[aOff] === 0.0) info = 1; // A(1,1) == 0
  } else if (n === 1) {
    // Use unblocked code for one column case

    // Compute machine safe minimum
    const sfmin = dlamch(MACH_SFMIN);

    // Find pivot and test for singularity
    // IDAMAX(M, A(1,1), 1) => start at A(1,1) = aOff, incx=1
    const pivot = idamax(m, a, aOff, 1);
    ipiv[ipivOff] = pivot; // IPIV(1) = I  (1-based)

    // A(I,1) => a[aOff + (pivot-1)]
    if (a[aOff + (pivot - 1)] !== 0.0) {
      // Apply the interchange: swap A(1,1) and A(I,1)
      if (pivot !== 1) {
        const tmp = a[aOff];
        a[aOff] = a[aOff + (pivot - 1)];
        a[aOff + (pivot - 1)] = tmp;
      }

      // Compute elements 2:M of the column: A(2:M,1) /= A(1,1)
      if (Math.abs(a[aOff]) >= sfmin) {
        // DSCAL(M-1, ONE/A(1,1), A(2,1), 1)
        // A(2,1) is at offset aOff+1 (row 2, col 1, 0-based = row index 1)
        dscal(m - 1, 1.0 / a[aOff], a, aOff + 1, 1);
      } else {
        for (let i = 1; i <= m - 1; i++) {
          // A(1+I, 1) = A(1+I, 1) / A(1, 1)
          a[aOff + i] = a[aOff + i] / a[aOff];
        }
      }
    } else {
      info = 1;
    }
  } else {
    // Use recursive code
    const n1 = Math.floor(Math.min(m, n) / 2);
    const n2 = n - n1;

    //         [ A11 ]
    // Factor  [ --- ]
    //         [ A21 ]
    // DGETRF2(M, N1, A, LDA, IPIV, IINFO)
    let iinfo = dgetrf2(m, n1, a, aOff, lda, ipiv, ipivOff);
    if (info === 0 && iinfo > 0) info = iinfo;

    //                        [ A12 ]
    // Apply interchanges to  [ --- ]
    //                        [ A22 ]
    // DLASWP(N2, A(1,N1+1), LDA, 1, N1, IPIV, 1)
    // A(1, N1+1) => aOff + 0 + n1*lda
    dlaswp(n2, a, aOff + n1 * lda, lda, 1, n1, ipiv, ipivOff, 1);

    // Solve A12
    // DTRSM('L','L','N','U', N1, N2, ONE, A, LDA, A(1,N1+1), LDA)
    dtrsm(
      LEFT,
      LOWER,
      NOTRANS,
      UNIT,
      n1,
      n2,
      1.0,
      a,
      aOff,
      lda,
      a,
      aOff + n1 * lda,
      lda
    );

    // Update A22
    // DGEMM('N','N', M-N1, N2, N1, -ONE, A(N1+1,1), LDA,
    //        A(1,N1+1), LDA, ONE, A(N1+1,N1+1), LDA)
    // A(N1+1, 1)    => aOff + n1          (row N1+1, col 1)
    // A(1,    N1+1) => aOff + n1*lda      (row 1, col N1+1)
    // A(N1+1, N1+1) => aOff + n1 + n1*lda (row N1+1, col N1+1)
    dgemm(
      NOTRANS,
      NOTRANS,
      m - n1,
      n2,
      n1,
      -1.0,
      a,
      aOff + n1,
      lda,
      a,
      aOff + n1 * lda,
      lda,
      1.0,
      a,
      aOff + n1 + n1 * lda,
      lda
    );

    // Factor A22
    // DGETRF2(M-N1, N2, A(N1+1,N1+1), LDA, IPIV(N1+1), IINFO)
    iinfo = dgetrf2(
      m - n1,
      n2,
      a,
      aOff + n1 + n1 * lda,
      lda,
      ipiv,
      ipivOff + n1
    );

    // Adjust INFO and the pivot indices
    if (info === 0 && iinfo > 0) info = iinfo + n1;
    for (let i = n1 + 1; i <= Math.min(m, n); i++) {
      ipiv[ipivOff + (i - 1)] += n1;
    }

    // Apply interchanges to A21
    // DLASWP(N1, A(1,1), LDA, N1+1, MIN(M,N), IPIV, 1)
    dlaswp(n1, a, aOff, lda, n1 + 1, Math.min(m, n), ipiv, ipivOff, 1);
  }

  return info;
}
