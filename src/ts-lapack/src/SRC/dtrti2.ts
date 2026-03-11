// Translated from SRC/dtrti2.f
// DTRTI2 computes the inverse of a real upper or lower triangular matrix
// (unblocked Level 2 BLAS algorithm).
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//
// Returns INFO:
//   0   => successful exit
//   < 0 => -INFO-th argument had an illegal value (thrown as error)

import { dscal } from "../BLAS/dscal.js";
import { dtrmv } from "../BLAS/dtrmv.js";
import { xerbla } from "../utils/xerbla.js";
import { UPPER, LOWER, NOTRANS, UNIT, NONUNIT } from "../utils/constants.js";

export function dtrti2(
  uplo: number,
  diag: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  const upper = uplo === UPPER;
  const nounit = diag === NONUNIT;

  let info = 0;
  if (uplo !== UPPER && uplo !== LOWER) {
    info = -1;
  } else if (diag !== NONUNIT && diag !== UNIT) {
    info = -2;
  } else if (n < 0) {
    info = -3;
  } else if (lda < Math.max(1, n)) {
    info = -5;
  }
  if (info !== 0) {
    xerbla("DTRTI2", -info);
    return info;
  }

  if (upper) {
    // Compute inverse of upper triangular matrix
    for (let j = 1; j <= n; j++) {
      let ajj: number;
      if (nounit) {
        // A(J,J) = ONE / A(J,J)
        a[aOff + (j - 1) + (j - 1) * lda] =
          1.0 / a[aOff + (j - 1) + (j - 1) * lda];
        ajj = -a[aOff + (j - 1) + (j - 1) * lda];
      } else {
        ajj = -1.0;
      }

      // Compute elements 1:j-1 of j-th column
      // DTRMV('Upper','No transpose', DIAG, J-1, A, LDA, A(1,J), 1)
      // A(1,J) is at aOff + 0 + (j-1)*lda
      if (j - 1 >= 1) {
        dtrmv(
          UPPER,
          NOTRANS,
          diag,
          j - 1,
          a,
          aOff,
          lda,
          a,
          aOff + (j - 1) * lda,
          1
        );
        // DSCAL(J-1, AJJ, A(1,J), 1)
        dscal(j - 1, ajj, a, aOff + (j - 1) * lda, 1);
      }
    }
  } else {
    // Compute inverse of lower triangular matrix
    for (let j = n; j >= 1; j--) {
      let ajj: number;
      if (nounit) {
        a[aOff + (j - 1) + (j - 1) * lda] =
          1.0 / a[aOff + (j - 1) + (j - 1) * lda];
        ajj = -a[aOff + (j - 1) + (j - 1) * lda];
      } else {
        ajj = -1.0;
      }

      if (j < n) {
        // Compute elements j+1:n of j-th column
        // DTRMV('Lower','No transpose', DIAG, N-J, A(J+1,J+1), LDA, A(J+1,J), 1)
        // A(J+1,J+1) => aOff + j + j*lda
        // A(J+1,J)   => aOff + j + (j-1)*lda
        dtrmv(
          LOWER,
          NOTRANS,
          diag,
          n - j,
          a,
          aOff + j + j * lda,
          lda,
          a,
          aOff + j + (j - 1) * lda,
          1
        );
        // DSCAL(N-J, AJJ, A(J+1,J), 1)
        dscal(n - j, ajj, a, aOff + j + (j - 1) * lda, 1);
      }
    }
  }

  return 0;
}
