// Translated from SRC/dtrtri.f
// DTRTRI computes the inverse of a real upper or lower triangular matrix A
// (Level 3 BLAS blocked algorithm).
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//
// Returns INFO:
//   0   => successful exit
//   < 0 => -INFO-th argument had an illegal value (thrown as error)
//   > 0 => A(INFO,INFO) is exactly zero (singular matrix)

import { dtrmm } from "../BLAS/dtrmm.js";
import { dtrsm } from "../BLAS/dtrsm.js";
import { xerbla } from "../utils/xerbla.js";
import { ilaenv } from "../utils/ilaenv.js";
import { dtrti2 } from "./dtrti2.js";
import {
  UPPER,
  LOWER,
  NOTRANS,
  UNIT,
  NONUNIT,
  LEFT,
  RIGHT,
} from "../utils/constants.js";

export function dtrtri(
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
    xerbla("DTRTRI", -info);
    return info;
  }

  if (n === 0) return 0;

  // Check for singularity if non-unit
  if (nounit) {
    for (let i = 1; i <= n; i++) {
      if (a[aOff + (i - 1) + (i - 1) * lda] === 0.0) return i;
    }
  }

  // Determine block size
  const nb = ilaenv(1, "DTRTRI", " ", n, -1, -1, -1);

  if (nb <= 1 || nb >= n) {
    // Use unblocked code
    return dtrti2(uplo, diag, n, a, aOff, lda);
  }

  // Use blocked code
  if (upper) {
    // Compute inverse of upper triangular matrix
    for (let j = 1; j <= n; j += nb) {
      const jb = Math.min(nb, n - j + 1);

      // Compute rows 1:j-1 of current block column
      // DTRMM('Left','Upper','No transpose', DIAG, J-1, JB, ONE, A, LDA, A(1,J), LDA)
      // A(1,J) => aOff + 0 + (j-1)*lda
      if (j - 1 >= 1) {
        dtrmm(
          LEFT,
          UPPER,
          NOTRANS,
          diag,
          j - 1,
          jb,
          1.0,
          a,
          aOff,
          lda,
          a,
          aOff + (j - 1) * lda,
          lda
        );
        // DTRSM('Right','Upper','No transpose', DIAG, J-1, JB, -ONE, A(J,J), LDA, A(1,J), LDA)
        // A(J,J) => aOff + (j-1) + (j-1)*lda
        dtrsm(
          RIGHT,
          UPPER,
          NOTRANS,
          diag,
          j - 1,
          jb,
          -1.0,
          a,
          aOff + (j - 1) + (j - 1) * lda,
          lda,
          a,
          aOff + (j - 1) * lda,
          lda
        );
      }

      // Compute inverse of current diagonal block
      // DTRTI2('Upper', DIAG, JB, A(J,J), LDA, INFO)
      const iinfo = dtrti2(
        UPPER,
        diag,
        jb,
        a,
        aOff + (j - 1) + (j - 1) * lda,
        lda
      );
      if (info === 0 && iinfo !== 0) info = iinfo;
    }
  } else {
    // Compute inverse of lower triangular matrix
    const nn = Math.floor((n - 1) / nb) * nb + 1;
    for (let j = nn; j >= 1; j -= nb) {
      const jb = Math.min(nb, n - j + 1);

      if (j + jb <= n) {
        // Compute rows j+jb:n of current block column
        // DTRMM('Left','Lower','No transpose', DIAG, N-J-JB+1, JB, ONE, A(J+JB,J+JB), LDA, A(J+JB,J), LDA)
        // A(J+JB, J+JB) => aOff + (j+jb-1) + (j+jb-1)*lda
        // A(J+JB, J)    => aOff + (j+jb-1) + (j-1)*lda
        dtrmm(
          LEFT,
          LOWER,
          NOTRANS,
          diag,
          n - j - jb + 1,
          jb,
          1.0,
          a,
          aOff + (j + jb - 1) + (j + jb - 1) * lda,
          lda,
          a,
          aOff + (j + jb - 1) + (j - 1) * lda,
          lda
        );
        // DTRSM('Right','Lower','No transpose', DIAG, N-J-JB+1, JB, -ONE, A(J,J), LDA, A(J+JB,J), LDA)
        // A(J,J)     => aOff + (j-1) + (j-1)*lda
        // A(J+JB, J) => aOff + (j+jb-1) + (j-1)*lda
        dtrsm(
          RIGHT,
          LOWER,
          NOTRANS,
          diag,
          n - j - jb + 1,
          jb,
          -1.0,
          a,
          aOff + (j - 1) + (j - 1) * lda,
          lda,
          a,
          aOff + (j + jb - 1) + (j - 1) * lda,
          lda
        );
      }

      // Compute inverse of current diagonal block
      const iinfo = dtrti2(
        LOWER,
        diag,
        jb,
        a,
        aOff + (j - 1) + (j - 1) * lda,
        lda
      );
      if (info === 0 && iinfo !== 0) info = iinfo;
    }
  }

  return info;
}
