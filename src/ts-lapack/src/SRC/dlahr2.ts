// Translated from SRC/dlahr2.f
// DLAHR2 reduces the first NB columns of a real general n-by-(n-k+1) matrix A
// so that elements below the k-th subdiagonal are zero. The reduction is
// performed by an orthogonal similarity transformation Q**T * A * Q. The
// routine returns the matrices V and T which determine Q as a block reflector
// I - V*T*V**T, and also the matrix Y = A * V * T.
//
// This is an auxiliary routine called by DGEHRD.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   TAU(I)  =>  tau[tauOff + (I-1)]              (I is 1-based)
//   T(I,J)  =>  t[tOff + (I-1) + (J-1)*ldt]     (I,J are 1-based)
//   Y(I,J)  =>  y[yOff + (I-1) + (J-1)*ldy]     (I,J are 1-based)
//
// Parameters:
//   n      - order of the matrix A (n > 0)
//   k      - offset for reduction; elements below the k-th subdiagonal in
//            the first NB columns are reduced to zero (k < n)
//   nb     - number of columns to be reduced
//   a      - Float64Array; n-by-(n-k+1) general matrix
//   aOff   - offset into a for A(1,1)
//   lda    - leading dimension of a (>= max(1,n))
//   tau    - Float64Array of length nb; scalar factors of reflectors
//   tauOff - offset into tau for TAU(1)
//   t      - Float64Array; ldt-by-nb upper triangular factor
//   tOff   - offset into t for T(1,1)
//   ldt    - leading dimension of t (>= nb)
//   y      - Float64Array; n-by-nb matrix Y
//   yOff   - offset into y for Y(1,1)
//   ldy    - leading dimension of y (>= n)

import { dgemv } from "../BLAS/dgemv.js";
import { dcopy } from "../BLAS/dcopy.js";
import { dlarfg } from "./dlarfg.js";
import { dlacpy } from "./dlacpy.js";
import { daxpy } from "../BLAS/daxpy.js";
import { dgemm } from "../BLAS/dgemm.js";
import { dtrmm } from "../BLAS/dtrmm.js";
import { dscal } from "../BLAS/dscal.js";
import { dtrmv } from "../BLAS/dtrmv.js";
import {
  NOTRANS,
  TRANS,
  UPPER,
  LOWER,
  UNIT,
  NONUNIT,
  // LEFT,
  RIGHT,
} from "../utils/constants.js";

const ZERO = 0.0;
const ONE = 1.0;

export function dlahr2(
  n: number,
  k: number,
  nb: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number,
  t: Float64Array,
  tOff: number,
  ldt: number,
  y: Float64Array,
  yOff: number,
  ldy: number
): void {
  // Quick return if possible
  if (n <= 1) return;

  let ei = 0.0;

  for (let i = 1; i <= nb; i++) {
    if (i > 1) {
      // Update A(K+1:N,I)

      // Update I-th column of A - Y * V**T
      // Fortran: DGEMV('NO TRANSPOSE', N-K, I-1, -ONE, Y(K+1,1), LDY,
      //                  A(K+I-1,1), LDA, ONE, A(K+1,I), 1)
      dgemv(
        NOTRANS,
        n - k,
        i - 1,
        -ONE,
        y,
        yOff + k + 0 * ldy, // Y(K+1,1)
        ldy,
        a,
        aOff + (k + i - 2) + 0 * lda, // A(K+I-1,1)
        lda,
        ONE,
        a,
        aOff + k + (i - 1) * lda, // A(K+1,I)
        1
      );

      // Apply I - V * T**T * V**T to this column (call it b) from the left,
      // using the last column of T as workspace
      //
      // Let V = ( V1 ) and b = ( b1 )   (first I-1 rows)
      //         ( V2 )         ( b2 )
      // where V1 is unit lower triangular

      // w := V1**T * b1
      // Fortran: DCOPY(I-1, A(K+1,I), 1, T(1,NB), 1)
      dcopy(
        i - 1,
        a,
        aOff + k + (i - 1) * lda, // A(K+1,I)
        1,
        t,
        tOff + (nb - 1) * ldt, // T(1,NB)
        1
      );
      // Fortran: DTRMV('Lower', 'Transpose', 'UNIT', I-1, A(K+1,1), LDA, T(1,NB), 1)
      dtrmv(
        LOWER,
        TRANS,
        UNIT,
        i - 1,
        a,
        aOff + k + 0 * lda, // A(K+1,1)
        lda,
        t,
        tOff + (nb - 1) * ldt, // T(1,NB)
        1
      );

      // w := w + V2**T * b2
      // Fortran: DGEMV('Transpose', N-K-I+1, I-1, ONE, A(K+I,1), LDA,
      //                 A(K+I,I), 1, ONE, T(1,NB), 1)
      dgemv(
        TRANS,
        n - k - i + 1,
        i - 1,
        ONE,
        a,
        aOff + (k + i - 1) + 0 * lda, // A(K+I,1)
        lda,
        a,
        aOff + (k + i - 1) + (i - 1) * lda, // A(K+I,I)
        1,
        ONE,
        t,
        tOff + (nb - 1) * ldt, // T(1,NB)
        1
      );

      // w := T**T * w
      // Fortran: DTRMV('Upper', 'Transpose', 'NON-UNIT', I-1, T, LDT, T(1,NB), 1)
      dtrmv(
        UPPER,
        TRANS,
        NONUNIT,
        i - 1,
        t,
        tOff, // T(1,1)
        ldt,
        t,
        tOff + (nb - 1) * ldt, // T(1,NB)
        1
      );

      // b2 := b2 - V2*w
      // Fortran: DGEMV('NO TRANSPOSE', N-K-I+1, I-1, -ONE, A(K+I,1), LDA,
      //                 T(1,NB), 1, ONE, A(K+I,I), 1)
      dgemv(
        NOTRANS,
        n - k - i + 1,
        i - 1,
        -ONE,
        a,
        aOff + (k + i - 1) + 0 * lda, // A(K+I,1)
        lda,
        t,
        tOff + (nb - 1) * ldt, // T(1,NB)
        1,
        ONE,
        a,
        aOff + (k + i - 1) + (i - 1) * lda, // A(K+I,I)
        1
      );

      // b1 := b1 - V1*w
      // Fortran: DTRMV('Lower', 'NO TRANSPOSE', 'UNIT', I-1, A(K+1,1), LDA, T(1,NB), 1)
      dtrmv(
        LOWER,
        NOTRANS,
        UNIT,
        i - 1,
        a,
        aOff + k + 0 * lda, // A(K+1,1)
        lda,
        t,
        tOff + (nb - 1) * ldt, // T(1,NB)
        1
      );
      // Fortran: DAXPY(I-1, -ONE, T(1,NB), 1, A(K+1,I), 1)
      daxpy(
        i - 1,
        -ONE,
        t,
        tOff + (nb - 1) * ldt, // T(1,NB)
        1,
        a,
        aOff + k + (i - 1) * lda, // A(K+1,I)
        1
      );

      // Fortran: A(K+I-1,I-1) = EI
      a[aOff + (k + i - 2) + (i - 2) * lda] = ei;
    }

    // Generate the elementary reflector H(I) to annihilate A(K+I+1:N,I)
    //
    // Fortran: DLARFG(N-K-I+1, A(K+I,I), A(MIN(K+I+1,N),I), 1, TAU(I))
    const alphaIdx = aOff + (k + i - 1) + (i - 1) * lda; // A(K+I,I)
    const xIdx = aOff + (Math.min(k + i + 1, n) - 1) + (i - 1) * lda; // A(MIN(K+I+1,N),I)
    const result = dlarfg(n - k - i + 1, a[alphaIdx], a, xIdx, 1);
    a[alphaIdx] = result.alpha;
    tau[tauOff + (i - 1)] = result.tau;

    // Fortran: EI = A(K+I,I)
    ei = a[alphaIdx];
    // Fortran: A(K+I,I) = ONE
    a[alphaIdx] = ONE;

    // Compute Y(K+1:N,I)
    //
    // Fortran: DGEMV('NO TRANSPOSE', N-K, N-K-I+1, ONE, A(K+1,I+1), LDA,
    //                 A(K+I,I), 1, ZERO, Y(K+1,I), 1)
    dgemv(
      NOTRANS,
      n - k,
      n - k - i + 1,
      ONE,
      a,
      aOff + k + i * lda, // A(K+1,I+1)
      lda,
      a,
      aOff + (k + i - 1) + (i - 1) * lda, // A(K+I,I)
      1,
      ZERO,
      y,
      yOff + k + (i - 1) * ldy, // Y(K+1,I)
      1
    );

    // Fortran: DGEMV('Transpose', N-K-I+1, I-1, ONE, A(K+I,1), LDA,
    //                 A(K+I,I), 1, ZERO, T(1,I), 1)
    dgemv(
      TRANS,
      n - k - i + 1,
      i - 1,
      ONE,
      a,
      aOff + (k + i - 1) + 0 * lda, // A(K+I,1)
      lda,
      a,
      aOff + (k + i - 1) + (i - 1) * lda, // A(K+I,I)
      1,
      ZERO,
      t,
      tOff + (i - 1) * ldt, // T(1,I)
      1
    );

    // Fortran: DGEMV('NO TRANSPOSE', N-K, I-1, -ONE, Y(K+1,1), LDY,
    //                 T(1,I), 1, ONE, Y(K+1,I), 1)
    dgemv(
      NOTRANS,
      n - k,
      i - 1,
      -ONE,
      y,
      yOff + k + 0 * ldy, // Y(K+1,1)
      ldy,
      t,
      tOff + (i - 1) * ldt, // T(1,I)
      1,
      ONE,
      y,
      yOff + k + (i - 1) * ldy, // Y(K+1,I)
      1
    );

    // Fortran: DSCAL(N-K, TAU(I), Y(K+1,I), 1)
    dscal(
      n - k,
      tau[tauOff + (i - 1)],
      y,
      yOff + k + (i - 1) * ldy, // Y(K+1,I)
      1
    );

    // Compute T(1:I,I)
    //
    // Fortran: DSCAL(I-1, -TAU(I), T(1,I), 1)
    dscal(
      i - 1,
      -tau[tauOff + (i - 1)],
      t,
      tOff + (i - 1) * ldt, // T(1,I)
      1
    );

    // Fortran: DTRMV('Upper', 'No Transpose', 'NON-UNIT', I-1, T, LDT, T(1,I), 1)
    dtrmv(
      UPPER,
      NOTRANS,
      NONUNIT,
      i - 1,
      t,
      tOff, // T(1,1)
      ldt,
      t,
      tOff + (i - 1) * ldt, // T(1,I)
      1
    );

    // Fortran: T(I,I) = TAU(I)
    t[tOff + (i - 1) + (i - 1) * ldt] = tau[tauOff + (i - 1)];
  }

  // Fortran: A(K+NB,NB) = EI
  a[aOff + (k + nb - 1) + (nb - 1) * lda] = ei;

  // Compute Y(1:K,1:NB)
  //
  // Fortran: DLACPY('ALL', K, NB, A(1,2), LDA, Y, LDY)
  dlacpy(
    -1, // ALL
    k,
    nb,
    a,
    aOff + 1 * lda, // A(1,2)
    lda,
    y,
    yOff, // Y(1,1)
    ldy
  );

  // Fortran: DTRMM('RIGHT', 'Lower', 'NO TRANSPOSE', 'UNIT', K, NB,
  //                 ONE, A(K+1,1), LDA, Y, LDY)
  dtrmm(
    RIGHT,
    LOWER,
    NOTRANS,
    UNIT,
    k,
    nb,
    ONE,
    a,
    aOff + k + 0 * lda, // A(K+1,1)
    lda,
    y,
    yOff, // Y(1,1)
    ldy
  );

  if (n > k + nb) {
    // Fortran: DGEMM('NO TRANSPOSE', 'NO TRANSPOSE', K, NB, N-K-NB, ONE,
    //                 A(1,2+NB), LDA, A(K+1+NB,1), LDA, ONE, Y, LDY)
    dgemm(
      NOTRANS,
      NOTRANS,
      k,
      nb,
      n - k - nb,
      ONE,
      a,
      aOff + (1 + nb) * lda, // A(1,2+NB)
      lda,
      a,
      aOff + (k + nb) + 0 * lda, // A(K+1+NB,1)
      lda,
      ONE,
      y,
      yOff, // Y(1,1)
      ldy
    );
  }

  // Fortran: DTRMM('RIGHT', 'Upper', 'NO TRANSPOSE', 'NON-UNIT', K, NB,
  //                 ONE, T, LDT, Y, LDY)
  dtrmm(
    RIGHT,
    UPPER,
    NOTRANS,
    NONUNIT,
    k,
    nb,
    ONE,
    t,
    tOff, // T(1,1)
    ldt,
    y,
    yOff, // Y(1,1)
    ldy
  );
}
