// Translated from LAPACK/SRC/dlabrd.f
// DLABRD reduces the first NB rows and columns of a real general
// m by n matrix A to upper or lower bidiagonal form by an orthogonal
// transformation Q**T * A * P, and returns the matrices X and Y which
// are needed to apply the transformation to the unreduced part of A.
//
// If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
// bidiagonal form.
//
// This is an auxiliary routine called by DGEBRD.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]      (I,J are 1-based)
//   D(I)      =>  d[dOff + (I-1)]                    (I is 1-based)
//   E(I)      =>  e[eOff + (I-1)]                    (I is 1-based)
//   TAUQ(I)   =>  tauq[tauqOff + (I-1)]              (I is 1-based)
//   TAUP(I)   =>  taup[taupOff + (I-1)]              (I is 1-based)
//   X(I,J)    =>  x[xOff + (I-1) + (J-1)*ldx]       (I,J are 1-based)
//   Y(I,J)    =>  y[yOff + (I-1) + (J-1)*ldy]       (I,J are 1-based)

import { dlarfg } from "./dlarfg.js";
import { dscal } from "../BLAS/dscal.js";
import { dgemv } from "../BLAS/dgemv.js";
import { NOTRANS, TRANS } from "../utils/constants.js";

const ZERO = 0.0;
const ONE = 1.0;

export function dlabrd(
  m: number,
  n: number,
  nb: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  d: Float64Array,
  dOff: number,
  e: Float64Array,
  eOff: number,
  tauq: Float64Array,
  tauqOff: number,
  taup: Float64Array,
  taupOff: number,
  x: Float64Array,
  xOff: number,
  ldx: number,
  y: Float64Array,
  yOff: number,
  ldy: number
): void {
  // Quick return if possible
  if (m <= 0 || n <= 0) {
    return;
  }

  let result: { alpha: number; tau: number };

  if (m >= n) {
    // Reduce to upper bidiagonal form
    for (let i = 1; i <= nb; i++) {
      // Update A(i:m,i)
      dgemv(
        NOTRANS,
        m - i + 1,
        i - 1,
        -ONE,
        a,
        aOff + (i - 1) + 0 * lda,
        lda,
        y,
        yOff + (i - 1) + 0 * ldy,
        ldy,
        ONE,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        1
      );
      dgemv(
        NOTRANS,
        m - i + 1,
        i - 1,
        -ONE,
        x,
        xOff + (i - 1) + 0 * ldx,
        ldx,
        a,
        aOff + 0 + (i - 1) * lda,
        1,
        ONE,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        1
      );

      // Generate reflection Q(i) to annihilate A(i+1:m,i)
      result = dlarfg(
        m - i + 1,
        a[aOff + (i - 1) + (i - 1) * lda],
        a,
        aOff + (Math.min(i + 1, m) - 1) + (i - 1) * lda,
        1
      );
      d[dOff + (i - 1)] = result.alpha;
      tauq[tauqOff + (i - 1)] = result.tau;

      if (i < n) {
        a[aOff + (i - 1) + (i - 1) * lda] = ONE;

        // Compute Y(i+1:n,i)
        dgemv(
          TRANS,
          m - i + 1,
          n - i,
          ONE,
          a,
          aOff + (i - 1) + i * lda,
          lda,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          1,
          ZERO,
          y,
          yOff + i + (i - 1) * ldy,
          1
        );
        dgemv(
          TRANS,
          m - i + 1,
          i - 1,
          ONE,
          a,
          aOff + (i - 1) + 0 * lda,
          lda,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          1,
          ZERO,
          y,
          yOff + 0 + (i - 1) * ldy,
          1
        );
        dgemv(
          NOTRANS,
          n - i,
          i - 1,
          -ONE,
          y,
          yOff + i + 0 * ldy,
          ldy,
          y,
          yOff + 0 + (i - 1) * ldy,
          1,
          ONE,
          y,
          yOff + i + (i - 1) * ldy,
          1
        );
        dgemv(
          TRANS,
          m - i + 1,
          i - 1,
          ONE,
          x,
          xOff + (i - 1) + 0 * ldx,
          ldx,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          1,
          ZERO,
          y,
          yOff + 0 + (i - 1) * ldy,
          1
        );
        dgemv(
          TRANS,
          i - 1,
          n - i,
          -ONE,
          a,
          aOff + 0 + i * lda,
          lda,
          y,
          yOff + 0 + (i - 1) * ldy,
          1,
          ONE,
          y,
          yOff + i + (i - 1) * ldy,
          1
        );
        dscal(n - i, tauq[tauqOff + (i - 1)], y, yOff + i + (i - 1) * ldy, 1);

        // Update A(i,i+1:n)
        dgemv(
          NOTRANS,
          n - i,
          i,
          -ONE,
          y,
          yOff + i + 0 * ldy,
          ldy,
          a,
          aOff + (i - 1) + 0 * lda,
          lda,
          ONE,
          a,
          aOff + (i - 1) + i * lda,
          lda
        );
        dgemv(
          TRANS,
          i - 1,
          n - i,
          -ONE,
          a,
          aOff + 0 + i * lda,
          lda,
          x,
          xOff + (i - 1) + 0 * ldx,
          ldx,
          ONE,
          a,
          aOff + (i - 1) + i * lda,
          lda
        );

        // Generate reflection P(i) to annihilate A(i,i+2:n)
        result = dlarfg(
          n - i,
          a[aOff + (i - 1) + i * lda],
          a,
          aOff + (i - 1) + (Math.min(i + 2, n) - 1) * lda,
          lda
        );
        e[eOff + (i - 1)] = result.alpha;
        taup[taupOff + (i - 1)] = result.tau;
        a[aOff + (i - 1) + i * lda] = ONE;

        // Compute X(i+1:m,i)
        dgemv(
          NOTRANS,
          m - i,
          n - i,
          ONE,
          a,
          aOff + i + i * lda,
          lda,
          a,
          aOff + (i - 1) + i * lda,
          lda,
          ZERO,
          x,
          xOff + i + (i - 1) * ldx,
          1
        );
        dgemv(
          TRANS,
          n - i,
          i,
          ONE,
          y,
          yOff + i + 0 * ldy,
          ldy,
          a,
          aOff + (i - 1) + i * lda,
          lda,
          ZERO,
          x,
          xOff + 0 + (i - 1) * ldx,
          1
        );
        dgemv(
          NOTRANS,
          m - i,
          i,
          -ONE,
          a,
          aOff + i + 0 * lda,
          lda,
          x,
          xOff + 0 + (i - 1) * ldx,
          1,
          ONE,
          x,
          xOff + i + (i - 1) * ldx,
          1
        );
        dgemv(
          NOTRANS,
          i - 1,
          n - i,
          ONE,
          a,
          aOff + 0 + i * lda,
          lda,
          a,
          aOff + (i - 1) + i * lda,
          lda,
          ZERO,
          x,
          xOff + 0 + (i - 1) * ldx,
          1
        );
        dgemv(
          NOTRANS,
          m - i,
          i - 1,
          -ONE,
          x,
          xOff + i + 0 * ldx,
          ldx,
          x,
          xOff + 0 + (i - 1) * ldx,
          1,
          ONE,
          x,
          xOff + i + (i - 1) * ldx,
          1
        );
        dscal(m - i, taup[taupOff + (i - 1)], x, xOff + i + (i - 1) * ldx, 1);
      } else {
        a[aOff + (i - 1) + (i - 1) * lda] = result.alpha;
      }
    }
  } else {
    // Reduce to lower bidiagonal form
    for (let i = 1; i <= nb; i++) {
      // Update A(i,i:n)
      dgemv(
        NOTRANS,
        n - i + 1,
        i - 1,
        -ONE,
        y,
        yOff + (i - 1) + 0 * ldy,
        ldy,
        a,
        aOff + (i - 1) + 0 * lda,
        lda,
        ONE,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda
      );
      dgemv(
        TRANS,
        i - 1,
        n - i + 1,
        -ONE,
        a,
        aOff + 0 + (i - 1) * lda,
        lda,
        x,
        xOff + (i - 1) + 0 * ldx,
        ldx,
        ONE,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda
      );

      // Generate reflection P(i) to annihilate A(i,i+1:n)
      result = dlarfg(
        n - i + 1,
        a[aOff + (i - 1) + (i - 1) * lda],
        a,
        aOff + (i - 1) + (Math.min(i + 1, n) - 1) * lda,
        lda
      );
      d[dOff + (i - 1)] = result.alpha;
      taup[taupOff + (i - 1)] = result.tau;

      if (i < m) {
        a[aOff + (i - 1) + (i - 1) * lda] = ONE;

        // Compute X(i+1:m,i)
        dgemv(
          NOTRANS,
          m - i,
          n - i + 1,
          ONE,
          a,
          aOff + i + (i - 1) * lda,
          lda,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          lda,
          ZERO,
          x,
          xOff + i + (i - 1) * ldx,
          1
        );
        dgemv(
          TRANS,
          n - i + 1,
          i - 1,
          ONE,
          y,
          yOff + (i - 1) + 0 * ldy,
          ldy,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          lda,
          ZERO,
          x,
          xOff + 0 + (i - 1) * ldx,
          1
        );
        dgemv(
          NOTRANS,
          m - i,
          i - 1,
          -ONE,
          a,
          aOff + i + 0 * lda,
          lda,
          x,
          xOff + 0 + (i - 1) * ldx,
          1,
          ONE,
          x,
          xOff + i + (i - 1) * ldx,
          1
        );
        dgemv(
          NOTRANS,
          i - 1,
          n - i + 1,
          ONE,
          a,
          aOff + 0 + (i - 1) * lda,
          lda,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          lda,
          ZERO,
          x,
          xOff + 0 + (i - 1) * ldx,
          1
        );
        dgemv(
          NOTRANS,
          m - i,
          i - 1,
          -ONE,
          x,
          xOff + i + 0 * ldx,
          ldx,
          x,
          xOff + 0 + (i - 1) * ldx,
          1,
          ONE,
          x,
          xOff + i + (i - 1) * ldx,
          1
        );
        dscal(m - i, taup[taupOff + (i - 1)], x, xOff + i + (i - 1) * ldx, 1);

        // Update A(i+1:m,i)
        dgemv(
          NOTRANS,
          m - i,
          i - 1,
          -ONE,
          a,
          aOff + i + 0 * lda,
          lda,
          y,
          yOff + (i - 1) + 0 * ldy,
          ldy,
          ONE,
          a,
          aOff + i + (i - 1) * lda,
          1
        );
        dgemv(
          NOTRANS,
          m - i,
          i,
          -ONE,
          x,
          xOff + i + 0 * ldx,
          ldx,
          a,
          aOff + 0 + (i - 1) * lda,
          1,
          ONE,
          a,
          aOff + i + (i - 1) * lda,
          1
        );

        // Generate reflection Q(i) to annihilate A(i+2:m,i)
        result = dlarfg(
          m - i,
          a[aOff + i + (i - 1) * lda],
          a,
          aOff + (Math.min(i + 2, m) - 1) + (i - 1) * lda,
          1
        );
        e[eOff + (i - 1)] = result.alpha;
        tauq[tauqOff + (i - 1)] = result.tau;
        a[aOff + i + (i - 1) * lda] = ONE;

        // Compute Y(i+1:n,i)
        dgemv(
          TRANS,
          m - i,
          n - i,
          ONE,
          a,
          aOff + i + i * lda,
          lda,
          a,
          aOff + i + (i - 1) * lda,
          1,
          ZERO,
          y,
          yOff + i + (i - 1) * ldy,
          1
        );
        dgemv(
          TRANS,
          m - i,
          i - 1,
          ONE,
          a,
          aOff + i + 0 * lda,
          lda,
          a,
          aOff + i + (i - 1) * lda,
          1,
          ZERO,
          y,
          yOff + 0 + (i - 1) * ldy,
          1
        );
        dgemv(
          NOTRANS,
          n - i,
          i - 1,
          -ONE,
          y,
          yOff + i + 0 * ldy,
          ldy,
          y,
          yOff + 0 + (i - 1) * ldy,
          1,
          ONE,
          y,
          yOff + i + (i - 1) * ldy,
          1
        );
        dgemv(
          TRANS,
          m - i,
          i,
          ONE,
          x,
          xOff + i + 0 * ldx,
          ldx,
          a,
          aOff + i + (i - 1) * lda,
          1,
          ZERO,
          y,
          yOff + 0 + (i - 1) * ldy,
          1
        );
        dgemv(
          TRANS,
          i,
          n - i,
          -ONE,
          a,
          aOff + 0 + i * lda,
          lda,
          y,
          yOff + 0 + (i - 1) * ldy,
          1,
          ONE,
          y,
          yOff + i + (i - 1) * ldy,
          1
        );
        dscal(n - i, tauq[tauqOff + (i - 1)], y, yOff + i + (i - 1) * ldy, 1);
      } else {
        a[aOff + (i - 1) + (i - 1) * lda] = result.alpha;
      }
    }
  }
}
