// Translated from BLAS/SRC/dtrmv.f
// DTRMV performs one of the matrix-vector operations
//   x := A*x   or   x := A**T*x
// where A is a unit or non-unit upper or lower triangular matrix.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   X(I)    =>  x[xOff + (I-1)*incx]            (I is 1-based)

import { xerbla } from "../utils/xerbla.js";
import {
  UPPER,
  LOWER,
  NOTRANS,
  TRANS,
  CONJTRANS,
  UNIT,
  NONUNIT,
} from "../utils/constants.js";

export function dtrmv(
  uplo: number,
  trans: number,
  diag: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  x: Float64Array,
  xOff: number,
  incx: number
): void {
  let info = 0;
  if (uplo !== UPPER && uplo !== LOWER) {
    info = 1;
  } else if (trans !== NOTRANS && trans !== TRANS && trans !== CONJTRANS) {
    info = 2;
  } else if (diag !== UNIT && diag !== NONUNIT) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (lda < Math.max(1, n)) {
    info = 6;
  } else if (incx === 0) {
    info = 8;
  }
  if (info !== 0) {
    xerbla("DTRMV ", info);
    return;
  }

  if (n === 0) return;

  const nounit = diag === NONUNIT;

  // Set up start point KX (1-based) for non-unit or negative increments
  let kx = 0;
  if (incx <= 0) {
    kx = 1 - (n - 1) * incx;
  } else if (incx !== 1) {
    kx = 1;
  }

  if (trans === NOTRANS) {
    // Form x := A*x
    if (uplo === UPPER) {
      if (incx === 1) {
        for (let j = 1; j <= n; j++) {
          if (x[xOff + (j - 1)] !== 0.0) {
            const temp = x[xOff + (j - 1)];
            for (let i = 1; i <= j - 1; i++) {
              x[xOff + (i - 1)] += temp * a[aOff + (i - 1) + (j - 1) * lda];
            }
            if (nounit) x[xOff + (j - 1)] *= a[aOff + (j - 1) + (j - 1) * lda];
          }
        }
      } else {
        let jx = kx;
        for (let j = 1; j <= n; j++) {
          if (x[xOff + (jx - 1)] !== 0.0) {
            const temp = x[xOff + (jx - 1)];
            let ix = kx;
            for (let i = 1; i <= j - 1; i++) {
              x[xOff + (ix - 1)] += temp * a[aOff + (i - 1) + (j - 1) * lda];
              ix += incx;
            }
            if (nounit) x[xOff + (jx - 1)] *= a[aOff + (j - 1) + (j - 1) * lda];
          }
          jx += incx;
        }
      }
    } else {
      if (incx === 1) {
        for (let j = n; j >= 1; j--) {
          if (x[xOff + (j - 1)] !== 0.0) {
            const temp = x[xOff + (j - 1)];
            for (let i = n; i >= j + 1; i--) {
              x[xOff + (i - 1)] += temp * a[aOff + (i - 1) + (j - 1) * lda];
            }
            if (nounit) x[xOff + (j - 1)] *= a[aOff + (j - 1) + (j - 1) * lda];
          }
        }
      } else {
        kx += (n - 1) * incx;
        let jx = kx;
        for (let j = n; j >= 1; j--) {
          if (x[xOff + (jx - 1)] !== 0.0) {
            const temp = x[xOff + (jx - 1)];
            let ix = kx;
            for (let i = n; i >= j + 1; i--) {
              x[xOff + (ix - 1)] += temp * a[aOff + (i - 1) + (j - 1) * lda];
              ix -= incx;
            }
            if (nounit) x[xOff + (jx - 1)] *= a[aOff + (j - 1) + (j - 1) * lda];
          }
          jx -= incx;
        }
      }
    }
  } else {
    // Form x := A**T*x
    if (uplo === UPPER) {
      if (incx === 1) {
        for (let j = n; j >= 1; j--) {
          let temp = x[xOff + (j - 1)];
          if (nounit) temp *= a[aOff + (j - 1) + (j - 1) * lda];
          for (let i = j - 1; i >= 1; i--) {
            temp += a[aOff + (i - 1) + (j - 1) * lda] * x[xOff + (i - 1)];
          }
          x[xOff + (j - 1)] = temp;
        }
      } else {
        let jx = kx + (n - 1) * incx;
        for (let j = n; j >= 1; j--) {
          let temp = x[xOff + (jx - 1)];
          let ix = jx;
          if (nounit) temp *= a[aOff + (j - 1) + (j - 1) * lda];
          for (let i = j - 1; i >= 1; i--) {
            ix -= incx;
            temp += a[aOff + (i - 1) + (j - 1) * lda] * x[xOff + (ix - 1)];
          }
          x[xOff + (jx - 1)] = temp;
          jx -= incx;
        }
      }
    } else {
      if (incx === 1) {
        for (let j = 1; j <= n; j++) {
          let temp = x[xOff + (j - 1)];
          if (nounit) temp *= a[aOff + (j - 1) + (j - 1) * lda];
          for (let i = j + 1; i <= n; i++) {
            temp += a[aOff + (i - 1) + (j - 1) * lda] * x[xOff + (i - 1)];
          }
          x[xOff + (j - 1)] = temp;
        }
      } else {
        let jx = kx;
        for (let j = 1; j <= n; j++) {
          let temp = x[xOff + (jx - 1)];
          let ix = jx;
          if (nounit) temp *= a[aOff + (j - 1) + (j - 1) * lda];
          for (let i = j + 1; i <= n; i++) {
            ix += incx;
            temp += a[aOff + (i - 1) + (j - 1) * lda] * x[xOff + (ix - 1)];
          }
          x[xOff + (jx - 1)] = temp;
          jx += incx;
        }
      }
    }
  }
}
