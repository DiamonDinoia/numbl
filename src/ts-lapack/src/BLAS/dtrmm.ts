// Translated from BLAS/SRC/dtrmm.f
// DTRMM performs one of the matrix-matrix operations
//   B := alpha*op(A)*B   or   B := alpha*B*op(A)
// where A is a unit or non-unit upper or lower triangular matrix.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   B(I,J)  =>  b[bOff + (I-1) + (J-1)*ldb]   (I,J are 1-based)

import { xerbla } from "../utils/xerbla.js";
import {
  LEFT,
  RIGHT,
  UPPER,
  LOWER,
  NOTRANS,
  TRANS,
  CONJTRANS,
  UNIT,
  NONUNIT,
} from "../utils/constants.js";

export function dtrmm(
  side: number,
  uplo: number,
  transa: number,
  diag: number,
  m: number,
  n: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number
): void {
  const lside = side === LEFT;
  const nrowa = lside ? m : n;
  const nounit = diag === NONUNIT;
  const upper = uplo === UPPER;

  let info = 0;
  if (side !== LEFT && side !== RIGHT) {
    info = 1;
  } else if (uplo !== UPPER && uplo !== LOWER) {
    info = 2;
  } else if (transa !== NOTRANS && transa !== TRANS && transa !== CONJTRANS) {
    info = 3;
  } else if (diag !== UNIT && diag !== NONUNIT) {
    info = 4;
  } else if (m < 0) {
    info = 5;
  } else if (n < 0) {
    info = 6;
  } else if (lda < Math.max(1, nrowa)) {
    info = 9;
  } else if (ldb < Math.max(1, m)) {
    info = 11;
  }
  if (info !== 0) {
    xerbla("DTRMM ", info);
    return;
  }

  if (m === 0 || n === 0) return;

  if (alpha === 0.0) {
    for (let j = 1; j <= n; j++) {
      for (let i = 1; i <= m; i++) {
        b[bOff + (i - 1) + (j - 1) * ldb] = 0.0;
      }
    }
    return;
  }

  if (lside) {
    if (transa === NOTRANS) {
      // Form B := alpha*A*B
      if (upper) {
        for (let j = 1; j <= n; j++) {
          for (let k = 1; k <= m; k++) {
            if (b[bOff + (k - 1) + (j - 1) * ldb] !== 0.0) {
              const temp = alpha * b[bOff + (k - 1) + (j - 1) * ldb];
              for (let i = 1; i <= k - 1; i++) {
                b[bOff + (i - 1) + (j - 1) * ldb] +=
                  temp * a[aOff + (i - 1) + (k - 1) * lda];
              }
              b[bOff + (k - 1) + (j - 1) * ldb] = nounit
                ? temp * a[aOff + (k - 1) + (k - 1) * lda]
                : temp;
            }
          }
        }
      } else {
        for (let j = 1; j <= n; j++) {
          for (let k = m; k >= 1; k--) {
            if (b[bOff + (k - 1) + (j - 1) * ldb] !== 0.0) {
              const temp = alpha * b[bOff + (k - 1) + (j - 1) * ldb];
              b[bOff + (k - 1) + (j - 1) * ldb] = temp;
              if (nounit) {
                b[bOff + (k - 1) + (j - 1) * ldb] *=
                  a[aOff + (k - 1) + (k - 1) * lda];
              }
              for (let i = k + 1; i <= m; i++) {
                b[bOff + (i - 1) + (j - 1) * ldb] +=
                  temp * a[aOff + (i - 1) + (k - 1) * lda];
              }
            }
          }
        }
      }
    } else {
      // Form B := alpha*A**T*B
      if (upper) {
        for (let j = 1; j <= n; j++) {
          for (let i = m; i >= 1; i--) {
            let temp = b[bOff + (i - 1) + (j - 1) * ldb];
            if (nounit) temp *= a[aOff + (i - 1) + (i - 1) * lda];
            for (let k = 1; k <= i - 1; k++) {
              temp +=
                a[aOff + (k - 1) + (i - 1) * lda] *
                b[bOff + (k - 1) + (j - 1) * ldb];
            }
            b[bOff + (i - 1) + (j - 1) * ldb] = alpha * temp;
          }
        }
      } else {
        for (let j = 1; j <= n; j++) {
          for (let i = 1; i <= m; i++) {
            let temp = b[bOff + (i - 1) + (j - 1) * ldb];
            if (nounit) temp *= a[aOff + (i - 1) + (i - 1) * lda];
            for (let k = i + 1; k <= m; k++) {
              temp +=
                a[aOff + (k - 1) + (i - 1) * lda] *
                b[bOff + (k - 1) + (j - 1) * ldb];
            }
            b[bOff + (i - 1) + (j - 1) * ldb] = alpha * temp;
          }
        }
      }
    }
  } else {
    if (transa === NOTRANS) {
      // Form B := alpha*B*A
      if (upper) {
        for (let j = n; j >= 1; j--) {
          let temp = alpha;
          if (nounit) temp *= a[aOff + (j - 1) + (j - 1) * lda];
          for (let i = 1; i <= m; i++) {
            b[bOff + (i - 1) + (j - 1) * ldb] =
              temp * b[bOff + (i - 1) + (j - 1) * ldb];
          }
          for (let k = 1; k <= j - 1; k++) {
            if (a[aOff + (k - 1) + (j - 1) * lda] !== 0.0) {
              temp = alpha * a[aOff + (k - 1) + (j - 1) * lda];
              for (let i = 1; i <= m; i++) {
                b[bOff + (i - 1) + (j - 1) * ldb] +=
                  temp * b[bOff + (i - 1) + (k - 1) * ldb];
              }
            }
          }
        }
      } else {
        for (let j = 1; j <= n; j++) {
          let temp = alpha;
          if (nounit) temp *= a[aOff + (j - 1) + (j - 1) * lda];
          for (let i = 1; i <= m; i++) {
            b[bOff + (i - 1) + (j - 1) * ldb] =
              temp * b[bOff + (i - 1) + (j - 1) * ldb];
          }
          for (let k = j + 1; k <= n; k++) {
            if (a[aOff + (k - 1) + (j - 1) * lda] !== 0.0) {
              temp = alpha * a[aOff + (k - 1) + (j - 1) * lda];
              for (let i = 1; i <= m; i++) {
                b[bOff + (i - 1) + (j - 1) * ldb] +=
                  temp * b[bOff + (i - 1) + (k - 1) * ldb];
              }
            }
          }
        }
      }
    } else {
      // Form B := alpha*B*A**T
      if (upper) {
        for (let k = 1; k <= n; k++) {
          for (let j = 1; j <= k - 1; j++) {
            if (a[aOff + (j - 1) + (k - 1) * lda] !== 0.0) {
              const temp = alpha * a[aOff + (j - 1) + (k - 1) * lda];
              for (let i = 1; i <= m; i++) {
                b[bOff + (i - 1) + (j - 1) * ldb] +=
                  temp * b[bOff + (i - 1) + (k - 1) * ldb];
              }
            }
          }
          let temp = alpha;
          if (nounit) temp *= a[aOff + (k - 1) + (k - 1) * lda];
          if (temp !== 1.0) {
            for (let i = 1; i <= m; i++) {
              b[bOff + (i - 1) + (k - 1) * ldb] =
                temp * b[bOff + (i - 1) + (k - 1) * ldb];
            }
          }
        }
      } else {
        for (let k = n; k >= 1; k--) {
          for (let j = k + 1; j <= n; j++) {
            if (a[aOff + (j - 1) + (k - 1) * lda] !== 0.0) {
              const temp = alpha * a[aOff + (j - 1) + (k - 1) * lda];
              for (let i = 1; i <= m; i++) {
                b[bOff + (i - 1) + (j - 1) * ldb] +=
                  temp * b[bOff + (i - 1) + (k - 1) * ldb];
              }
            }
          }
          let temp = alpha;
          if (nounit) temp *= a[aOff + (k - 1) + (k - 1) * lda];
          if (temp !== 1.0) {
            for (let i = 1; i <= m; i++) {
              b[bOff + (i - 1) + (k - 1) * ldb] =
                temp * b[bOff + (i - 1) + (k - 1) * ldb];
            }
          }
        }
      }
    }
  }
}
