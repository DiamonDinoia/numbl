// Translated from BLAS/SRC/dgemm.f
// DGEMM performs one of the matrix-matrix operations
//   C := alpha*op(A)*op(B) + beta*C
// where op(X) is X or X**T.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)

import { xerbla } from "../utils/xerbla.js";
import { NOTRANS, TRANS, CONJTRANS } from "../utils/constants.js";

export function dgemm(
  transa: number,
  transb: number,
  m: number,
  n: number,
  k: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  b: Float64Array,
  bOff: number,
  ldb: number,
  beta: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  // Set NOTA and NOTB as true if A and B respectively are not transposed
  const nota = transa === NOTRANS;
  const notb = transb === NOTRANS;
  const nrowa = nota ? m : k;
  const nrowb = notb ? k : n;

  // Test the input parameters
  let info = 0;
  if (!nota && transa !== CONJTRANS && transa !== TRANS) {
    info = 1;
  } else if (!notb && transb !== CONJTRANS && transb !== TRANS) {
    info = 2;
  } else if (m < 0) {
    info = 3;
  } else if (n < 0) {
    info = 4;
  } else if (k < 0) {
    info = 5;
  } else if (lda < Math.max(1, nrowa)) {
    info = 8;
  } else if (ldb < Math.max(1, nrowb)) {
    info = 10;
  } else if (ldc < Math.max(1, m)) {
    info = 13;
  }
  if (info !== 0) {
    xerbla("DGEMM ", info);
    return;
  }

  // Quick return if possible
  if (m === 0 || n === 0 || ((alpha === 0.0 || k === 0) && beta === 1.0))
    return;

  // And if alpha === 0
  if (alpha === 0.0) {
    if (beta === 0.0) {
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= m; i++) {
          c[cOff + (i - 1) + (j - 1) * ldc] = 0.0;
        }
      }
    } else {
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= m; i++) {
          c[cOff + (i - 1) + (j - 1) * ldc] =
            beta * c[cOff + (i - 1) + (j - 1) * ldc];
        }
      }
    }
    return;
  }

  // Start the operations
  if (notb) {
    if (nota) {
      // Form C := alpha*A*B + beta*C
      for (let j = 1; j <= n; j++) {
        if (beta === 0.0) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] =
              beta * c[cOff + (i - 1) + (j - 1) * ldc];
          }
        }
        for (let l = 1; l <= k; l++) {
          const temp = alpha * b[bOff + (l - 1) + (j - 1) * ldb];
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] +=
              temp * a[aOff + (i - 1) + (l - 1) * lda];
          }
        }
      }
    } else {
      // Form C := alpha*A**T*B + beta*C
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= m; i++) {
          let temp = 0.0;
          for (let l = 1; l <= k; l++) {
            temp +=
              a[aOff + (l - 1) + (i - 1) * lda] *
              b[bOff + (l - 1) + (j - 1) * ldb];
          }
          if (beta === 0.0) {
            c[cOff + (i - 1) + (j - 1) * ldc] = alpha * temp;
          } else {
            c[cOff + (i - 1) + (j - 1) * ldc] =
              alpha * temp + beta * c[cOff + (i - 1) + (j - 1) * ldc];
          }
        }
      }
    }
  } else {
    if (nota) {
      // Form C := alpha*A*B**T + beta*C
      for (let j = 1; j <= n; j++) {
        if (beta === 0.0) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] = 0.0;
          }
        } else if (beta !== 1.0) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] =
              beta * c[cOff + (i - 1) + (j - 1) * ldc];
          }
        }
        for (let l = 1; l <= k; l++) {
          const temp = alpha * b[bOff + (j - 1) + (l - 1) * ldb];
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] +=
              temp * a[aOff + (i - 1) + (l - 1) * lda];
          }
        }
      }
    } else {
      // Form C := alpha*A**T*B**T + beta*C
      for (let j = 1; j <= n; j++) {
        for (let i = 1; i <= m; i++) {
          let temp = 0.0;
          for (let l = 1; l <= k; l++) {
            temp +=
              a[aOff + (l - 1) + (i - 1) * lda] *
              b[bOff + (j - 1) + (l - 1) * ldb];
          }
          if (beta === 0.0) {
            c[cOff + (i - 1) + (j - 1) * ldc] = alpha * temp;
          } else {
            c[cOff + (i - 1) + (j - 1) * ldc] =
              alpha * temp + beta * c[cOff + (i - 1) + (j - 1) * ldc];
          }
        }
      }
    }
  }
}
