// dgemm_optimized.ts
// Fully inlined + JS-engine-optimized implementation of DGEMM.
//
// Computes one of the matrix-matrix operations
//   C := alpha*op(A)*op(B) + beta*C
// where op(X) is X or X**T.
//
// Optimization principles (matching dgeqrf_optimized.ts / dorgqr_optimized.ts):
//
//   1. 0-based loop variables — eliminates (i-1), (j-1), (l-1) subtractions
//      from every array index in the inner loops. Each index is just
//      `base + row` or `base + col`, a single addition.
//
//   2. Column base offsets hoisted — jBBase = bOff + j*ldb, jCBase = cOff + j*ldc,
//      and lABase = aOff + l*lda are computed once per outer-loop iteration.
//      Inner loops only add the row index (stride-1 linear scan). V8 can
//      bounds-check-hoist and auto-vectorise these loops.
//
//   3. TypedArray.fill() for zero-initialisation — V8 lowers fill(0) to memset
//      for Float64Array; faster than a scalar loop over m elements.
//
//   4. Scalar temp kept in register — for the A**T cases the dot-product
//      accumulates into a JS local `temp`; only one store to C[i,j] after
//      the inner loop, avoiding repeated read-modify-write on a memory cell.
//
//   5. beta fast paths preserved — beta===0 and beta===1 branches remain
//      outside the inner loops so the common cases pay zero multiply cost.
//
//   6. No work arrays — all temporaries are scalar locals (w, temp) that the
//      JIT keeps in registers. No heap allocation in the hot path.
//
// Array indexing (column-major, 0-based loops):
//   A(i,j)  =>  a[aOff + i + j*lda]   (i,j are 0-based)
//   B(i,j)  =>  b[bOff + i + j*ldb]
//   C(i,j)  =>  c[cOff + i + j*ldc]

import { xerbla } from "../utils/xerbla.js";
import { NOTRANS, TRANS, CONJTRANS } from "../utils/constants.js";

export function dgemm_optimized(
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
  const nota = transa === NOTRANS;
  const notb = transb === NOTRANS;
  const nrowa = nota ? m : k;
  const nrowb = notb ? k : n;

  // Validate inputs
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

  // Handle alpha === 0: C := beta*C
  if (alpha === 0.0) {
    if (beta === 0.0) {
      for (let j = 0; j < n; j++) {
        const jCBase = cOff + j * ldc;
        c.fill(0.0, jCBase, jCBase + m);
      }
    } else {
      for (let j = 0; j < n; j++) {
        const jCBase = cOff + j * ldc;
        for (let i = 0; i < m; i++) c[jCBase + i] *= beta;
      }
    }
    return;
  }

  if (notb) {
    if (nota) {
      // Form C := alpha*A*B + beta*C
      // Inner loop: a[lABase + i] is stride-1; b[jBBase + l] is stride-1.
      // c[jCBase + i] is stride-1. All three hot vectors are contiguous.
      for (let j = 0; j < n; j++) {
        const jBBase = bOff + j * ldb;
        const jCBase = cOff + j * ldc;

        // Scale C[:,j] by beta (or zero-init)
        if (beta === 0.0) {
          c.fill(0.0, jCBase, jCBase + m);
        } else if (beta !== 1.0) {
          for (let i = 0; i < m; i++) c[jCBase + i] *= beta;
        }

        // Accumulate alpha * A * B[:,j] into C[:,j]
        for (let l = 0; l < k; l++) {
          const temp = alpha * b[jBBase + l]; // scalar: stays in register
          const lABase = aOff + l * lda; // base of A[:,l] — hoisted
          for (let i = 0; i < m; i++) {
            c[jCBase + i] += temp * a[lABase + i];
          }
        }
      }
    } else {
      // Form C := alpha*A**T*B + beta*C
      // a[iABase + l] is stride-1 in l; b[jBBase + l] is stride-1 in l.
      // The dot product runs in stride-1 for both A and B — good cache use.
      for (let j = 0; j < n; j++) {
        const jBBase = bOff + j * ldb;
        const jCBase = cOff + j * ldc;
        for (let i = 0; i < m; i++) {
          const iABase = aOff + i * lda; // base of A[:,i] — A transposed
          let temp = 0.0;
          for (let l = 0; l < k; l++) {
            temp += a[iABase + l] * b[jBBase + l];
          }
          if (beta === 0.0) {
            c[jCBase + i] = alpha * temp;
          } else {
            c[jCBase + i] = alpha * temp + beta * c[jCBase + i];
          }
        }
      }
    }
  } else {
    if (nota) {
      // Form C := alpha*A*B**T + beta*C
      // b[bOff + j + l*ldb]: j is fixed per outer loop; l varies stride-ldb.
      // a[lABase + i] is stride-1; c[jCBase + i] is stride-1.
      for (let j = 0; j < n; j++) {
        const jCBase = cOff + j * ldc;

        if (beta === 0.0) {
          c.fill(0.0, jCBase, jCBase + m);
        } else if (beta !== 1.0) {
          for (let i = 0; i < m; i++) c[jCBase + i] *= beta;
        }

        for (let l = 0; l < k; l++) {
          const temp = alpha * b[bOff + j + l * ldb]; // B**T: B(j,l)
          const lABase = aOff + l * lda;
          for (let i = 0; i < m; i++) {
            c[jCBase + i] += temp * a[lABase + i];
          }
        }
      }
    } else {
      // Form C := alpha*A**T*B**T + beta*C
      // a[iABase + l] stride-1 in l; b[bOff + j + l*ldb] stride-ldb in l.
      for (let j = 0; j < n; j++) {
        const jCBase = cOff + j * ldc;
        for (let i = 0; i < m; i++) {
          const iABase = aOff + i * lda;
          let temp = 0.0;
          for (let l = 0; l < k; l++) {
            temp += a[iABase + l] * b[bOff + j + l * ldb];
          }
          if (beta === 0.0) {
            c[jCBase + i] = alpha * temp;
          } else {
            c[jCBase + i] = alpha * temp + beta * c[jCBase + i];
          }
        }
      }
    }
  }
}
