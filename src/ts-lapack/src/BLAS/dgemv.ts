// Translated from BLAS/SRC/dgemv.f
// DGEMV performs one of the matrix-vector operations
//   y := alpha*A*x + beta*y   or   y := alpha*A**T*x + beta*y
//
// Optimizations vs. the Fortran reference implementation:
//
//   1. 0-based indexing: Fortran uses 1-based indices, so every array access
//      required subtracting 1 (e.g. a[aOff + (i-1) + (j-1)*lda]). Switching
//      to 0-based removes those subtractions from every inner-loop iteration,
//      saving O(m*n) arithmetic operations.
//
//   2. Column offset hoisted out of inner loop: aCol = aOff + j*lda is
//      computed once per outer iteration. The inner loop then only adds i,
//      avoiding a multiply and extra addition on every a[] access.
//
//   3. trans check hoisted: trans === NOTRANS is evaluated once into the
//      boolean `notrans` rather than being re-evaluated inside conditionals
//      that appear multiple times through the function.
//
//   4. Float64Array.fill() for zeroing: when beta===0 and incy===1, the
//      native typed-array fill() is used instead of a JS loop. V8 can
//      lower this to a memset-equivalent, which is faster and SIMD-friendly.
//
//   5. Tight contiguous fast path: the incx===1 && incy===1 case (by far the
//      most common in practice) gets its own inner loop with no stride
//      arithmetic, allowing V8 to apply loop optimizations more aggressively.
//
// Array indexing convention (column-major, 0-based offsets):
//   A(i,j)  =>  a[aOff + i + j*lda]   (i,j are 0-based)
//   X(i)    =>  x[xOff + i*incx]       (i is 0-based)
//   Y(i)    =>  y[yOff + i*incy]       (i is 0-based)

import { xerbla } from "../utils/xerbla.js";
import { NOTRANS, TRANS, CONJTRANS } from "../utils/constants.js";

export function dgemv(
  trans: number,
  m: number,
  n: number,
  alpha: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  x: Float64Array,
  xOff: number,
  incx: number,
  beta: number,
  y: Float64Array,
  yOff: number,
  incy: number
): void {
  const notrans = trans === NOTRANS;

  // Test the input parameters
  let info = 0;
  if (!notrans && trans !== TRANS && trans !== CONJTRANS) {
    info = 1;
  } else if (m < 0) {
    info = 2;
  } else if (n < 0) {
    info = 3;
  } else if (lda < Math.max(1, m)) {
    info = 6;
  } else if (incx === 0) {
    info = 8;
  } else if (incy === 0) {
    info = 11;
  }
  if (info !== 0) {
    xerbla("DGEMV ", info);
    return;
  }

  // Quick return if possible
  if (m === 0 || n === 0 || (alpha === 0.0 && beta === 1.0)) return;

  const lenx = notrans ? n : m;
  const leny = notrans ? m : n;

  // 0-based start points (for negative strides, start at the far end)
  const kx = incx > 0 ? 0 : (lenx - 1) * -incx;
  const ky = incy > 0 ? 0 : (leny - 1) * -incy;

  // Form y := beta*y
  if (beta !== 1.0) {
    if (beta === 0.0) {
      if (incy === 1) {
        y.fill(0.0, yOff, yOff + leny);
      } else {
        let iy = ky;
        for (let i = 0; i < leny; i++) {
          y[yOff + iy] = 0.0;
          iy += incy;
        }
      }
    } else {
      if (incy === 1) {
        const yEnd = yOff + leny;
        for (let i = yOff; i < yEnd; i++) {
          y[i] *= beta;
        }
      } else {
        let iy = ky;
        for (let i = 0; i < leny; i++) {
          y[yOff + iy] *= beta;
          iy += incy;
        }
      }
    }
  }
  if (alpha === 0.0) return;

  if (notrans) {
    // Form y := alpha*A*x + y
    if (incx === 1 && incy === 1) {
      // Hottest path: both arrays contiguous
      for (let j = 0; j < n; j++) {
        const temp = alpha * x[xOff + j];
        const aCol = aOff + j * lda;
        for (let i = 0; i < m; i++) {
          y[yOff + i] += temp * a[aCol + i];
        }
      }
    } else if (incy === 1) {
      let jx = kx;
      for (let j = 0; j < n; j++) {
        const temp = alpha * x[xOff + jx];
        const aCol = aOff + j * lda;
        for (let i = 0; i < m; i++) {
          y[yOff + i] += temp * a[aCol + i];
        }
        jx += incx;
      }
    } else {
      let jx = kx;
      for (let j = 0; j < n; j++) {
        const temp = alpha * x[xOff + jx];
        const aCol = aOff + j * lda;
        let iy = ky;
        for (let i = 0; i < m; i++) {
          y[yOff + iy] += temp * a[aCol + i];
          iy += incy;
        }
        jx += incx;
      }
    }
  } else {
    // Form y := alpha*A**T*x + y
    if (incx === 1 && incy === 1) {
      // Hottest path: both arrays contiguous
      for (let j = 0; j < n; j++) {
        let temp = 0.0;
        const aCol = aOff + j * lda;
        for (let i = 0; i < m; i++) {
          temp += a[aCol + i] * x[xOff + i];
        }
        y[yOff + j] += alpha * temp;
      }
    } else if (incx === 1) {
      let jy = ky;
      for (let j = 0; j < n; j++) {
        let temp = 0.0;
        const aCol = aOff + j * lda;
        for (let i = 0; i < m; i++) {
          temp += a[aCol + i] * x[xOff + i];
        }
        y[yOff + jy] += alpha * temp;
        jy += incy;
      }
    } else {
      let jy = ky;
      for (let j = 0; j < n; j++) {
        let temp = 0.0;
        const aCol = aOff + j * lda;
        let ix = kx;
        for (let i = 0; i < m; i++) {
          temp += a[aCol + i] * x[xOff + ix];
          ix += incx;
        }
        y[yOff + jy] += alpha * temp;
        jy += incy;
      }
    }
  }
}
