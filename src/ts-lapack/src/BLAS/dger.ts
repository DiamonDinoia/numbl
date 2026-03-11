// Translated from BLAS/SRC/dger.f
// DGER performs the rank 1 operation
//   A := alpha*x*y**T + A
// where alpha is a scalar, x is an m element vector, y is an n element
// vector and A is an m by n matrix.
//
// Optimizations vs. the Fortran reference implementation:
//
//   1. 0-based indexing: eliminates (i-1), (j-1), (jy-1), (ix-1) subtractions
//      from every inner-loop array access.
//
//   2. Column offset hoisted out of inner loop: aCol = aOff + j*lda is
//      computed once per outer iteration; the inner loop only adds i.
//
//   3. Tight contiguous fast path: the incx===1 && incy===1 case (by far the
//      most common in practice) gets its own inner loop with no stride
//      arithmetic, allowing V8 to apply loop optimizations more aggressively.
//
// Array indexing convention (column-major, 0-based offsets):
//   A(i,j)  =>  a[aOff + i + j*lda]   (i,j are 0-based)
//   X(i)    =>  x[xOff + i*incx]       (i is 0-based)
//   Y(j)    =>  y[yOff + j*incy]       (j is 0-based)

import { xerbla } from "../utils/xerbla.js";

export function dger(
  m: number,
  n: number,
  alpha: number,
  x: Float64Array,
  xOff: number,
  incx: number,
  y: Float64Array,
  yOff: number,
  incy: number,
  a: Float64Array,
  aOff: number,
  lda: number
): void {
  // Test the input parameters
  let info = 0;
  if (m < 0) {
    info = 1;
  } else if (n < 0) {
    info = 2;
  } else if (incx === 0) {
    info = 5;
  } else if (incy === 0) {
    info = 7;
  } else if (lda < Math.max(1, m)) {
    info = 9;
  }
  if (info !== 0) {
    xerbla("DGER  ", info);
    return;
  }

  // Quick return if possible
  if (m === 0 || n === 0 || alpha === 0.0) return;

  // 0-based start points (for negative strides, start at the far end)
  const kx = incx > 0 ? 0 : (m - 1) * -incx;
  const ky = incy > 0 ? 0 : (n - 1) * -incy;

  if (incx === 1 && incy === 1) {
    // Hottest path: both arrays contiguous
    for (let j = 0; j < n; j++) {
      const yj = y[yOff + j];
      if (yj !== 0.0) {
        const temp = alpha * yj;
        const aCol = aOff + j * lda;
        for (let i = 0; i < m; i++) {
          a[aCol + i] += x[xOff + i] * temp;
        }
      }
    }
  } else if (incx === 1) {
    let jy = ky;
    for (let j = 0; j < n; j++) {
      const yj = y[yOff + jy];
      if (yj !== 0.0) {
        const temp = alpha * yj;
        const aCol = aOff + j * lda;
        for (let i = 0; i < m; i++) {
          a[aCol + i] += x[xOff + i] * temp;
        }
      }
      jy += incy;
    }
  } else {
    let jy = ky;
    for (let j = 0; j < n; j++) {
      const yj = y[yOff + jy];
      if (yj !== 0.0) {
        const temp = alpha * yj;
        const aCol = aOff + j * lda;
        let ix = kx;
        for (let i = 0; i < m; i++) {
          a[aCol + i] += x[xOff + ix] * temp;
          ix += incx;
        }
      }
      jy += incy;
    }
  }
}
