// Translated from BLAS/SRC/ddot.f
// DDOT forms the dot product of two vectors.
//   DDOT := DX' * DY
// Uses unrolled loops for increments equal to one.
//
// Indexing convention (matching Fortran):
//   DX(I)  =>  dx[dxOff + (I-1)*incx]   (I is 1-based)
//   DY(I)  =>  dy[dyOff + (I-1)*incy]   (I is 1-based)

export function ddot(
  n: number,
  dx: Float64Array,
  dxOff: number,
  incx: number,
  dy: Float64Array,
  dyOff: number,
  incy: number
): number {
  let dtemp = 0.0;
  if (n <= 0) return 0.0;

  if (incx === 1 && incy === 1) {
    // Code for both increments equal to 1 (unrolled loop of 5)
    const m = n % 5;
    if (m !== 0) {
      for (let i = 1; i <= m; i++) {
        dtemp += dx[dxOff + (i - 1)] * dy[dyOff + (i - 1)];
      }
      if (n < 5) return dtemp;
    }
    const mp1 = m + 1;
    for (let i = mp1; i <= n; i += 5) {
      dtemp +=
        dx[dxOff + (i - 1)] * dy[dyOff + (i - 1)] +
        dx[dxOff + i] * dy[dyOff + i] +
        dx[dxOff + (i + 1)] * dy[dyOff + (i + 1)] +
        dx[dxOff + (i + 2)] * dy[dyOff + (i + 2)] +
        dx[dxOff + (i + 3)] * dy[dyOff + (i + 3)];
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = 1;
    let iy = 1;
    if (incx < 0) ix = (-n + 1) * incx + 1;
    if (incy < 0) iy = (-n + 1) * incy + 1;
    for (let i = 1; i <= n; i++) {
      dtemp += dx[dxOff + (ix - 1)] * dy[dyOff + (iy - 1)];
      ix += incx;
      iy += incy;
    }
  }
  return dtemp;
}
