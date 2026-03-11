// Translated from BLAS/SRC/dcopy.f
// DCOPY copies a vector, x, to a vector, y.
//   DY := DX
// Uses unrolled loops for increments equal to one.
//
// Indexing convention (matching Fortran):
//   DX(I)  =>  dx[dxOff + (I-1)*incx]   (I is 1-based)
//   DY(I)  =>  dy[dyOff + (I-1)*incy]   (I is 1-based)

export function dcopy(
  n: number,
  dx: Float64Array,
  dxOff: number,
  incx: number,
  dy: Float64Array,
  dyOff: number,
  incy: number
): void {
  if (n <= 0) return;

  if (incx === 1 && incy === 1) {
    // Code for both increments equal to 1 (unrolled loop of 7)
    const m = n % 7;
    if (m !== 0) {
      for (let i = 1; i <= m; i++) {
        dy[dyOff + (i - 1)] = dx[dxOff + (i - 1)];
      }
      if (n < 7) return;
    }
    const mp1 = m + 1;
    for (let i = mp1; i <= n; i += 7) {
      dy[dyOff + (i - 1)] = dx[dxOff + (i - 1)];
      dy[dyOff + i] = dx[dxOff + i];
      dy[dyOff + (i + 1)] = dx[dxOff + (i + 1)];
      dy[dyOff + (i + 2)] = dx[dxOff + (i + 2)];
      dy[dyOff + (i + 3)] = dx[dxOff + (i + 3)];
      dy[dyOff + (i + 4)] = dx[dxOff + (i + 4)];
      dy[dyOff + (i + 5)] = dx[dxOff + (i + 5)];
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = 1;
    let iy = 1;
    if (incx < 0) ix = (-n + 1) * incx + 1;
    if (incy < 0) iy = (-n + 1) * incy + 1;
    for (let i = 1; i <= n; i++) {
      dy[dyOff + (iy - 1)] = dx[dxOff + (ix - 1)];
      ix += incx;
      iy += incy;
    }
  }
}
