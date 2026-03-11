// Translated from BLAS/SRC/dswap.f
// DSWAP interchanges two vectors.
//
// Indexing convention (matching Fortran):
//   DX(I)  =>  dx[dxOff + (I-1)*incx]   (I is 1-based)
//   DY(I)  =>  dy[dyOff + (I-1)*incy]   (I is 1-based)

export function dswap(
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
    // Code for both increments equal to 1 (unrolled loop of 3)
    const m = n % 3;
    if (m !== 0) {
      for (let i = 1; i <= m; i++) {
        const dtemp = dx[dxOff + (i - 1)];
        dx[dxOff + (i - 1)] = dy[dyOff + (i - 1)];
        dy[dyOff + (i - 1)] = dtemp;
      }
      if (n < 3) return;
    }
    const mp1 = m + 1;
    for (let i = mp1; i <= n; i += 3) {
      let dtemp = dx[dxOff + (i - 1)];
      dx[dxOff + (i - 1)] = dy[dyOff + (i - 1)];
      dy[dyOff + (i - 1)] = dtemp;

      dtemp = dx[dxOff + i];
      dx[dxOff + i] = dy[dyOff + i];
      dy[dyOff + i] = dtemp;

      dtemp = dx[dxOff + (i + 1)];
      dx[dxOff + (i + 1)] = dy[dyOff + (i + 1)];
      dy[dyOff + (i + 1)] = dtemp;
    }
  } else {
    // Code for unequal increments or increments not equal to 1
    let ix = incx >= 0 ? 1 : (-n + 1) * incx + 1;
    let iy = incy >= 0 ? 1 : (-n + 1) * incy + 1;
    for (let i = 1; i <= n; i++) {
      const dtemp = dx[dxOff + (ix - 1)];
      dx[dxOff + (ix - 1)] = dy[dyOff + (iy - 1)];
      dy[dyOff + (iy - 1)] = dtemp;
      ix += incx;
      iy += incy;
    }
  }
}
