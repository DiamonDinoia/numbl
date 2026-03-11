// Translated from BLAS/SRC/dscal.f
// DSCAL scales a vector by a constant.
//
// Indexing convention (matching Fortran):
//   DX(I)  =>  dx[dxOff + (I-1)*incx]   (I is 1-based)

export function dscal(
  n: number,
  da: number,
  dx: Float64Array,
  dxOff: number,
  incx: number
): void {
  if (n <= 0 || incx <= 0 || da === 1.0) return;

  if (incx === 1) {
    // Code for increment equal to 1 (unrolled loop of 5)
    const m = n % 5;
    if (m !== 0) {
      for (let i = 1; i <= m; i++) {
        dx[dxOff + (i - 1)] = da * dx[dxOff + (i - 1)];
      }
      if (n < 5) return;
    }
    const mp1 = m + 1;
    for (let i = mp1; i <= n; i += 5) {
      dx[dxOff + (i - 1)] = da * dx[dxOff + (i - 1)];
      dx[dxOff + i] = da * dx[dxOff + i];
      dx[dxOff + (i + 1)] = da * dx[dxOff + (i + 1)];
      dx[dxOff + (i + 2)] = da * dx[dxOff + (i + 2)];
      dx[dxOff + (i + 3)] = da * dx[dxOff + (i + 3)];
    }
  } else {
    // Code for increment not equal to 1
    const nincx = n * incx;
    for (let i = 1; i <= nincx; i += incx) {
      dx[dxOff + (i - 1)] = da * dx[dxOff + (i - 1)];
    }
  }
}
