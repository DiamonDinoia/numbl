// Translated from BLAS/SRC/drot.f
// DROT applies a plane rotation.
//   DX(I) :=  C*DX(I) + S*DY(I)
//   DY(I) :=  C*DY(I) - S*DX(I)
//
// Indexing convention (matching Fortran):
//   DX(I)  =>  dx[dxOff + (I-1)*incx]   (I is 1-based)
//   DY(I)  =>  dy[dyOff + (I-1)*incy]   (I is 1-based)

export function drot(
  n: number,
  dx: Float64Array,
  dxOff: number,
  incx: number,
  dy: Float64Array,
  dyOff: number,
  incy: number,
  c: number,
  s: number
): void {
  if (n <= 0) return;

  if (incx === 1 && incy === 1) {
    // Code for both increments equal to 1
    for (let i = 1; i <= n; i++) {
      const dtemp = c * dx[dxOff + (i - 1)] + s * dy[dyOff + (i - 1)];
      dy[dyOff + (i - 1)] = c * dy[dyOff + (i - 1)] - s * dx[dxOff + (i - 1)];
      dx[dxOff + (i - 1)] = dtemp;
    }
  } else {
    // Code for unequal increments or equal increments not equal to 1
    let ix = 1;
    let iy = 1;
    if (incx < 0) ix = (-n + 1) * incx + 1;
    if (incy < 0) iy = (-n + 1) * incy + 1;
    for (let i = 1; i <= n; i++) {
      const dtemp = c * dx[dxOff + (ix - 1)] + s * dy[dyOff + (iy - 1)];
      dy[dyOff + (iy - 1)] =
        c * dy[dyOff + (iy - 1)] - s * dx[dxOff + (ix - 1)];
      dx[dxOff + (ix - 1)] = dtemp;
      ix += incx;
      iy += incy;
    }
  }
}
