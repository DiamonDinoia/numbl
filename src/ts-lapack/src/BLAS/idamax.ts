// Translated from BLAS/SRC/idamax.f
// IDAMAX finds the index of the first element having maximum absolute value.
//
// Indexing convention (matching Fortran):
//   DX(I)  =>  dx[dxOff + (I-1)*incx]   (I is 1-based)
//
// Returns a 1-based index (0 if n < 1 or incx <= 0), matching Fortran behaviour.

export function idamax(
  n: number,
  dx: Float64Array,
  dxOff: number,
  incx: number
): number {
  if (n < 1 || incx <= 0) return 0;
  if (n === 1) return 1;

  let result = 1;

  if (incx === 1) {
    // Code for increment equal to 1
    let dmax = Math.abs(dx[dxOff]);
    for (let i = 2; i <= n; i++) {
      if (Math.abs(dx[dxOff + (i - 1)]) > dmax) {
        result = i;
        dmax = Math.abs(dx[dxOff + (i - 1)]);
      }
    }
  } else {
    // Code for increment not equal to 1
    let ix = 0; // 0-based position of DX(1)
    let dmax = Math.abs(dx[dxOff]);
    ix += incx;
    for (let i = 2; i <= n; i++) {
      if (Math.abs(dx[dxOff + ix]) > dmax) {
        result = i;
        dmax = Math.abs(dx[dxOff + ix]);
      }
      ix += incx;
    }
  }

  return result;
}
