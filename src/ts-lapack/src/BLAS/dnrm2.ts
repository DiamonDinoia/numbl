// Translated from BLAS reference implementation of DNRM2.
// (No Fortran source available in the local lapack directory; this follows
//  the standard BLAS reference algorithm with scaled accumulation.)
//
// DNRM2 returns the Euclidean norm of a vector via the function
// name, so that DNRM2 := sqrt( x'*x ).
//
// Indexing convention (matching Fortran):
//   X(I)  =>  x[xOff + (I-1)*incx]   (I is 1-based)

export function dnrm2(
  n: number,
  x: Float64Array,
  xOff: number,
  incx: number
): number {
  if (n <= 0 || incx <= 0) return 0.0;
  if (n === 1) return Math.abs(x[xOff]);

  // Use scaled accumulation to avoid overflow/underflow
  let scale = 0.0;
  let ssq = 1.0;

  for (let i = 0; i < n; i++) {
    const absxi = Math.abs(x[xOff + i * incx]);
    if (absxi > 0.0) {
      if (scale < absxi) {
        ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
        scale = absxi;
      } else {
        ssq += (absxi / scale) * (absxi / scale);
      }
    }
  }
  return scale * Math.sqrt(ssq);
}
