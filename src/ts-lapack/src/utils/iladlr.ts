// Translated from SRC/iladlr.f
// ILADLR scans A for its last non-zero row.
// Returns the 1-based row index of the last non-zero row, or 0 if A is all zero.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)

export function iladlr(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  // Quick test for the common case where one corner is non-zero
  if (m === 0) {
    return m;
  } else if (
    a[aOff + (m - 1)] !== 0.0 || // A(M,1)
    a[aOff + (m - 1) + (n - 1) * lda] !== 0.0 // A(M,N)
  ) {
    return m;
  }

  // Scan up each column tracking the last non-zero row seen
  let result = 0;
  for (let j = 1; j <= n; j++) {
    let i = m;
    while (i >= 1 && a[aOff + (Math.max(i, 1) - 1) + (j - 1) * lda] === 0.0) {
      i--;
    }
    result = Math.max(result, i);
  }
  return result;
}
