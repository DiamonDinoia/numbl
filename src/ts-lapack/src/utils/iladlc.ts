// Translated from SRC/iladlc.f
// ILADLC scans A for its last non-zero column.
// Returns the 1-based column index of the last non-zero column, or 0 if A is all zero.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)

export function iladlc(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number
): number {
  // Quick test for the common case where one corner is non-zero
  if (n === 0) {
    return n;
  } else if (
    a[aOff + (n - 1) * lda] !== 0.0 || // A(1,N)
    a[aOff + (m - 1) + (n - 1) * lda] !== 0.0 // A(M,N)
  ) {
    return n;
  }

  // Scan each column from the end, returning with the first non-zero
  for (let col = n; col >= 1; col--) {
    for (let i = 1; i <= m; i++) {
      if (a[aOff + (i - 1) + (col - 1) * lda] !== 0.0) {
        return col;
      }
    }
  }
  return 0;
}
