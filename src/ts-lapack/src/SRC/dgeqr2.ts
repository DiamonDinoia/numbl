// Translated from SRC/dgeqr2.f
// DGEQR2 computes a QR factorization of a real m-by-n matrix A:
//
//    A = Q * ( R ),
//            ( 0 )
//
// where Q is a m-by-m orthogonal matrix and R is an upper-triangular
// min(m,n)-by-n matrix (upper triangular if m >= n).
//
// The matrix Q is represented as a product of elementary reflectors
//    Q = H(1) H(2) . . . H(k), where k = min(m,n).
// Each H(i) has the form H(i) = I - tau * v * v**T where tau is a real
// scalar and v is a real vector with v(1:i-1) = 0 and v(i) = 1;
// v(i+1:m) is stored on exit in A(i+1:m,i), and tau in TAU(i).
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)    =>  tau[tauOff + (I-1)]              (I is 1-based)
//   WORK(I)   =>  work[workOff + (I-1)]            (I is 1-based)
//
// Parameters:
//   m       - number of rows    (>= 0)
//   n       - number of columns (>= 0)
//   a       - Float64Array of the matrix (modified in place)
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,m))
//   tau     - Float64Array of length min(m,n); output scalar factors
//   tauOff  - offset into tau for TAU(1)
//   work    - Float64Array workspace of length n
//   workOff - offset into work for WORK(1)
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dlarfg } from "./dlarfg.js";
import { dlarf1f } from "./dlarf1f.js";
import { xerbla } from "../utils/xerbla.js";
import { LEFT } from "../utils/constants.js";

export function dgeqr2(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number,
  work: Float64Array,
  workOff: number
): number {
  // Test the input arguments
  let info = 0;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, m)) {
    info = -4;
  }
  if (info !== 0) {
    xerbla("DGEQR2", -info);
    return info;
  }

  const k = Math.min(m, n);

  for (let i = 1; i <= k; i++) {
    // Generate elementary reflector H(i) to annihilate A(i+1:m,i)
    // DLARFG( M-I+1, A(I,I), A(MIN(I+1,M),I), 1, TAU(I) )
    // A(I,I) is at aOff + (I-1) + (I-1)*lda
    // A(MIN(I+1,M),I) is at aOff + MIN(I,M-1) + (I-1)*lda
    const aIIOff = aOff + (i - 1) + (i - 1) * lda;
    const aXOff = aOff + Math.min(i, m - 1) + (i - 1) * lda;
    const { alpha: beta, tau: tauI } = dlarfg(
      m - i + 1,
      a[aIIOff],
      a,
      aXOff,
      1
    );
    a[aIIOff] = beta;
    tau[tauOff + (i - 1)] = tauI;

    if (i < n) {
      // Apply H(i) to A(i:m, i+1:n) from the left
      // DLARF1F('Left', M-I+1, N-I, A(I,I), 1, TAU(I), A(I,I+1), LDA, WORK)
      // A(I,I+1) is at aOff + (I-1) + I*lda
      dlarf1f(
        LEFT,
        m - i + 1,
        n - i,
        a,
        aIIOff,
        1,
        tauI,
        a,
        aOff + (i - 1) + i * lda,
        lda,
        work,
        workOff
      );
    }
  }

  return 0;
}
