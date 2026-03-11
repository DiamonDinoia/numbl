// Translated from SRC/dorg2r.f
// DORG2R generates an m by n real matrix Q with orthonormal columns,
// which is defined as the first n columns of a product of k elementary
// reflectors of order m:
//
//       Q = H(1) H(2) . . . H(k)
//
// as returned by DGEQRF.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)    =>  tau[tauOff + (I-1)]              (I is 1-based)
//   WORK(I)   =>  work[workOff + (I-1)]            (I is 1-based)
//
// Parameters:
//   m       - number of rows of Q (>= 0)
//   n       - number of columns of Q (m >= n >= 0)
//   k       - number of elementary reflectors (n >= k >= 0)
//   a       - Float64Array; on entry contains reflector vectors; on exit is Q
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,m))
//   tau     - Float64Array of length k; scalar factors of the reflectors
//   tauOff  - offset into tau for TAU(1)
//   work    - Float64Array workspace of length n
//   workOff - offset into work for WORK(1)
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dlarf1f } from "./dlarf1f.js";
import { dscal } from "../BLAS/dscal.js";
import { xerbla } from "../utils/xerbla.js";
import { LEFT } from "../utils/constants.js";

export function dorg2r(
  m: number,
  n: number,
  k: number,
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
  } else if (n < 0 || n > m) {
    info = -2;
  } else if (k < 0 || k > n) {
    info = -3;
  } else if (lda < Math.max(1, m)) {
    info = -5;
  }
  if (info !== 0) {
    xerbla("DORG2R", -info);
    return info;
  }

  // Quick return if possible
  if (n <= 0) return 0;

  // Initialise columns k+1:n to columns of the unit matrix
  for (let j = k + 1; j <= n; j++) {
    for (let l = 1; l <= m; l++) {
      a[aOff + (l - 1) + (j - 1) * lda] = 0.0;
    }
    a[aOff + (j - 1) + (j - 1) * lda] = 1.0;
  }

  // Apply reflectors in reverse order
  for (let i = k; i >= 1; i--) {
    // A(I,I) is at aOff + (I-1) + (I-1)*lda
    const aIIOff = aOff + (i - 1) + (i - 1) * lda;
    const tauI = tau[tauOff + (i - 1)];

    if (i < n) {
      // Apply H(i) to A(i:m,i+1:n) from the left
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

    if (i < m) {
      // DSCAL( M-I, -TAU(I), A(I+1,I), 1 )
      // A(I+1,I) is at aOff + I + (I-1)*lda
      dscal(m - i, -tauI, a, aOff + i + (i - 1) * lda, 1);
    }

    // A(I,I) = ONE - TAU(I)
    a[aIIOff] = 1.0 - tauI;

    // Set A(1:I-1,I) to zero
    for (let l = 1; l <= i - 1; l++) {
      a[aOff + (l - 1) + (i - 1) * lda] = 0.0;
    }
  }

  return 0;
}
