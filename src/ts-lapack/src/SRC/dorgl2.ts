// Translated from LAPACK/SRC/dorgl2.f
// DORGL2 generates an m by n real matrix Q with orthonormal rows,
// which is defined as the first m rows of a product of k elementary
// reflectors of order n:
//
//       Q = H(k) . . . H(2) H(1)
//
// as returned by DGELQF.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)    =>  tau[tauOff + (I-1)]              (I is 1-based)
//   WORK(I)   =>  work[workOff + (I-1)]            (I is 1-based)
//
// Parameters:
//   m       - number of rows of Q (>= 0)
//   n       - number of columns of Q (n >= m >= 0)
//   k       - number of elementary reflectors (m >= k >= 0)
//   a       - Float64Array; on entry contains reflector vectors; on exit is Q
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,m))
//   tau     - Float64Array of length k; scalar factors of the reflectors
//   tauOff  - offset into tau for TAU(1)
//   work    - Float64Array workspace of length m
//   workOff - offset into work for WORK(1)
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dlarf1f } from "./dlarf1f.js";
import { dscal } from "../BLAS/dscal.js";
import { xerbla } from "../utils/xerbla.js";
import { RIGHT } from "../utils/constants.js";

export function dorgl2(
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
  } else if (n < m) {
    info = -2;
  } else if (k < 0 || k > m) {
    info = -3;
  } else if (lda < Math.max(1, m)) {
    info = -5;
  }
  if (info !== 0) {
    xerbla("DORGL2", -info);
    return info;
  }

  // Quick return if possible
  if (m <= 0) return 0;

  if (k < m) {
    // Initialise rows k+1:m to rows of the unit matrix
    for (let j = 1; j <= n; j++) {
      for (let l = k + 1; l <= m; l++) {
        a[aOff + (l - 1) + (j - 1) * lda] = 0.0;
      }
      if (j > k && j <= m) {
        a[aOff + (j - 1) + (j - 1) * lda] = 1.0;
      }
    }
  }

  // Apply reflectors in reverse order (i = k down to 1)
  for (let i = k; i >= 1; i--) {
    // A(I,I) is at aOff + (I-1) + (I-1)*lda
    const aIIOff = aOff + (i - 1) + (i - 1) * lda;
    const tauI = tau[tauOff + (i - 1)];

    if (i < n) {
      if (i < m) {
        // Apply H(i) to A(i+1:m,i:n) from the right
        // DLARF1F('Right', M-I, N-I+1, A(I,I), LDA, TAU(I), A(I+1,I), LDA, WORK)
        // A(I+1,I) is at aOff + I + (I-1)*lda
        dlarf1f(
          RIGHT,
          m - i,
          n - i + 1,
          a,
          aIIOff,
          lda,
          tauI,
          a,
          aOff + i + (i - 1) * lda,
          lda,
          work,
          workOff
        );
      }

      // DSCAL( N-I, -TAU(I), A(I,I+1), LDA )
      // A(I,I+1) is at aOff + (I-1) + I*lda
      dscal(n - i, -tauI, a, aOff + (i - 1) + i * lda, lda);
    }

    // A(I,I) = ONE - TAU(I)
    a[aIIOff] = 1.0 - tauI;

    // Set A(I,1:I-1) to zero
    for (let l = 1; l <= i - 1; l++) {
      a[aOff + (i - 1) + (l - 1) * lda] = 0.0;
    }
  }

  return 0;
}
