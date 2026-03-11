// Translated from LAPACK/SRC/dgebd2.f
// DGEBD2 reduces a real general m by n matrix A to upper or lower
// bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   D(I)      =>  d[dOff + (I-1)]                  (I is 1-based)
//   E(I)      =>  e[eOff + (I-1)]                  (I is 1-based)
//   TAUQ(I)   =>  tauq[tauqOff + (I-1)]            (I is 1-based)
//   TAUP(I)   =>  taup[taupOff + (I-1)]            (I is 1-based)
//   WORK(I)   =>  work[workOff + (I-1)]             (I is 1-based)
//
// Parameters:
//   m        - number of rows    (>= 0)
//   n        - number of columns (>= 0)
//   a        - Float64Array of the matrix (modified in place)
//   aOff     - offset into a for A(1,1)
//   lda      - leading dimension of a (>= max(1,m))
//   d        - Float64Array of length min(m,n); diagonal elements of B
//   dOff     - offset into d for D(1)
//   e        - Float64Array of length min(m,n)-1; off-diagonal elements of B
//   eOff     - offset into e for E(1)
//   tauq     - Float64Array of length min(m,n); scalar factors for Q reflectors
//   tauqOff  - offset into tauq for TAUQ(1)
//   taup     - Float64Array of length min(m,n); scalar factors for P reflectors
//   taupOff  - offset into taup for TAUP(1)
//   work     - Float64Array workspace of length max(m,n)
//   workOff  - offset into work for WORK(1)
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dlarfg } from "./dlarfg.js";
import { dlarf1f } from "./dlarf1f.js";
import { xerbla } from "../utils/xerbla.js";
import { LEFT, RIGHT } from "../utils/constants.js";

export function dgebd2(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  d: Float64Array,
  dOff: number,
  e: Float64Array,
  eOff: number,
  tauq: Float64Array,
  tauqOff: number,
  taup: Float64Array,
  taupOff: number,
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
    xerbla("DGEBD2", -info);
    return info;
  }

  if (m >= n) {
    //
    // Reduce to upper bidiagonal form
    //
    for (let i = 1; i <= n; i++) {
      // Generate elementary reflector H(i) to annihilate A(i+1:m,i)
      // DLARFG( M-I+1, A(I,I), A(MIN(I+1,M),I), 1, TAUQ(I) )
      const aIIOff = aOff + (i - 1) + (i - 1) * lda;
      const aColOff = aOff + Math.min(i, m - 1) + (i - 1) * lda;
      const { alpha: alphaQ, tau: tauqI } = dlarfg(
        m - i + 1,
        a[aIIOff],
        a,
        aColOff,
        1
      );
      a[aIIOff] = alphaQ;
      tauq[tauqOff + (i - 1)] = tauqI;
      d[dOff + (i - 1)] = a[aIIOff];

      // Apply H(i) to A(i:m, i+1:n) from the left
      if (i < n) {
        // DLARF1F('Left', M-I+1, N-I, A(I,I), 1, TAUQ(I), A(I,I+1), LDA, WORK)
        dlarf1f(
          LEFT,
          m - i + 1,
          n - i,
          a,
          aIIOff,
          1,
          tauqI,
          a,
          aOff + (i - 1) + i * lda,
          lda,
          work,
          workOff
        );
      }

      if (i < n) {
        // Generate elementary reflector G(i) to annihilate A(i,i+2:n)
        // DLARFG( N-I, A(I,I+1), A(I,MIN(I+2,N)), LDA, TAUP(I) )
        const aRowOff = aOff + (i - 1) + i * lda; // A(I, I+1)
        const aRowXOff = aOff + (i - 1) + Math.min(i + 1, n - 1) * lda; // A(I, MIN(I+2,N))
        const { alpha: alphaP, tau: taupI } = dlarfg(
          n - i,
          a[aRowOff],
          a,
          aRowXOff,
          lda
        );
        a[aRowOff] = alphaP;
        taup[taupOff + (i - 1)] = taupI;
        e[eOff + (i - 1)] = a[aRowOff];

        // Apply G(i) to A(i+1:m, i+1:n) from the right
        // DLARF1F('Right', M-I, N-I, A(I,I+1), LDA, TAUP(I), A(I+1,I+1), LDA, WORK)
        dlarf1f(
          RIGHT,
          m - i,
          n - i,
          a,
          aRowOff,
          lda,
          taupI,
          a,
          aOff + i + i * lda,
          lda,
          work,
          workOff
        );
      } else {
        taup[taupOff + (i - 1)] = 0.0;
      }
    }
  } else {
    //
    // Reduce to lower bidiagonal form
    //
    for (let i = 1; i <= m; i++) {
      // Generate elementary reflector G(i) to annihilate A(i,i+1:n)
      // DLARFG( N-I+1, A(I,I), A(I,MIN(I+1,N)), LDA, TAUP(I) )
      const aIIOff = aOff + (i - 1) + (i - 1) * lda;
      const aRowOff = aOff + (i - 1) + Math.min(i, n - 1) * lda; // A(I, MIN(I+1,N))
      const { alpha: alphaP, tau: taupI } = dlarfg(
        n - i + 1,
        a[aIIOff],
        a,
        aRowOff,
        lda
      );
      a[aIIOff] = alphaP;
      taup[taupOff + (i - 1)] = taupI;
      d[dOff + (i - 1)] = a[aIIOff];

      // Apply G(i) to A(i+1:m, i:n) from the right
      if (i < m) {
        // DLARF1F('Right', M-I, N-I+1, A(I,I), LDA, TAUP(I), A(I+1,I), LDA, WORK)
        dlarf1f(
          RIGHT,
          m - i,
          n - i + 1,
          a,
          aIIOff,
          lda,
          taupI,
          a,
          aOff + i + (i - 1) * lda,
          lda,
          work,
          workOff
        );
      }

      if (i < m) {
        // Generate elementary reflector H(i) to annihilate A(i+2:m,i)
        // DLARFG( M-I, A(I+1,I), A(MIN(I+2,M),I), 1, TAUQ(I) )
        const aColOff2 = aOff + i + (i - 1) * lda; // A(I+1, I)
        const aColXOff = aOff + Math.min(i + 1, m - 1) + (i - 1) * lda; // A(MIN(I+2,M), I)
        const { alpha: alphaQ, tau: tauqI } = dlarfg(
          m - i,
          a[aColOff2],
          a,
          aColXOff,
          1
        );
        a[aColOff2] = alphaQ;
        tauq[tauqOff + (i - 1)] = tauqI;
        e[eOff + (i - 1)] = a[aColOff2];

        // Apply H(i) to A(i+1:m, i+1:n) from the left
        // DLARF1F('Left', M-I, N-I, A(I+1,I), 1, TAUQ(I), A(I+1,I+1), LDA, WORK)
        dlarf1f(
          LEFT,
          m - i,
          n - i,
          a,
          aColOff2,
          1,
          tauqI,
          a,
          aOff + i + i * lda,
          lda,
          work,
          workOff
        );
      } else {
        tauq[tauqOff + (i - 1)] = 0.0;
      }
    }
  }

  return 0;
}
