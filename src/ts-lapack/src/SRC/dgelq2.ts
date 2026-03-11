// Translated from LAPACK/SRC/dgelq2.f
// DGELQ2 computes an LQ factorization of a real m-by-n matrix A:
//
//    A = ( L 0 ) * Q
//
// where Q is a n-by-n orthogonal matrix and L is a lower-triangular
// m-by-m matrix (lower triangular if m <= n).
//
// The matrix Q is represented as a product of elementary reflectors
//    Q = H(k) . . . H(2) H(1), where k = min(m,n).
// Each H(i) has the form H(i) = I - tau * v * v**T where tau is a real
// scalar and v is a real vector with v(1:i-1) = 0 and v(i) = 1;
// v(i+1:n) is stored on exit in A(i,i+1:n), and tau in TAU(i).
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
//   work    - Float64Array workspace of length m
//   workOff - offset into work for WORK(1)
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dlarfg } from "./dlarfg.js";
import { dlarf1f } from "./dlarf1f.js";
import { xerbla } from "../utils/xerbla.js";
import { RIGHT } from "../utils/constants.js";

export function dgelq2(
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
    xerbla("DGELQ2", -info);
    return info;
  }

  const k = Math.min(m, n);

  for (let i = 1; i <= k; i++) {
    // Generate elementary reflector H(i) to annihilate A(i,i+1:n)
    // DLARFG( N-I+1, A(I,I), A(I,MIN(I+1,N)), LDA, TAU(I) )
    // A(I,I) is at aOff + (I-1) + (I-1)*lda
    // A(I,MIN(I+1,N)) is at aOff + (I-1) + MIN(I,N-1)*lda
    const aIIOff = aOff + (i - 1) + (i - 1) * lda;
    const aXOff = aOff + (i - 1) + Math.min(i, n - 1) * lda;
    const { alpha: beta, tau: tauI } = dlarfg(
      n - i + 1,
      a[aIIOff],
      a,
      aXOff,
      lda
    );
    a[aIIOff] = beta;
    tau[tauOff + (i - 1)] = tauI;

    if (i < m) {
      // Apply H(i) to A(i+1:m, i:n) from the right
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
  }

  return 0;
}
