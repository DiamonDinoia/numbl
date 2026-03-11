// Translated from LAPACK/SRC/dgelqf.f
// DGELQF computes an LQ factorization of a real M-by-N matrix A:
//
//    A = ( L 0 ) * Q
//
// where Q is a N-by-N orthogonal matrix and L is a lower-triangular
// M-by-M matrix (lower triangular if M <= N).
//
// The matrix Q is represented as a product of elementary reflectors
//    Q = H(k) . . . H(2) H(1), where k = min(m,n).
// Each H(i) has the form H(i) = I - tau * v * v**T where tau is a real
// scalar and v is a real vector with v(1:i-1) = 0 and v(i) = 1;
// v(i+1:n) is stored on exit in A(i,i+1:n), and tau in TAU(i).
//
// This is the blocked (Level 3 BLAS) version that calls dlarft/dlarfb
// for blocks, falling back to the unblocked dgelq2 when the block size
// is too small.
//
// Array indexing convention (column-major, matching Fortran):
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
//   work    - Float64Array workspace of dimension max(1, lwork)
//   workOff - offset into work for WORK(1)
//   lwork   - dimension of work array; if lwork=-1, workspace query
//
// Returns INFO:
//   = 0: successful exit
//   < 0: if INFO = -i, the i-th argument had an illegal value

import { dgelq2 } from "./dgelq2.js";
import { dlarft } from "./dlarft.js";
import { dlarfb } from "./dlarfb.js";
import { ilaenv } from "../utils/ilaenv.js";
import { xerbla } from "../utils/xerbla.js";
import { RIGHT, NOTRANS } from "../utils/constants.js";

// Constants for dlarft/dlarfb direction and storage
const FORWARD = 0;
const ROWWISE = 1; // LQ stores reflectors in rows (unlike QR which uses COLUMNWISE=0)

export function dgelqf(
  m: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): number {
  // Test the input arguments
  let info = 0;
  const k = Math.min(m, n);
  let nb = ilaenv(1, "DGELQF", " ", m, n, -1, -1);
  const lquery = lwork === -1;

  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, m)) {
    info = -4;
  } else if (!lquery) {
    if (lwork <= 0 || (n > 0 && lwork < Math.max(1, m))) {
      info = -7;
    }
  }

  if (info !== 0) {
    xerbla("DGELQF", -info);
    return info;
  } else if (lquery) {
    let lwkopt: number;
    if (k === 0) {
      lwkopt = 1;
    } else {
      lwkopt = m * nb;
    }
    work[workOff] = lwkopt;
    return 0;
  }

  // Quick return if possible
  if (k === 0) {
    work[workOff] = 1;
    return 0;
  }

  let nbmin = 2;
  let nx = 0;
  let iws = m;
  let ldwork = m;

  if (nb > 1 && nb < k) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Math.max(0, ilaenv(3, "DGELQF", " ", m, n, -1, -1));
    if (nx < k) {
      // Determine if workspace is large enough for blocked code.
      ldwork = m;
      iws = ldwork * nb;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: reduce NB and
        // determine the minimum value of NB.
        nb = Math.floor(lwork / ldwork);
        nbmin = Math.max(2, ilaenv(2, "DGELQF", " ", m, n, -1, -1));
      }
    }
  }

  let i = 1;

  if (nb >= nbmin && nb < k && nx < k) {
    // Use blocked code initially
    for (i = 1; i <= k - nx; i += nb) {
      const ib = Math.min(k - i + 1, nb);

      // Compute the LQ factorization of the current block
      // A(i:i+ib-1, i:n)
      dgelq2(
        ib,
        n - i + 1,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda,
        tau,
        tauOff + (i - 1),
        work,
        workOff
      );

      if (i + ib <= m) {
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) ... H(i+ib-1)
        // T is stored at work[workOff], ldt = ldwork
        dlarft(
          FORWARD,
          ROWWISE,
          n - i + 1,
          ib,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          lda,
          tau,
          tauOff + (i - 1),
          work,
          workOff,
          ldwork
        );

        // Apply H to A(i+ib:m, i:n) from the right
        // dlarfb workspace is at work[workOff + ib] (i.e., WORK(IB+1) in Fortran 1-based)
        dlarfb(
          RIGHT,
          NOTRANS,
          FORWARD,
          ROWWISE,
          m - i - ib + 1,
          n - i + 1,
          ib,
          a,
          aOff + (i - 1) + (i - 1) * lda,
          lda,
          work,
          workOff,
          ldwork,
          a,
          aOff + (i + ib - 1) + (i - 1) * lda,
          lda,
          work,
          workOff + ib,
          ldwork
        );
      }
    }
  } else {
    i = 1;
  }

  // Use unblocked code to factor the last or only block.
  if (i <= k) {
    dgelq2(
      m - i + 1,
      n - i + 1,
      a,
      aOff + (i - 1) + (i - 1) * lda,
      lda,
      tau,
      tauOff + (i - 1),
      work,
      workOff
    );
  }

  work[workOff] = iws;
  return 0;
}
