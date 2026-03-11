// Translated from LAPACK/SRC/dorglq.f
// DORGLQ generates an M-by-N real matrix Q with orthonormal rows,
// which is defined as the first M rows of a product of K elementary
// reflectors of order N:
//
//       Q = H(k) . . . H(2) H(1)
//
// as returned by DGELQF.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)    =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)    =>  tau[tauOff + (I-1)]              (I is 1-based)
//
// Parameters:
//   m       - number of rows of Q (>= 0)
//   n       - number of columns of Q (n >= m >= 0)
//   k       - number of elementary reflectors (m >= k >= 0)
//   a       - Float64Array; on entry reflector vectors; on exit is Q
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,m))
//   tau     - Float64Array of length k; scalar factors of the reflectors
//   tauOff  - offset into tau for TAU(1)
//   work    - Float64Array workspace, dimension max(1,lwork)
//   workOff - offset into work
//   lwork   - workspace size; if -1, workspace query
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dorgl2 } from "./dorgl2.js";
import { dlarft } from "./dlarft.js";
import { dlarfb } from "./dlarfb.js";
import { ilaenv } from "../utils/ilaenv.js";
import { xerbla } from "../utils/xerbla.js";
import { RIGHT, NOTRANS } from "../utils/constants.js";

// dlarft/dlarfb direction and storage constants (not exported from constants)
const FORWARD = 0;
const ROWWISE = 1;
const TRANS = 1;

export function dorglq(
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): number {
  // suppress unused-import warnings
  void NOTRANS;

  // Test the input arguments
  let info = 0;
  const nb = ilaenv(1, "DORGLQ", " ", m, n, k, -1);
  const lwkopt = Math.max(1, m) * nb;
  work[workOff] = lwkopt;
  const lquery = lwork === -1;

  if (m < 0) {
    info = -1;
  } else if (n < m) {
    info = -2;
  } else if (k < 0 || k > m) {
    info = -3;
  } else if (lda < Math.max(1, m)) {
    info = -5;
  } else if (lwork < Math.max(1, m) && !lquery) {
    info = -8;
  }
  if (info !== 0) {
    xerbla("DORGLQ", -info);
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  if (m <= 0) {
    work[workOff] = 1;
    return 0;
  }

  let nbmin = 2;
  let nx = 0;
  let iws = m;
  let ldwork = m;
  let nbEff = nb;

  if (nbEff > 1 && nbEff < k) {
    // Determine when to cross over from blocked to unblocked code.
    nx = Math.max(0, ilaenv(3, "DORGLQ", " ", m, n, k, -1));
    if (nx < k) {
      // Determine if workspace is large enough for blocked code.
      ldwork = m;
      iws = ldwork * nbEff;
      if (lwork < iws) {
        // Not enough workspace to use optimal NB: reduce NB and
        // determine the minimum value of NB.
        nbEff = Math.floor(lwork / ldwork);
        nbmin = Math.max(2, ilaenv(2, "DORGLQ", " ", m, n, k, -1));
      }
    }
  }

  let kk = 0;
  let ki = 0;

  if (nbEff >= nbmin && nbEff < k && nx < k) {
    // Use blocked code after the last block.
    // The first kk rows are handled by the block method.
    ki = Math.floor((k - nx - 1) / nbEff) * nbEff;
    kk = Math.min(k, ki + nbEff);

    // Set A(kk+1:m, 1:kk) to zero.
    for (let j = 1; j <= kk; j++) {
      for (let i = kk + 1; i <= m; i++) {
        a[aOff + (i - 1) + (j - 1) * lda] = 0.0;
      }
    }
  } else {
    kk = 0;
  }

  // Use unblocked code for the last or only block.
  if (kk < m) {
    dorgl2(
      m - kk,
      n - kk,
      k - kk,
      a,
      aOff + kk + kk * lda,
      lda,
      tau,
      tauOff + kk,
      work,
      workOff
    );
  }

  if (kk > 0) {
    // Use blocked code
    for (let i = ki + 1; i >= 1; i -= nbEff) {
      const ib = Math.min(nbEff, k - i + 1);

      if (i + ib <= m) {
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1)
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

        // Apply H**T to A(i+ib:m, i:n) from the right
        dlarfb(
          RIGHT,
          TRANS,
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

      // Apply H**T to columns i:n of current block
      dorgl2(
        ib,
        n - i + 1,
        ib,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda,
        tau,
        tauOff + (i - 1),
        work,
        workOff
      );

      // Set columns 1:i-1 of current block to zero
      for (let j = 1; j <= i - 1; j++) {
        for (let l = i; l <= i + ib - 1; l++) {
          a[aOff + (l - 1) + (j - 1) * lda] = 0.0;
        }
      }
    }
  }

  work[workOff] = iws;
  return 0;
}
