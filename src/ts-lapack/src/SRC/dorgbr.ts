// Translated from LAPACK/SRC/dorgbr.f
// DORGBR generates one of the real orthogonal matrices Q or P**T
// determined by DGEBRD when reducing a real matrix A to bidiagonal
// form: A = Q * B * P**T. Q and P**T are defined as products of
// elementary reflectors H(i) or G(i) respectively.
//
// If VECT = 'Q' (vect=VECT_Q), A is assumed to have been an M-by-K matrix,
// and Q is of order M:
//   if m >= k, Q = H(1) H(2) . . . H(k) and DORGBR returns the first n
//   columns of Q, where m >= n >= k;
//   if m < k, Q = H(1) H(2) . . . H(m-1) and DORGBR returns Q as an
//   M-by-M matrix.
//
// If VECT = 'P' (vect=VECT_P), A is assumed to have been a K-by-N matrix,
// and P**T is of order N:
//   if k < n, P**T = G(k) . . . G(2) G(1) and DORGBR returns the first m
//   rows of P**T, where n >= m >= k;
//   if k >= n, P**T = G(n-1) . . . G(2) G(1) and DORGBR returns P**T as
//   an N-by-N matrix.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J)   =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)   =>  tau[tauOff + (I-1)]              (I is 1-based)
//   WORK(I)  =>  work[workOff + (I-1)]            (I is 1-based)
//
// Parameters:
//   vect    - VECT_Q (0) to generate Q, VECT_P (1) to generate P**T
//   m       - number of rows of the matrix Q or P**T to be returned (>= 0)
//   n       - number of columns of the matrix Q or P**T to be returned (>= 0)
//   k       - if VECT_Q: columns in original M-by-K matrix;
//             if VECT_P: rows in original K-by-N matrix (>= 0)
//   a       - Float64Array; on entry reflector vectors from DGEBRD;
//             on exit the M-by-N matrix Q or P**T
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,m))
//   tau     - Float64Array; scalar factors of the reflectors
//   tauOff  - offset into tau for TAU(1)
//   work    - Float64Array workspace of length max(1, lwork)
//   workOff - offset into work
//   lwork   - length of work array (>= max(1, min(m,n))); if lwork=-1,
//             workspace query
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { VECT_Q, VECT_P } from "../utils/constants.js";
import { dorgqr } from "./dorgqr.js";
import { dorglq } from "./dorglq.js";
import { xerbla } from "../utils/xerbla.js";
import { ilaenv } from "../utils/ilaenv.js";

const ZERO = 0.0;
const ONE = 1.0;

export function dorgbr(
  vect: number,
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
  // Test the input arguments
  let info = 0;
  const wantq = vect === VECT_Q;
  const mn = Math.min(m, n);
  const lquery = lwork === -1;

  if (!wantq && vect !== VECT_P) {
    info = -1;
  } else if (m < 0) {
    info = -2;
  } else if (
    n < 0 ||
    (wantq && (n > m || n < Math.min(m, k))) ||
    (!wantq && (m > n || m < Math.min(n, k)))
  ) {
    info = -3;
  } else if (k < 0) {
    info = -4;
  } else if (lda < Math.max(1, m)) {
    info = -6;
  } else if (lwork < Math.max(1, mn) && !lquery) {
    info = -9;
  }

  let lwkopt = 1;
  if (info === 0) {
    work[workOff] = 1;
    if (wantq) {
      if (m >= k) {
        const nb = ilaenv(1, "DORGQR", " ", m, n, k, -1);
        lwkopt = Math.max(1, n) * nb;
      } else {
        if (m > 1) {
          const nb = ilaenv(1, "DORGQR", " ", m - 1, m - 1, m - 1, -1);
          lwkopt = Math.max(1, m - 1) * nb;
        }
      }
    } else {
      if (k < n) {
        const nb = ilaenv(1, "DORGLQ", " ", m, n, k, -1);
        lwkopt = Math.max(1, m) * nb;
      } else {
        if (n > 1) {
          const nb = ilaenv(1, "DORGLQ", " ", n - 1, n - 1, n - 1, -1);
          lwkopt = Math.max(1, n - 1) * nb;
        }
      }
    }
    lwkopt = Math.max(lwkopt, mn);
  }

  if (info !== 0) {
    xerbla("DORGBR", -info);
    return info;
  } else if (lquery) {
    work[workOff] = lwkopt;
    return 0;
  }

  // Quick return if possible
  if (m === 0 || n === 0) {
    work[workOff] = 1;
    return 0;
  }

  if (wantq) {
    // Form Q, determined by a call to DGEBRD to reduce an m-by-k matrix

    if (m >= k) {
      // If m >= k, assume m >= n >= k
      dorgqr(m, n, k, a, aOff, lda, tau, tauOff);
    } else {
      // If m < k, assume m = n
      //
      // Shift the vectors which define the elementary reflectors one
      // column to the right, and set the first row and column of Q
      // to those of the unit matrix

      // DO 20 J = M, 2, -1
      for (let j = m; j >= 2; j--) {
        // A(1,J) = ZERO
        a[aOff + 0 + (j - 1) * lda] = ZERO;
        // DO 10 I = J+1, M
        for (let i = j + 1; i <= m; i++) {
          // A(I,J) = A(I,J-1)
          a[aOff + (i - 1) + (j - 1) * lda] = a[aOff + (i - 1) + (j - 2) * lda];
        }
      }
      // A(1,1) = ONE
      a[aOff] = ONE;
      // DO 30 I = 2, M
      for (let i = 2; i <= m; i++) {
        // A(I,1) = ZERO
        a[aOff + (i - 1)] = ZERO;
      }

      if (m > 1) {
        // Form Q(2:m,2:m)
        // CALL DORGQR(M-1, M-1, M-1, A(2,2), LDA, TAU, WORK, LWORK, IINFO)
        dorgqr(
          m - 1,
          m - 1,
          m - 1,
          a,
          aOff + 1 + 1 * lda, // A(2,2)
          lda,
          tau,
          tauOff
        );
      }
    }
  } else {
    // Form P**T, determined by a call to DGEBRD to reduce a k-by-n matrix

    if (k < n) {
      // If k < n, assume k <= m <= n
      dorglq(m, n, k, a, aOff, lda, tau, tauOff, work, workOff, lwork);
    } else {
      // If k >= n, assume m = n
      //
      // Shift the vectors which define the elementary reflectors one
      // row downward, and set the first row and column of P**T to
      // those of the unit matrix

      // A(1,1) = ONE
      a[aOff] = ONE;
      // DO 40 I = 2, N
      for (let i = 2; i <= n; i++) {
        // A(I,1) = ZERO
        a[aOff + (i - 1)] = ZERO;
      }

      // DO 60 J = 2, N
      for (let j = 2; j <= n; j++) {
        // DO 50 I = J-1, 2, -1
        for (let i = j - 1; i >= 2; i--) {
          // A(I,J) = A(I-1,J)
          a[aOff + (i - 1) + (j - 1) * lda] = a[aOff + (i - 2) + (j - 1) * lda];
        }
        // A(1,J) = ZERO
        a[aOff + (j - 1) * lda] = ZERO;
      }

      if (n > 1) {
        // Form P**T(2:n,2:n)
        // CALL DORGLQ(N-1, N-1, N-1, A(2,2), LDA, TAU, WORK, LWORK, IINFO)
        dorglq(
          n - 1,
          n - 1,
          n - 1,
          a,
          aOff + 1 + 1 * lda, // A(2,2)
          lda,
          tau,
          tauOff,
          work,
          workOff,
          lwork
        );
      }
    }
  }

  work[workOff] = lwkopt;
  return 0;
}
