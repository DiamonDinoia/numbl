// Translated from SRC/dorghr.f
// DORGHR generates a real orthogonal matrix Q which is defined as the
// product of IHI-ILO elementary reflectors of order N, as returned by
// DGEHRD:
//
//    Q = H(ilo) H(ilo+1) . . . H(ihi-1).
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J)   =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)   =>  tau[tauOff + (I-1)]               (I is 1-based)
//   WORK(I)  =>  work[workOff + (I-1)]              (I is 1-based)
//
// Parameters:
//   n       - order of the matrix Q (n >= 0)
//   ilo     - 1 <= ilo <= ihi <= max(1,n); if n=0, ilo=1 and ihi=0
//   ihi     - ILO and IHI must match the previous call to DGEHRD
//   a       - Float64Array; on entry the reflector vectors from DGEHRD;
//             on exit the n-by-n orthogonal matrix Q
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,n))
//   tau     - Float64Array of length n-1; scalar factors from DGEHRD
//   tauOff  - offset into tau for TAU(1)
//   work    - Float64Array workspace of length max(1, lwork)
//   workOff - offset into work
//   lwork   - length of work array (>= max(1, ihi-ilo)); if lwork=-1,
//             workspace query
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dorgqr } from "./dorgqr.js";
import { ilaenv } from "../utils/ilaenv.js";

const ZERO = 0.0;
const ONE = 1.0;

export function dorghr(
  n: number,
  ilo: number,
  ihi: number,
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
  const nh = ihi - ilo;
  const lquery = lwork === -1;

  if (n < 0) {
    info = -1;
  } else if (ilo < 1 || ilo > Math.max(1, n)) {
    info = -2;
  } else if (ihi < Math.min(ilo, n) || ihi > n) {
    info = -3;
  } else if (lda < Math.max(1, n)) {
    info = -5;
  } else if (lwork < Math.max(1, nh) && !lquery) {
    info = -8;
  }

  let lwkopt: number;
  if (info === 0) {
    const nb = ilaenv(1, "DORGQR", " ", nh, nh, nh, -1);
    lwkopt = Math.max(1, nh) * nb;
    work[workOff] = lwkopt;
  } else {
    lwkopt = 1;
  }

  if (info !== 0) {
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  if (n === 0) {
    work[workOff] = 1;
    return 0;
  }

  // Shift the vectors which define the elementary reflectors one
  // column to the right, and set the first ilo and the last n-ihi
  // rows and columns to those of the unit matrix
  //
  // Fortran: DO 40 J = IHI, ILO+1, -1
  for (let j = ihi; j >= ilo + 1; j--) {
    // Fortran: DO 10 I = 1, J-1
    //            A(I,J) = ZERO
    for (let i = 1; i <= j - 1; i++) {
      a[aOff + (i - 1) + (j - 1) * lda] = ZERO;
    }
    // Fortran: DO 20 I = J+1, IHI
    //            A(I,J) = A(I,J-1)
    for (let i = j + 1; i <= ihi; i++) {
      a[aOff + (i - 1) + (j - 1) * lda] = a[aOff + (i - 1) + (j - 2) * lda];
    }
    // Fortran: DO 30 I = IHI+1, N
    //            A(I,J) = ZERO
    for (let i = ihi + 1; i <= n; i++) {
      a[aOff + (i - 1) + (j - 1) * lda] = ZERO;
    }
  }

  // Fortran: DO 60 J = 1, ILO
  for (let j = 1; j <= ilo; j++) {
    // Fortran: DO 50 I = 1, N
    //            A(I,J) = ZERO
    for (let i = 1; i <= n; i++) {
      a[aOff + (i - 1) + (j - 1) * lda] = ZERO;
    }
    // Fortran: A(J,J) = ONE
    a[aOff + (j - 1) + (j - 1) * lda] = ONE;
  }

  // Fortran: DO 80 J = IHI+1, N
  for (let j = ihi + 1; j <= n; j++) {
    // Fortran: DO 70 I = 1, N
    //            A(I,J) = ZERO
    for (let i = 1; i <= n; i++) {
      a[aOff + (i - 1) + (j - 1) * lda] = ZERO;
    }
    // Fortran: A(J,J) = ONE
    a[aOff + (j - 1) + (j - 1) * lda] = ONE;
  }

  if (nh > 0) {
    // Generate Q(ilo+1:ihi,ilo+1:ihi)
    //
    // Fortran: CALL DORGQR(NH, NH, NH, A(ILO+1,ILO+1), LDA, TAU(ILO),
    //                       WORK, LWORK, IINFO)
    dorgqr(
      nh,
      nh,
      nh,
      a,
      aOff + ilo + ilo * lda, // A(ILO+1,ILO+1)
      lda,
      tau,
      tauOff + (ilo - 1) // TAU(ILO)
    );
  }

  work[workOff] = lwkopt;
  return 0;
}
