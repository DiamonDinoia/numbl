// Translated from SRC/dgehrd.f
// DGEHRD reduces a real general matrix A to upper Hessenberg form H by
// an orthogonal similarity transformation:  Q**T * A * Q = H .
//
// This is the blocked algorithm.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J)   =>  a[aOff + (I-1) + (J-1)*lda]    (I,J are 1-based)
//   TAU(I)   =>  tau[tauOff + (I-1)]               (I is 1-based)
//   WORK(I)  =>  work[workOff + (I-1)]              (I is 1-based)
//
// Parameters:
//   n       - order of the matrix A (n >= 0)
//   ilo     - 1 <= ilo <= ihi <= max(1,n); if n=0, ilo=1 and ihi=0
//   ihi     - assumed that A is already upper triangular in rows/columns
//             1:ilo-1 and ihi+1:n
//   a       - Float64Array; on entry the n-by-n general matrix; on exit upper
//             Hessenberg form with reflector vectors below the first subdiagonal
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of a (>= max(1,n))
//   tau     - Float64Array of length n-1; scalar factors of reflectors
//   tauOff  - offset into tau for TAU(1)
//   work    - Float64Array workspace of length max(1, lwork)
//   workOff - offset into work
//   lwork   - length of work array (>= max(1,n)); if lwork=-1, workspace query
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dgehd2 } from "./dgehd2.js";
import { dlahr2 } from "./dlahr2.js";
import { dlarfb } from "./dlarfb.js";
import { dgemm } from "../BLAS/dgemm.js";
import { dtrmm } from "../BLAS/dtrmm.js";
import { daxpy } from "../BLAS/daxpy.js";
import { ilaenv } from "../utils/ilaenv.js";
import {
  NOTRANS,
  TRANS,
  LOWER,
  UNIT,
  LEFT,
  RIGHT,
} from "../utils/constants.js";

const ZERO = 0.0;
const ONE = 1.0;

// Constants matching the Fortran PARAMETER block
const NBMAX = 64;
const LDT = NBMAX + 1;
const TSIZE = LDT * NBMAX;

// Forward/Columnwise constants for dlarfb
const FORWARD = 0;
const COLUMNWISE = 0;

export function dgehrd(
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
  // Test the input parameters
  let info = 0;
  const lquery = lwork === -1;
  if (n < 0) {
    info = -1;
  } else if (ilo < 1 || ilo > Math.max(1, n)) {
    info = -2;
  } else if (ihi < Math.min(ilo, n) || ihi > n) {
    info = -3;
  } else if (lda < Math.max(1, n)) {
    info = -5;
  } else if (lwork < Math.max(1, n) && !lquery) {
    info = -8;
  }

  const nh = ihi - ilo + 1;
  let lwkopt: number;
  if (info === 0) {
    // Compute the workspace requirements
    if (nh <= 1) {
      lwkopt = 1;
    } else {
      const nb0 = Math.min(NBMAX, ilaenv(1, "DGEHRD", " ", n, ilo, ihi, -1));
      lwkopt = n * nb0 + TSIZE;
    }
    work[workOff] = lwkopt;
  } else {
    lwkopt = 1;
  }

  if (info !== 0) {
    return info;
  } else if (lquery) {
    return 0;
  }

  // Set elements 1:ILO-1 and IHI:N-1 of TAU to zero
  for (let i = 1; i <= ilo - 1; i++) {
    tau[tauOff + (i - 1)] = ZERO;
  }
  for (let i = Math.max(1, ihi); i <= n - 1; i++) {
    tau[tauOff + (i - 1)] = ZERO;
  }

  // Quick return if possible
  if (nh <= 1) {
    work[workOff] = 1;
    return 0;
  }

  // Determine the block size
  let nb = Math.min(NBMAX, ilaenv(1, "DGEHRD", " ", n, ilo, ihi, -1));
  let nbmin = 2;
  let nx = 0;
  let i = ilo; // will be used after the blocked section

  if (nb > 1 && nb < nh) {
    // Determine when to cross over from blocked to unblocked code
    // (last block is always handled by unblocked code)
    nx = Math.max(nb, ilaenv(3, "DGEHRD", " ", n, ilo, ihi, -1));
    if (nx < nh) {
      // Determine if workspace is large enough for blocked code
      if (lwork < lwkopt) {
        // Not enough workspace to use optimal NB: determine the
        // minimum value of NB, and reduce NB or force use of
        // unblocked code
        nbmin = Math.max(2, ilaenv(2, "DGEHRD", " ", n, ilo, ihi, -1));
        if (lwork >= n * nbmin + TSIZE) {
          nb = Math.floor((lwork - TSIZE) / n);
        } else {
          nb = 1;
        }
      }
    }
  }
  const ldwork = n;

  if (nb < nbmin || nb >= nh) {
    // Use unblocked code below
    i = ilo;
  } else {
    // Use blocked code
    const iwt = 1 + n * nb; // 1-based index into work for T storage

    for (i = ilo; i <= ihi - 1 - nx; i += nb) {
      const ib = Math.min(nb, ihi - i);

      // Reduce columns i:i+ib-1 to Hessenberg form, returning the
      // matrices V and T of the block reflector H = I - V*T*V**T
      // which performs the reduction, and also the matrix Y = A*V*T
      //
      // Fortran: CALL DLAHR2(IHI, I, IB, A(1,I), LDA, TAU(I),
      //                       WORK(IWT), LDT, WORK, LDWORK)
      dlahr2(
        ihi,
        i,
        ib,
        a,
        aOff + (i - 1) * lda, // A(1,I)
        lda,
        tau,
        tauOff + (i - 1), // TAU(I)
        work,
        workOff + (iwt - 1), // WORK(IWT)
        LDT,
        work,
        workOff, // WORK(1)
        ldwork
      );

      // Apply the block reflector H to A(1:ihi,i+ib:ihi) from the
      // right, computing  A := A - Y * V**T. V(i+ib,ib-1) must be set to 1
      //
      // Fortran: EI = A(I+IB, I+IB-1)
      const eiIdx = aOff + (i + ib - 1) + (i + ib - 2) * lda;
      const ei = a[eiIdx];
      // Fortran: A(I+IB, I+IB-1) = ONE
      a[eiIdx] = ONE;

      // Fortran: DGEMM('No transpose', 'Transpose', IHI, IHI-I-IB+1, IB,
      //                 -ONE, WORK, LDWORK, A(I+IB,I), LDA, ONE, A(1,I+IB), LDA)
      dgemm(
        NOTRANS,
        TRANS,
        ihi,
        ihi - i - ib + 1,
        ib,
        -ONE,
        work,
        workOff, // WORK(1)
        ldwork,
        a,
        aOff + (i + ib - 1) + (i - 1) * lda, // A(I+IB,I)
        lda,
        ONE,
        a,
        aOff + (i + ib - 1) * lda, // A(1,I+IB)
        lda
      );

      // Fortran: A(I+IB, I+IB-1) = EI
      a[eiIdx] = ei;

      // Apply the block reflector H to A(1:i,i+1:i+ib-1) from the right
      //
      // Fortran: DTRMM('Right', 'Lower', 'Transpose', 'Unit', I, IB-1,
      //                 ONE, A(I+1,I), LDA, WORK, LDWORK)
      dtrmm(
        RIGHT,
        LOWER,
        TRANS,
        UNIT,
        i,
        ib - 1,
        ONE,
        a,
        aOff + i + (i - 1) * lda, // A(I+1,I)
        lda,
        work,
        workOff, // WORK(1)
        ldwork
      );

      // Fortran: DO 30 J = 0, IB-2
      //            CALL DAXPY(I, -ONE, WORK(LDWORK*J+1), 1, A(1,I+J+1), 1)
      // 30       CONTINUE
      for (let j = 0; j <= ib - 2; j++) {
        daxpy(
          i,
          -ONE,
          work,
          workOff + ldwork * j, // WORK(LDWORK*J+1)
          1,
          a,
          aOff + (i + j) * lda, // A(1,I+J+1)
          1
        );
      }

      // Apply the block reflector H to A(i+1:ihi,i+ib:n) from the left
      //
      // Fortran: CALL DLARFB('Left', 'Transpose', 'Forward', 'Columnwise',
      //                       IHI-I, N-I-IB+1, IB, A(I+1,I), LDA,
      //                       WORK(IWT), LDT, A(I+1,I+IB), LDA,
      //                       WORK, LDWORK)
      dlarfb(
        LEFT,
        TRANS,
        FORWARD,
        COLUMNWISE,
        ihi - i,
        n - i - ib + 1,
        ib,
        a,
        aOff + i + (i - 1) * lda, // A(I+1,I)
        lda,
        work,
        workOff + (iwt - 1), // WORK(IWT)
        LDT,
        a,
        aOff + i + (i + ib - 1) * lda, // A(I+1,I+IB)
        lda,
        work,
        workOff, // WORK(1)
        ldwork
      );
    }
    // After the loop, i holds the next start column for the unblocked tail
  }

  // Use unblocked code to reduce the rest of the matrix
  //
  // Fortran: CALL DGEHD2(N, I, IHI, A, LDA, TAU, WORK, IINFO)
  dgehd2(n, i, ihi, a, aOff, lda, tau, tauOff, work, workOff);

  work[workOff] = lwkopt;

  return 0;
}
