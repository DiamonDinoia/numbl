// Translated from LAPACK/SRC/dormbr.f
// DORMBR overwrites the general real M-by-N matrix C with
//
// If VECT = 'Q':
//                 SIDE = LEFT     SIDE = RIGHT
// TRANS = NOTRANS:  Q * C          C * Q
// TRANS = TRANS:    Q**T * C       C * Q**T
//
// If VECT = 'P':
//                 SIDE = LEFT     SIDE = RIGHT
// TRANS = NOTRANS:  P * C          C * P
// TRANS = TRANS:    P**T * C       C * P**T
//
// Here Q and P**T are the orthogonal matrices determined by DGEBRD when
// reducing a real matrix A to bidiagonal form: A = Q * B * P**T. Q and
// P**T are defined as products of elementary reflectors H(i) and G(i)
// respectively.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]
//   C(I,J) => c[cOff + (I-1) + (J-1)*ldc]
//   TAU(I) => tau[tauOff + (I-1)]
//
// Parameters:
//   vect    - VECT_Q (0) or VECT_P (1)
//   side    - LEFT (0) or RIGHT (1)
//   trans   - NOTRANS (0) or TRANS (1)
//   m       - number of rows of C (m >= 0)
//   n       - number of columns of C (n >= 0)
//   k       - number of columns (VECT='Q') or rows (VECT='P') in the
//             original matrix reduced by DGEBRD. k >= 0.
//   a       - contains the reflector vectors as returned by dgebrd
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of A
//   tau     - scalar factors of the reflectors
//   tauOff  - offset into tau for TAU(1)
//   c       - the m-by-n matrix C; overwritten on exit
//   cOff    - offset into c for C(1,1)
//   ldc     - leading dimension of C
//   work    - workspace array of dimension max(1, lwork)
//   workOff - offset into work
//   lwork   - dimension of work; if lwork=-1, workspace query
//
// Returns INFO:
//   = 0: successful exit
//   < 0: if INFO = -i, the i-th argument had an illegal value

import {
  VECT_Q,
  VECT_P,
  LEFT,
  RIGHT,
  NOTRANS,
  TRANS,
} from "../utils/constants.js";
import { dormqr } from "./dormqr.js";
import { dormlq } from "./dormlq.js";
import { xerbla } from "../utils/xerbla.js";
import { ilaenv } from "../utils/ilaenv.js";

export function dormbr(
  vect: number,
  side: number,
  trans: number,
  m: number,
  n: number,
  k: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number,
  c: Float64Array,
  cOff: number,
  ldc: number,
  work: Float64Array,
  workOff: number,
  lwork: number
): number {
  // Test the input arguments
  let info = 0;
  const applyq = vect === VECT_Q;
  const left = side === LEFT;
  const notran = trans === NOTRANS;
  const lquery = lwork === -1;

  // NQ is the order of Q or P and NW is the minimum dimension of WORK
  let nq: number;
  let nw: number;
  if (left) {
    nq = m;
    nw = Math.max(1, n);
  } else {
    nq = n;
    nw = Math.max(1, m);
  }

  if (!applyq && vect !== VECT_P) {
    info = -1;
  } else if (!left && side !== RIGHT) {
    info = -2;
  } else if (!notran && trans !== TRANS) {
    info = -3;
  } else if (m < 0) {
    info = -4;
  } else if (n < 0) {
    info = -5;
  } else if (k < 0) {
    info = -6;
  } else if (
    (applyq && lda < Math.max(1, nq)) ||
    (!applyq && lda < Math.max(1, Math.min(nq, k)))
  ) {
    info = -8;
  } else if (ldc < Math.max(1, m)) {
    info = -11;
  } else if (lwork < nw && !lquery) {
    info = -13;
  }

  let lwkopt = 0;
  if (info === 0) {
    let nb: number;
    if (applyq) {
      if (left) {
        nb = ilaenv(1, "DORMQR", "LN", m - 1, n, m - 1, -1);
      } else {
        nb = ilaenv(1, "DORMQR", "LN", m, n - 1, n - 1, -1);
      }
    } else {
      if (left) {
        nb = ilaenv(1, "DORMLQ", "LN", m - 1, n, m - 1, -1);
      } else {
        nb = ilaenv(1, "DORMLQ", "LN", m, n - 1, n - 1, -1);
      }
    }
    lwkopt = nw * nb;
    work[workOff] = lwkopt;
  }

  if (info !== 0) {
    xerbla("DORMBR", -info);
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  work[workOff] = 1;
  if (m === 0 || n === 0) {
    return 0;
  }

  if (applyq) {
    // Apply Q
    if (nq >= k) {
      // Q was determined by a call to DGEBRD with nq >= k
      dormqr(
        side,
        trans,
        m,
        n,
        k,
        a,
        aOff,
        lda,
        tau,
        tauOff,
        c,
        cOff,
        ldc,
        work,
        workOff,
        lwork
      );
    } else if (nq > 1) {
      // Q was determined by a call to DGEBRD with nq < k
      let mi: number;
      let ni: number;
      let i1: number;
      let i2: number;
      if (left) {
        mi = m - 1;
        ni = n;
        i1 = 2;
        i2 = 1;
      } else {
        mi = m;
        ni = n - 1;
        i1 = 1;
        i2 = 2;
      }
      // A(2,1) => aOff + 1, C(I1,I2) => cOff + (I1-1) + (I2-1)*ldc
      dormqr(
        side,
        trans,
        mi,
        ni,
        nq - 1,
        a,
        aOff + 1,
        lda,
        tau,
        tauOff,
        c,
        cOff + (i1 - 1) + (i2 - 1) * ldc,
        ldc,
        work,
        workOff,
        lwork
      );
    }
  } else {
    // Apply P
    // When applying P, we transpose the trans argument for dormlq
    const transt = notran ? TRANS : NOTRANS;

    if (nq > k) {
      // P was determined by a call to DGEBRD with nq > k
      dormlq(
        side,
        transt,
        m,
        n,
        k,
        a,
        aOff,
        lda,
        tau,
        tauOff,
        c,
        cOff,
        ldc,
        work,
        workOff,
        lwork
      );
    } else if (nq > 1) {
      // P was determined by a call to DGEBRD with nq <= k
      let mi: number;
      let ni: number;
      let i1: number;
      let i2: number;
      if (left) {
        mi = m - 1;
        ni = n;
        i1 = 2;
        i2 = 1;
      } else {
        mi = m;
        ni = n - 1;
        i1 = 1;
        i2 = 2;
      }
      // A(1,2) => aOff + lda, C(I1,I2) => cOff + (I1-1) + (I2-1)*ldc
      dormlq(
        side,
        transt,
        mi,
        ni,
        nq - 1,
        a,
        aOff + lda,
        lda,
        tau,
        tauOff,
        c,
        cOff + (i1 - 1) + (i2 - 1) * ldc,
        ldc,
        work,
        workOff,
        lwork
      );
    }
  }

  work[workOff] = lwkopt;
  return 0;
}
