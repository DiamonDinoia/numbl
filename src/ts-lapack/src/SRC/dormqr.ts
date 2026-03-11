// Translated from SRC/dormqr.f
// DORMQR overwrites the general real M-by-N matrix C with
//
//                 SIDE = LEFT     SIDE = RIGHT
// TRANS = NOTRANS:  Q * C          C * Q
// TRANS = TRANS:    Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix defined as the product of k
// elementary reflectors
//
//       Q = H(1) H(2) . . . H(k)
//
// as returned by DGEQRF. Q is of order M if SIDE = LEFT and of order N
// if SIDE = RIGHT.
//
// This is the blocked (Level 3 BLAS) version that calls dlarft/dlarfb
// for blocks, falling back to the unblocked dorm2r when the block size
// is too small.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]
//   C(I,J) => c[cOff + (I-1) + (J-1)*ldc]
//   TAU(I) => tau[tauOff + (I-1)]
//
// Parameters:
//   side    - LEFT (0) or RIGHT (1)
//   trans   - NOTRANS (0) or TRANS (1)
//   m       - number of rows of C (m >= 0)
//   n       - number of columns of C (n >= 0)
//   k       - number of elementary reflectors (0 <= k <= nq)
//   a       - contains the reflector vectors as returned by dgeqrf
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

import { LEFT, RIGHT, NOTRANS, TRANS } from "../utils/constants.js";
import { dorm2r } from "./dorm2r.js";
import { dlarft } from "./dlarft.js";
import { dlarfb } from "./dlarfb.js";

// Constants matching Fortran PARAMETER block
const NBMAX = 64;
const LDT = NBMAX + 1; // 65
const TSIZE = LDT * NBMAX; // 4160

// Constants for dlarft/dlarfb
const FORWARD = 0;
const COLUMNWISE = 0;

export function dormqr(
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
  const left = side === LEFT;
  const notran = trans === NOTRANS;
  const lquery = lwork === -1;

  // NQ is the order of Q and NW is the minimum dimension of WORK
  let nq: number, nw: number;
  if (left) {
    nq = m;
    nw = Math.max(1, n);
  } else {
    nq = n;
    nw = Math.max(1, m);
  }

  if (side !== LEFT && side !== RIGHT) {
    info = -1;
  } else if (trans !== NOTRANS && trans !== TRANS) {
    info = -2;
  } else if (m < 0) {
    info = -3;
  } else if (n < 0) {
    info = -4;
  } else if (k < 0 || k > nq) {
    info = -5;
  } else if (lda < Math.max(1, nq)) {
    info = -7;
  } else if (ldc < Math.max(1, m)) {
    info = -10;
  } else if (lwork < nw && !lquery) {
    info = -12;
  }

  let nb: number;
  let lwkopt: number;

  if (info === 0) {
    // Compute the workspace requirements
    // ilaenv(1, 'DORMQR', ...) returns the optimal block size; we use 64
    nb = Math.min(NBMAX, 64);
    lwkopt = nw * nb + TSIZE;
    work[workOff] = lwkopt;
  } else {
    nb = 0;
    lwkopt = 0;
  }

  if (info !== 0) {
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  if (m === 0 || n === 0 || k === 0) {
    work[workOff] = 1;
    return 0;
  }

  let nbmin = 2;
  const ldwork = nw;
  if (nb > 1 && nb < k) {
    if (lwork < lwkopt) {
      nb = Math.floor((lwork - TSIZE) / ldwork);
      // ilaenv(2, 'DORMQR', ...) returns minimum block size; use 2
      nbmin = Math.max(2, 2);
    }
  }

  if (nb < nbmin || nb >= k) {
    // Use unblocked code
    dorm2r(
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
      workOff
    );
  } else {
    // Use blocked code
    const iwt = 1 + nw * nb; // 1-based offset into work for T storage

    // Determine loop direction
    let i1: number, i2: number, i3: number;
    if ((left && !notran) || (!left && notran)) {
      i1 = 1;
      i2 = k;
      i3 = nb;
    } else {
      i1 = Math.floor((k - 1) / nb) * nb + 1;
      i2 = 1;
      i3 = -nb;
    }

    let ni: number, mi: number, ic: number, jc: number;
    if (left) {
      ni = n;
      jc = 1;
      mi = 0; // will be set in loop
      ic = 0; // will be set in loop
    } else {
      mi = m;
      ic = 1;
      ni = 0; // will be set in loop
      jc = 0; // will be set in loop
    }

    for (let i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3) {
      const ib = Math.min(nb, k - i + 1);

      // Form the triangular factor of the block reflector
      // H = H(i) H(i+1) ... H(i+ib-1)
      // A(I,I) => a[aOff + (i-1) + (i-1)*lda]
      // TAU(I) => tau[tauOff + (i-1)]
      // WORK(IWT) => work[workOff + (iwt-1)]
      dlarft(
        FORWARD,
        COLUMNWISE,
        nq - i + 1,
        ib,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda,
        tau,
        tauOff + (i - 1),
        work,
        workOff + (iwt - 1),
        LDT
      );

      if (left) {
        // H or H**T is applied to C(i:m,1:n)
        mi = m - i + 1;
        ic = i;
      } else {
        // H or H**T is applied to C(1:m,i:n)
        ni = n - i + 1;
        jc = i;
      }

      // Apply H or H**T
      // C(IC,JC) => c[cOff + (ic-1) + (jc-1)*ldc]
      dlarfb(
        side,
        trans,
        FORWARD,
        COLUMNWISE,
        mi,
        ni,
        ib,
        a,
        aOff + (i - 1) + (i - 1) * lda,
        lda,
        work,
        workOff + (iwt - 1),
        LDT,
        c,
        cOff + (ic - 1) + (jc - 1) * ldc,
        ldc,
        work,
        workOff,
        ldwork
      );
    }
  }

  work[workOff] = lwkopt;
  return 0;
}
