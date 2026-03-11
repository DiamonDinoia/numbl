// Translated from SRC/dormhr.f
// DORMHR overwrites the general real M-by-N matrix C with
//
//                 SIDE = LEFT     SIDE = RIGHT
// TRANS = NOTRANS:  Q * C          C * Q
// TRANS = TRANS:    Q**T * C       C * Q**T
//
// where Q is a real orthogonal matrix of order nq, with nq = m if
// SIDE = LEFT and nq = n if SIDE = RIGHT. Q is defined as the product of
// IHI-ILO elementary reflectors, as returned by DGEHRD:
//
//       Q = H(ilo) H(ilo+1) . . . H(ihi-1).
//
// This routine adjusts indices and delegates to DORMQR.
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
//   ilo     - 1-based index (as returned by dgehrd)
//   ihi     - 1-based index (as returned by dgehrd)
//   a       - contains the reflector vectors as returned by dgehrd
//   aOff    - offset into a for A(1,1)
//   lda     - leading dimension of A
//   tau     - scalar factors of the reflectors (dimension nq-1)
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
import { dormqr } from "./dormqr.js";

export function dormhr(
  side: number,
  trans: number,
  m: number,
  n: number,
  ilo: number,
  ihi: number,
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
  const nh = ihi - ilo;
  const left = side === LEFT;
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
  } else if (ilo < 1 || ilo > Math.max(1, nq)) {
    info = -5;
  } else if (ihi < Math.min(ilo, nq) || ihi > nq) {
    info = -6;
  } else if (lda < Math.max(1, nq)) {
    info = -8;
  } else if (ldc < Math.max(1, m)) {
    info = -11;
  } else if (lwork < nw && !lquery) {
    info = -13;
  }

  let lwkopt: number;

  if (info === 0) {
    // Compute optimal workspace
    // ilaenv(1, 'DORMQR', ...) for block size; we use 64
    const nb = 64;
    lwkopt = nw * nb;
    work[workOff] = lwkopt;
  } else {
    lwkopt = 0;
  }

  if (info !== 0) {
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  if (m === 0 || n === 0 || nh === 0) {
    work[workOff] = 1;
    return 0;
  }

  let mi: number, ni: number, i1: number, i2: number;
  if (left) {
    mi = nh;
    ni = n;
    i1 = ilo + 1;
    i2 = 1;
  } else {
    mi = m;
    ni = nh;
    i1 = 1;
    i2 = ilo + 1;
  }

  // Call DORMQR with adjusted indices
  // A(ILO+1, ILO) => a[aOff + (ilo+1-1) + (ilo-1)*lda] = a[aOff + ilo + (ilo-1)*lda]
  // TAU(ILO) => tau[tauOff + (ilo-1)]
  // C(I1, I2) => c[cOff + (i1-1) + (i2-1)*ldc]
  dormqr(
    side,
    trans,
    mi,
    ni,
    nh,
    a,
    aOff + ilo + (ilo - 1) * lda,
    lda,
    tau,
    tauOff + (ilo - 1),
    c,
    cOff + (i1 - 1) + (i2 - 1) * ldc,
    ldc,
    work,
    workOff,
    lwork
  );

  work[workOff] = lwkopt;
  return 0;
}
