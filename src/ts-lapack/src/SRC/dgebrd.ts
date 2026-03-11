// Translated from LAPACK/SRC/dgebrd.f
// DGEBRD reduces a general real M-by-N matrix A to upper or lower
// bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//
// If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
//
// This is the blocked version using dlabrd for panel factorization
// and dgemm for trailing matrix updates.
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
//   work     - Float64Array workspace of length max(1,lwork)
//   workOff  - offset into work for WORK(1)
//   lwork    - length of workspace; lwork >= max(m,n) when min(m,n) > 0.
//              For optimal performance lwork >= (m+n)*nb.
//              If lwork = -1, workspace query: optimal size returned in work[workOff].
//
// Returns INFO:
//   = 0  successful exit
//   < 0  if -i, the i-th argument had an illegal value

import { dlabrd } from "./dlabrd.js";
import { dgebd2 } from "./dgebd2.js";
import { dgemm } from "../BLAS/dgemm.js";
import { ilaenv } from "../utils/ilaenv.js";
import { xerbla } from "../utils/xerbla.js";
import { NOTRANS, TRANS } from "../utils/constants.js";

const ONE = 1.0;

export function dgebrd(
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
  workOff: number,
  lwork: number
): number {
  let info = 0;
  const minmn = Math.min(m, n);

  let lwkmin: number;
  let lwkopt: number;
  let nb: number;

  if (minmn === 0) {
    lwkmin = 1;
    lwkopt = 1;
    nb = 1;
  } else {
    lwkmin = Math.max(m, n);
    nb = Math.max(1, ilaenv(1, "DGEBRD", " ", m, n, -1, -1));
    lwkopt = (m + n) * nb;
  }
  work[workOff] = lwkopt;

  const lquery = lwork === -1;
  if (m < 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (lda < Math.max(1, m)) {
    info = -4;
  } else if (lwork < lwkmin && !lquery) {
    info = -10;
  }
  if (info < 0) {
    xerbla("DGEBRD", -info);
    return info;
  } else if (lquery) {
    return 0;
  }

  // Quick return if possible
  if (minmn === 0) {
    work[workOff] = 1;
    return 0;
  }

  let ws = Math.max(m, n);
  const ldwrkx = m;
  const ldwrky = n;
  let nx: number;

  if (nb > 1 && nb < minmn) {
    // Set the crossover point NX
    nx = Math.max(nb, ilaenv(3, "DGEBRD", " ", m, n, -1, -1));

    // Determine when to switch from blocked to unblocked code
    if (nx < minmn) {
      ws = lwkopt;
      if (lwork < ws) {
        // Not enough work space for the optimal NB, consider using
        // a smaller block size.
        const nbmin = ilaenv(2, "DGEBRD", " ", m, n, -1, -1);
        if (lwork >= (m + n) * nbmin) {
          nb = Math.floor(lwork / (m + n));
        } else {
          nb = 1;
          nx = minmn;
        }
      }
    }
  } else {
    nx = minmn;
  }

  // Main blocked loop
  // Fortran: DO 30 I = 1, MINMN - NX, NB
  let i = 1;
  for (; i <= minmn - nx; i += nb) {
    // Reduce rows and columns i:i+nb-1 to bidiagonal form and return
    // the matrices X and Y which are needed to update the unreduced
    // part of the matrix.
    //
    // dlabrd(m-i+1, n-i+1, nb, A(i,i), lda, D(i), E(i),
    //        TAUQ(i), TAUP(i), WORK, ldwrkx, WORK(ldwrkx*nb+1), ldwrky)
    dlabrd(
      m - i + 1,
      n - i + 1,
      nb,
      a,
      aOff + (i - 1) + (i - 1) * lda,
      lda,
      d,
      dOff + (i - 1),
      e,
      eOff + (i - 1),
      tauq,
      tauqOff + (i - 1),
      taup,
      taupOff + (i - 1),
      work,
      workOff,
      ldwrkx,
      work,
      workOff + ldwrkx * nb,
      ldwrky
    );

    // Update the trailing submatrix A(i+nb:m, i+nb:n), using
    //   A := A - V*Y**T - X*U**T
    //
    // First: A(i+nb:m, i+nb:n) -= A(i+nb:m, i:i+nb-1) * Y(nb+1:n-i+1, 1:nb)**T
    // dgemm('N','T', m-i-nb+1, n-i-nb+1, nb, -ONE,
    //        A(i+nb,i), lda, WORK(ldwrkx*nb+nb+1), ldwrky, ONE, A(i+nb,i+nb), lda)
    dgemm(
      NOTRANS,
      TRANS,
      m - i - nb + 1,
      n - i - nb + 1,
      nb,
      -ONE,
      a,
      aOff + (i + nb - 1) + (i - 1) * lda,
      lda,
      work,
      workOff + ldwrkx * nb + nb,
      ldwrky,
      ONE,
      a,
      aOff + (i + nb - 1) + (i + nb - 1) * lda,
      lda
    );

    // Second: A(i+nb:m, i+nb:n) -= X(nb+1:m-i+1, 1:nb) * A(i:i+nb-1, i+nb:n)
    // dgemm('N','N', m-i-nb+1, n-i-nb+1, nb, -ONE,
    //        WORK(nb+1), ldwrkx, A(i,i+nb), lda, ONE, A(i+nb,i+nb), lda)
    dgemm(
      NOTRANS,
      NOTRANS,
      m - i - nb + 1,
      n - i - nb + 1,
      nb,
      -ONE,
      work,
      workOff + nb,
      ldwrkx,
      a,
      aOff + (i - 1) + (i + nb - 1) * lda,
      lda,
      ONE,
      a,
      aOff + (i + nb - 1) + (i + nb - 1) * lda,
      lda
    );

    // Copy diagonal and off-diagonal elements of B back into A
    if (m >= n) {
      for (let j = i; j <= i + nb - 1; j++) {
        a[aOff + (j - 1) + (j - 1) * lda] = d[dOff + (j - 1)];
        a[aOff + (j - 1) + j * lda] = e[eOff + (j - 1)];
      }
    } else {
      for (let j = i; j <= i + nb - 1; j++) {
        a[aOff + (j - 1) + (j - 1) * lda] = d[dOff + (j - 1)];
        a[aOff + j + (j - 1) * lda] = e[eOff + (j - 1)];
      }
    }
  }

  // Use unblocked code to reduce the remainder of the matrix
  // dgebd2(m-i+1, n-i+1, A(i,i), lda, D(i), E(i), TAUQ(i), TAUP(i), WORK, iinfo)
  dgebd2(
    m - i + 1,
    n - i + 1,
    a,
    aOff + (i - 1) + (i - 1) * lda,
    lda,
    d,
    dOff + (i - 1),
    e,
    eOff + (i - 1),
    tauq,
    tauqOff + (i - 1),
    taup,
    taupOff + (i - 1),
    work,
    workOff
  );

  work[workOff] = ws;
  return 0;
}
