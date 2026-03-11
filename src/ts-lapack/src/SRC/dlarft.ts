// Translated from SRC/dlarft.f and SRC/dlarft_lvl2.f
// DLARFT forms the triangular factor T of a real block reflector H
// of order n, which is defined as a product of k elementary reflectors.
//
// If DIRECT = FORWARD (0), H = H(1) H(2) ... H(k) and T is upper triangular;
// If DIRECT = BACKWARD (1), H = H(k) ... H(2) H(1) and T is lower triangular.
//
// If STOREV = COLUMNWISE (0), the vector which defines H(i) is stored in
// the i-th column of V, and  H = I - V * T * V**T
//
// If STOREV = ROWWISE (1), the vector which defines H(i) is stored in
// the i-th row of V, and  H = I - V**T * T * V
//
// DIRECT parameter: FORWARD=0, BACKWARD=1
// STOREV parameter: COLUMNWISE=0, ROWWISE=1
//
// Array indexing convention (column-major, matching Fortran):
//   V(I,J) => v[vOff + (I-1) + (J-1)*ldv]
//   T(I,J) => t[tOff + (I-1) + (J-1)*ldt]
//   TAU(I) => tau[tauOff + (I-1)]

import {
  TRANS,
  NOTRANS,
  UPPER,
  LOWER,
  UNIT,
  NONUNIT,
  LEFT,
  RIGHT,
} from "../utils/constants.js";
import { dgemv } from "../BLAS/dgemv.js";
import { dtrmv } from "../BLAS/dtrmv.js";
import { dgemm } from "../BLAS/dgemm.js";
import { dtrmm } from "../BLAS/dtrmm.js";
import { dlacpy } from "./dlacpy.js";

// Constants for DIRECT parameter
const FORWARD = 0;
// const BACKWARD = 1;

// Constants for STOREV parameter
const COLUMNWISE = 0;
// const ROWWISE = 1;

// Crossover point: for k < NX_CROSSOVER, use the level-2 implementation directly
const NX_CROSSOVER = 64;

/**
 * DLARFT forms the triangular factor T of a real block reflector H
 * of order n, which is defined as a product of k elementary reflectors.
 *
 * @param direct  - 0=FORWARD ('F'): T is upper triangular;
 *                  1=BACKWARD ('B'): T is lower triangular
 * @param storev  - 0=COLUMNWISE ('C'): reflectors stored in columns of V;
 *                  1=ROWWISE ('R'): reflectors stored in rows of V
 * @param n       - Order of the block reflector H (n >= 0)
 * @param k       - Number of elementary reflectors (k >= 1)
 * @param v       - The matrix V
 * @param vOff    - Offset into v for V(1,1)
 * @param ldv     - Leading dimension of V
 * @param tau     - TAU(i) = scalar factor of H(i)
 * @param tauOff  - Offset into tau for TAU(1)
 * @param t       - Output k-by-k triangular factor T
 * @param tOff    - Offset into t for T(1,1)
 * @param ldt     - Leading dimension of T (ldt >= k)
 */
export function dlarft(
  direct: number,
  storev: number,
  n: number,
  k: number,
  v: Float64Array,
  vOff: number,
  ldv: number,
  tau: Float64Array,
  tauOff: number,
  t: Float64Array,
  tOff: number,
  ldt: number
): void {
  // Quick return if possible
  if (n === 0 || k === 0) {
    return;
  }

  // Base case
  if (n === 1 || k === 1) {
    t[tOff] = tau[tauOff];
    return;
  }

  // Determine when to cross over to the level-2 implementation
  if (k < NX_CROSSOVER) {
    dlarft_lvl2(direct, storev, n, k, v, vOff, ldv, tau, tauOff, t, tOff, ldt);
    return;
  }

  const dirf = direct === FORWARD;
  const colv = storev === COLUMNWISE;

  // QR: forward + columnwise
  // LQ: forward + rowwise
  // QL: backward + columnwise
  // RQ: backward + rowwise
  const qr = dirf && colv;
  const lq = dirf && !colv;
  const ql = !dirf && colv;
  // rq = !dirf && !colv (the else case)

  const l = Math.floor(k / 2);

  if (qr) {
    // Compute T_{1,1} recursively
    dlarft(direct, storev, n, l, v, vOff, ldv, tau, tauOff, t, tOff, ldt);

    // Compute T_{2,2} recursively
    // V(L+1, L+1) => vOff + L + L*ldv
    // TAU(L+1) => tauOff + L
    // T(L+1, L+1) => tOff + L + L*ldt
    dlarft(
      direct,
      storev,
      n - l,
      k - l,
      v,
      vOff + l + l * ldv,
      ldv,
      tau,
      tauOff + l,
      t,
      tOff + l + l * ldt,
      ldt
    );

    // Compute T_{1,2}
    // T_{1,2} = V_{2,1}' (transpose of V(L+1:K, 1:L))
    for (let j = 1; j <= l; j++) {
      for (let i = 1; i <= k - l; i++) {
        // T(J, L+I) = V(L+I, J)
        t[tOff + (j - 1) + (l + i - 1) * ldt] =
          v[vOff + (l + i - 1) + (j - 1) * ldv];
      }
    }

    // T_{1,2} = T_{1,2} * V_{2,2}   (V_{2,2} is unit lower triangular at V(L+1,L+1))
    dtrmm(
      RIGHT,
      LOWER,
      NOTRANS,
      UNIT,
      l,
      k - l,
      1.0,
      v,
      vOff + l + l * ldv,
      ldv,
      t,
      tOff + l * ldt,
      ldt
    );

    // T_{1,2} = V_{3,1}' * V_{3,2} + T_{1,2}
    if (n > k) {
      dgemm(
        TRANS,
        NOTRANS,
        l,
        k - l,
        n - k,
        1.0,
        v,
        vOff + k + 0 * ldv,
        ldv,
        v,
        vOff + k + l * ldv,
        ldv,
        1.0,
        t,
        tOff + l * ldt,
        ldt
      );
    }

    // T_{1,2} = -T_{1,1} * T_{1,2}
    dtrmm(
      LEFT,
      UPPER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      -1.0,
      t,
      tOff,
      ldt,
      t,
      tOff + l * ldt,
      ldt
    );

    // T_{1,2} = T_{1,2} * T_{2,2}
    dtrmm(
      RIGHT,
      UPPER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      1.0,
      t,
      tOff + l + l * ldt,
      ldt,
      t,
      tOff + l * ldt,
      ldt
    );
  } else if (lq) {
    // Compute T_{1,1} recursively
    dlarft(direct, storev, n, l, v, vOff, ldv, tau, tauOff, t, tOff, ldt);

    // Compute T_{2,2} recursively
    dlarft(
      direct,
      storev,
      n - l,
      k - l,
      v,
      vOff + l + l * ldv,
      ldv,
      tau,
      tauOff + l,
      t,
      tOff + l + l * ldt,
      ldt
    );

    // T_{1,2} = V_{1,2}  (copy L rows, K-L cols from V(1,L+1))
    // uplo=-1 means copy all
    dlacpy(-1, l, k - l, v, vOff + l * ldv, ldv, t, tOff + l * ldt, ldt);

    // T_{1,2} = T_{1,2} * V_{2,2}'  (V_{2,2} is unit upper triangular at V(L+1,L+1))
    dtrmm(
      RIGHT,
      UPPER,
      TRANS,
      UNIT,
      l,
      k - l,
      1.0,
      v,
      vOff + l + l * ldv,
      ldv,
      t,
      tOff + l * ldt,
      ldt
    );

    // T_{1,2} = V_{1,3} * V_{2,3}' + T_{1,2}
    if (n > k) {
      dgemm(
        NOTRANS,
        TRANS,
        l,
        k - l,
        n - k,
        1.0,
        v,
        vOff + k * ldv,
        ldv,
        v,
        vOff + l + k * ldv,
        ldv,
        1.0,
        t,
        tOff + l * ldt,
        ldt
      );
    }

    // T_{1,2} = -T_{1,1} * T_{1,2}
    dtrmm(
      LEFT,
      UPPER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      -1.0,
      t,
      tOff,
      ldt,
      t,
      tOff + l * ldt,
      ldt
    );

    // T_{1,2} = T_{1,2} * T_{2,2}
    dtrmm(
      RIGHT,
      UPPER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      1.0,
      t,
      tOff + l + l * ldt,
      ldt,
      t,
      tOff + l * ldt,
      ldt
    );
  } else if (ql) {
    // Compute T_{1,1} recursively
    dlarft(
      direct,
      storev,
      n - l,
      k - l,
      v,
      vOff,
      ldv,
      tau,
      tauOff,
      t,
      tOff,
      ldt
    );

    // Compute T_{2,2} recursively
    // V(1, K-L+1) => vOff + (k-l)*ldv
    // TAU(K-L+1) => tauOff + (k-l)
    // T(K-L+1, K-L+1) => tOff + (k-l) + (k-l)*ldt
    dlarft(
      direct,
      storev,
      n,
      l,
      v,
      vOff + (k - l) * ldv,
      ldv,
      tau,
      tauOff + (k - l),
      t,
      tOff + (k - l) + (k - l) * ldt,
      ldt
    );

    // T_{2,1} = V_{2,2}'  (transpose: T(K-L+I, J) = V(N-K+J, K-L+I))
    for (let j = 1; j <= k - l; j++) {
      for (let i = 1; i <= l; i++) {
        t[tOff + (k - l + i - 1) + (j - 1) * ldt] =
          v[vOff + (n - k + j - 1) + (k - l + i - 1) * ldv];
      }
    }

    // T_{2,1} = T_{2,1} * V_{2,1}  (V_{2,1} is unit upper triangular at V(N-K+1, 1))
    dtrmm(
      RIGHT,
      UPPER,
      NOTRANS,
      UNIT,
      l,
      k - l,
      1.0,
      v,
      vOff + (n - k),
      ldv,
      t,
      tOff + (k - l),
      ldt
    );

    // T_{2,1} = V_{1,2}' * V_{1,1} + T_{2,1}
    if (n > k) {
      dgemm(
        TRANS,
        NOTRANS,
        l,
        k - l,
        n - k,
        1.0,
        v,
        vOff + (k - l) * ldv,
        ldv,
        v,
        vOff,
        ldv,
        1.0,
        t,
        tOff + (k - l),
        ldt
      );
    }

    // T_{2,1} = -T_{2,2} * T_{2,1}
    dtrmm(
      LEFT,
      LOWER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      -1.0,
      t,
      tOff + (k - l) + (k - l) * ldt,
      ldt,
      t,
      tOff + (k - l),
      ldt
    );

    // T_{2,1} = T_{2,1} * T_{1,1}
    dtrmm(
      RIGHT,
      LOWER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      1.0,
      t,
      tOff,
      ldt,
      t,
      tOff + (k - l),
      ldt
    );
  } else {
    // RQ case: backward + rowwise

    // Compute T_{1,1} recursively
    dlarft(
      direct,
      storev,
      n - l,
      k - l,
      v,
      vOff,
      ldv,
      tau,
      tauOff,
      t,
      tOff,
      ldt
    );

    // Compute T_{2,2} recursively
    // V(K-L+1, 1) => vOff + (k-l)
    // TAU(K-L+1) => tauOff + (k-l)
    // T(K-L+1, K-L+1) => tOff + (k-l) + (k-l)*ldt
    dlarft(
      direct,
      storev,
      n,
      l,
      v,
      vOff + (k - l),
      ldv,
      tau,
      tauOff + (k - l),
      t,
      tOff + (k - l) + (k - l) * ldt,
      ldt
    );

    // T_{2,1} = V_{2,2}  (copy L rows, K-L cols from V(K-L+1, N-K+1))
    dlacpy(
      -1,
      l,
      k - l,
      v,
      vOff + (k - l) + (n - k) * ldv,
      ldv,
      t,
      tOff + (k - l),
      ldt
    );

    // T_{2,1} = T_{2,1} * V_{1,2}'  (V_{1,2} is unit lower triangular at V(1, N-K+1))
    dtrmm(
      RIGHT,
      LOWER,
      TRANS,
      UNIT,
      l,
      k - l,
      1.0,
      v,
      vOff + (n - k) * ldv,
      ldv,
      t,
      tOff + (k - l),
      ldt
    );

    // T_{2,1} = V_{2,1} * V_{1,1}' + T_{2,1}
    if (n > k) {
      dgemm(
        NOTRANS,
        TRANS,
        l,
        k - l,
        n - k,
        1.0,
        v,
        vOff + (k - l),
        ldv,
        v,
        vOff,
        ldv,
        1.0,
        t,
        tOff + (k - l),
        ldt
      );
    }

    // T_{2,1} = -T_{2,2} * T_{2,1}
    dtrmm(
      LEFT,
      LOWER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      -1.0,
      t,
      tOff + (k - l) + (k - l) * ldt,
      ldt,
      t,
      tOff + (k - l),
      ldt
    );

    // T_{2,1} = T_{2,1} * T_{1,1}
    dtrmm(
      RIGHT,
      LOWER,
      NOTRANS,
      NONUNIT,
      l,
      k - l,
      1.0,
      t,
      tOff,
      ldt,
      t,
      tOff + (k - l),
      ldt
    );
  }
}

/**
 * Level-2 BLAS version of DLARFT, used as the base case for the
 * recursive algorithm. Translated from dlarft_lvl2.f.
 */
function dlarft_lvl2(
  direct: number,
  storev: number,
  n: number,
  k: number,
  v: Float64Array,
  vOff: number,
  ldv: number,
  tau: Float64Array,
  tauOff: number,
  t: Float64Array,
  tOff: number,
  ldt: number
): void {
  if (n === 0) return;

  const dirf = direct === FORWARD;
  const colv = storev === COLUMNWISE;

  if (dirf) {
    let prevlastv = n;
    for (let i = 1; i <= k; i++) {
      prevlastv = Math.max(i, prevlastv);
      if (tau[tauOff + (i - 1)] === 0.0) {
        // H(i) = I
        for (let j = 1; j <= i; j++) {
          t[tOff + (j - 1) + (i - 1) * ldt] = 0.0;
        }
      } else {
        // General case
        let lastv: number;
        if (colv) {
          // Skip any trailing zeros in column i of V
          lastv = i;
          for (let lv = n; lv >= i + 1; lv--) {
            if (v[vOff + (lv - 1) + (i - 1) * ldv] !== 0.0) {
              lastv = lv;
              break;
            }
          }

          for (let j = 1; j <= i - 1; j++) {
            // T(J,I) = -TAU(I) * V(I,J)
            t[tOff + (j - 1) + (i - 1) * ldt] =
              -tau[tauOff + (i - 1)] * v[vOff + (i - 1) + (j - 1) * ldv];
          }

          const jj = Math.min(lastv, prevlastv);

          // T(1:i-1,i) := -tau(i) * V(i+1:jj,1:i-1)**T * V(i+1:jj,i) + T(1:i-1,i)
          if (jj > i) {
            dgemv(
              TRANS,
              jj - i,
              i - 1,
              -tau[tauOff + (i - 1)],
              v,
              vOff + i + 0 * ldv,
              ldv,
              v,
              vOff + i + (i - 1) * ldv,
              1,
              1.0,
              t,
              tOff + (i - 1) * ldt,
              1
            );
          }
        } else {
          // Rowwise storage
          // Skip any trailing zeros in row i of V
          lastv = i;
          for (let lv = n; lv >= i + 1; lv--) {
            if (v[vOff + (i - 1) + (lv - 1) * ldv] !== 0.0) {
              lastv = lv;
              break;
            }
          }

          for (let j = 1; j <= i - 1; j++) {
            // T(J,I) = -TAU(I) * V(J,I)
            t[tOff + (j - 1) + (i - 1) * ldt] =
              -tau[tauOff + (i - 1)] * v[vOff + (j - 1) + (i - 1) * ldv];
          }

          const jj = Math.min(lastv, prevlastv);

          // T(1:i-1,i) := -tau(i) * V(1:i-1,i+1:jj) * V(i,i+1:jj)**T + T(1:i-1,i)
          if (jj > i) {
            dgemv(
              NOTRANS,
              i - 1,
              jj - i,
              -tau[tauOff + (i - 1)],
              v,
              vOff + i * ldv,
              ldv,
              v,
              vOff + (i - 1) + i * ldv,
              ldv,
              1.0,
              t,
              tOff + (i - 1) * ldt,
              1
            );
          }
        }

        // T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
        if (i > 1) {
          dtrmv(
            UPPER,
            NOTRANS,
            NONUNIT,
            i - 1,
            t,
            tOff,
            ldt,
            t,
            tOff + (i - 1) * ldt,
            1
          );
        }

        t[tOff + (i - 1) + (i - 1) * ldt] = tau[tauOff + (i - 1)];

        if (i > 1) {
          prevlastv = Math.max(prevlastv, lastv);
        } else {
          prevlastv = lastv;
        }
      }
    }
  } else {
    // Backward direction
    let prevlastv = 1;
    for (let i = k; i >= 1; i--) {
      if (tau[tauOff + (i - 1)] === 0.0) {
        // H(i) = I
        for (let j = i; j <= k; j++) {
          t[tOff + (j - 1) + (i - 1) * ldt] = 0.0;
        }
      } else {
        // General case
        if (i < k) {
          let lastv: number;
          if (colv) {
            // Skip any leading zeros in column i of V
            lastv = i;
            for (let lv = 1; lv <= i - 1; lv++) {
              if (v[vOff + (lv - 1) + (i - 1) * ldv] !== 0.0) {
                lastv = lv;
                break;
              }
            }

            for (let j = i + 1; j <= k; j++) {
              // T(J,I) = -TAU(I) * V(N-K+I, J)
              t[tOff + (j - 1) + (i - 1) * ldt] =
                -tau[tauOff + (i - 1)] *
                v[vOff + (n - k + i - 1) + (j - 1) * ldv];
            }

            const jj = Math.max(lastv, prevlastv);

            // T(i+1:k,i) = -tau(i) * V(jj:n-k+i,i+1:k)**T * V(jj:n-k+i,i) + T(i+1:k,i)
            if (n - k + i > jj) {
              dgemv(
                TRANS,
                n - k + i - jj,
                k - i,
                -tau[tauOff + (i - 1)],
                v,
                vOff + (jj - 1) + i * ldv,
                ldv,
                v,
                vOff + (jj - 1) + (i - 1) * ldv,
                1,
                1.0,
                t,
                tOff + i + (i - 1) * ldt,
                1
              );
            }
          } else {
            // Rowwise storage
            // Skip any leading zeros in row i of V
            lastv = i;
            for (let lv = 1; lv <= i - 1; lv++) {
              if (v[vOff + (i - 1) + (lv - 1) * ldv] !== 0.0) {
                lastv = lv;
                break;
              }
            }

            for (let j = i + 1; j <= k; j++) {
              // T(J,I) = -TAU(I) * V(J, N-K+I)
              t[tOff + (j - 1) + (i - 1) * ldt] =
                -tau[tauOff + (i - 1)] *
                v[vOff + (j - 1) + (n - k + i - 1) * ldv];
            }

            const jj = Math.max(lastv, prevlastv);

            // T(i+1:k,i) = -tau(i) * V(i+1:k,jj:n-k+i) * V(i,jj:n-k+i)**T + T(i+1:k,i)
            if (n - k + i > jj) {
              dgemv(
                NOTRANS,
                k - i,
                n - k + i - jj,
                -tau[tauOff + (i - 1)],
                v,
                vOff + i + (jj - 1) * ldv,
                ldv,
                v,
                vOff + (i - 1) + (jj - 1) * ldv,
                ldv,
                1.0,
                t,
                tOff + i + (i - 1) * ldt,
                1
              );
            }
          }

          // T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i)
          dtrmv(
            LOWER,
            NOTRANS,
            NONUNIT,
            k - i,
            t,
            tOff + i + i * ldt,
            ldt,
            t,
            tOff + i + (i - 1) * ldt,
            1
          );

          if (i > 1) {
            prevlastv = Math.min(prevlastv, lastv);
          } else {
            prevlastv = lastv;
          }
        }
        t[tOff + (i - 1) + (i - 1) * ldt] = tau[tauOff + (i - 1)];
      }
    }
  }
}
