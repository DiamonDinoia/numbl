// Translated from SRC/dlarfb.f
// DLARFB applies a real block reflector H or its transpose H**T to a
// real m by n matrix C, from either the left or the right.
//
// DIRECT parameter: FORWARD=0, BACKWARD=1
// STOREV parameter: COLUMNWISE=0, ROWWISE=1
//
// Array indexing convention (column-major, matching Fortran):
//   C(I,J)    => c[cOff + (I-1) + (J-1)*ldc]
//   V(I,J)    => v[vOff + (I-1) + (J-1)*ldv]
//   T(I,J)    => t[tOff + (I-1) + (J-1)*ldt]
//   WORK(I,J) => work[workOff + (I-1) + (J-1)*ldwork]

import {
  LEFT,
  RIGHT,
  NOTRANS,
  TRANS,
  UPPER,
  LOWER,
  UNIT,
  NONUNIT,
} from "../utils/constants.js";
import { dcopy } from "../BLAS/dcopy.js";
import { dgemm } from "../BLAS/dgemm.js";
import { dtrmm } from "../BLAS/dtrmm.js";

// Constants for DIRECT parameter
const FORWARD = 0;
// const BACKWARD = 1;

// Constants for STOREV parameter
const COLUMNWISE = 0;
// const ROWWISE = 1;

/**
 * DLARFB applies a real block reflector H or its transpose H**T to a
 * real m-by-n matrix C, from either the left or the right.
 *
 * @param side    - LEFT (0) or RIGHT (1)
 * @param trans   - NOTRANS (0) to apply H, TRANS (1) to apply H**T
 * @param direct  - 0=FORWARD, 1=BACKWARD
 * @param storev  - 0=COLUMNWISE, 1=ROWWISE
 * @param m       - Number of rows of C
 * @param n       - Number of columns of C
 * @param k       - Order of the matrix T (number of elementary reflectors)
 * @param v       - The matrix V
 * @param vOff    - Offset into v for V(1,1)
 * @param ldv     - Leading dimension of V
 * @param t       - The triangular factor T
 * @param tOff    - Offset into t for T(1,1)
 * @param ldt     - Leading dimension of T
 * @param c       - The m-by-n matrix C
 * @param cOff    - Offset into c for C(1,1)
 * @param ldc     - Leading dimension of C
 * @param work    - Workspace, dimension (ldwork, k)
 * @param workOff - Offset into work
 * @param ldwork  - Leading dimension of work (>= n if side=LEFT, >= m if side=RIGHT)
 */
export function dlarfb(
  side: number,
  trans: number,
  direct: number,
  storev: number,
  m: number,
  n: number,
  k: number,
  v: Float64Array,
  vOff: number,
  ldv: number,
  t: Float64Array,
  tOff: number,
  ldt: number,
  c: Float64Array,
  cOff: number,
  ldc: number,
  work: Float64Array,
  workOff: number,
  ldwork: number
): void {
  // Quick return if possible
  if (m <= 0 || n <= 0) return;

  // When TRANS='N', we need TRANST='T' and vice versa
  const transt = trans === NOTRANS ? TRANS : NOTRANS;

  if (storev === COLUMNWISE) {
    if (direct === FORWARD) {
      // Let  V = ( V1 )    (first K rows)
      //          ( V2 )
      // where V1 is unit lower triangular.

      if (side === LEFT) {
        // Form  H * C  or  H**T * C  where  C = ( C1 )
        //                                       ( C2 )
        //
        // W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)

        // W := C1**T
        for (let j = 1; j <= k; j++) {
          dcopy(n, c, cOff + (j - 1), ldc, work, workOff + (j - 1) * ldwork, 1);
        }

        // W := W * V1
        dtrmm(
          RIGHT,
          LOWER,
          NOTRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        if (m > k) {
          // W := W + C2**T * V2
          dgemm(
            TRANS,
            NOTRANS,
            n,
            k,
            m - k,
            1.0,
            c,
            cOff + k,
            ldc,
            v,
            vOff + k,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T**T  or  W * T
        dtrmm(
          RIGHT,
          UPPER,
          transt,
          NONUNIT,
          n,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - V * W**T
        if (m > k) {
          // C2 := C2 - V2 * W**T
          dgemm(
            NOTRANS,
            TRANS,
            m - k,
            n,
            k,
            -1.0,
            v,
            vOff + k,
            ldv,
            work,
            workOff,
            ldwork,
            1.0,
            c,
            cOff + k,
            ldc
          );
        }

        // W := W * V1**T
        dtrmm(
          RIGHT,
          LOWER,
          TRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        // C1 := C1 - W**T
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= n; i++) {
            c[cOff + (j - 1) + (i - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      } else {
        // side === RIGHT
        // Form  C * H  or  C * H**T  where  C = ( C1  C2 )
        //
        // W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)

        // W := C1
        for (let j = 1; j <= k; j++) {
          dcopy(
            m,
            c,
            cOff + (j - 1) * ldc,
            1,
            work,
            workOff + (j - 1) * ldwork,
            1
          );
        }

        // W := W * V1
        dtrmm(
          RIGHT,
          LOWER,
          NOTRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        if (n > k) {
          // W := W + C2 * V2
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            k,
            n - k,
            1.0,
            c,
            cOff + k * ldc,
            ldc,
            v,
            vOff + k,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T  or  W * T**T
        dtrmm(
          RIGHT,
          UPPER,
          trans,
          NONUNIT,
          m,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - W * V**T
        if (n > k) {
          // C2 := C2 - W * V2**T
          dgemm(
            NOTRANS,
            TRANS,
            m,
            n - k,
            k,
            -1.0,
            work,
            workOff,
            ldwork,
            v,
            vOff + k,
            ldv,
            1.0,
            c,
            cOff + k * ldc,
            ldc
          );
        }

        // W := W * V1**T
        dtrmm(
          RIGHT,
          LOWER,
          TRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        // C1 := C1 - W
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      }
    } else {
      // DIRECT = BACKWARD
      // Let  V = ( V1 )
      //          ( V2 )    (last K rows)
      // where V2 is unit upper triangular.

      if (side === LEFT) {
        // Form  H * C  or  H**T * C  where  C = ( C1 )
        //                                       ( C2 )
        //
        // W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)

        // W := C2**T
        for (let j = 1; j <= k; j++) {
          dcopy(
            n,
            c,
            cOff + (m - k + j - 1),
            ldc,
            work,
            workOff + (j - 1) * ldwork,
            1
          );
        }

        // W := W * V2
        dtrmm(
          RIGHT,
          UPPER,
          NOTRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff + (m - k),
          ldv,
          work,
          workOff,
          ldwork
        );

        if (m > k) {
          // W := W + C1**T * V1
          dgemm(
            TRANS,
            NOTRANS,
            n,
            k,
            m - k,
            1.0,
            c,
            cOff,
            ldc,
            v,
            vOff,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T**T  or  W * T
        dtrmm(
          RIGHT,
          LOWER,
          transt,
          NONUNIT,
          n,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - V * W**T
        if (m > k) {
          // C1 := C1 - V1 * W**T
          dgemm(
            NOTRANS,
            TRANS,
            m - k,
            n,
            k,
            -1.0,
            v,
            vOff,
            ldv,
            work,
            workOff,
            ldwork,
            1.0,
            c,
            cOff,
            ldc
          );
        }

        // W := W * V2**T
        dtrmm(
          RIGHT,
          UPPER,
          TRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff + (m - k),
          ldv,
          work,
          workOff,
          ldwork
        );

        // C2 := C2 - W**T
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= n; i++) {
            c[cOff + (m - k + j - 1) + (i - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      } else {
        // side === RIGHT
        // Form  C * H  or  C * H**T  where  C = ( C1  C2 )
        //
        // W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)

        // W := C2
        for (let j = 1; j <= k; j++) {
          dcopy(
            m,
            c,
            cOff + (n - k + j - 1) * ldc,
            1,
            work,
            workOff + (j - 1) * ldwork,
            1
          );
        }

        // W := W * V2
        dtrmm(
          RIGHT,
          UPPER,
          NOTRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff + (n - k),
          ldv,
          work,
          workOff,
          ldwork
        );

        if (n > k) {
          // W := W + C1 * V1
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            k,
            n - k,
            1.0,
            c,
            cOff,
            ldc,
            v,
            vOff,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T  or  W * T**T
        dtrmm(
          RIGHT,
          LOWER,
          trans,
          NONUNIT,
          m,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - W * V**T
        if (n > k) {
          // C1 := C1 - W * V1**T
          dgemm(
            NOTRANS,
            TRANS,
            m,
            n - k,
            k,
            -1.0,
            work,
            workOff,
            ldwork,
            v,
            vOff,
            ldv,
            1.0,
            c,
            cOff,
            ldc
          );
        }

        // W := W * V2**T
        dtrmm(
          RIGHT,
          UPPER,
          TRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff + (n - k),
          ldv,
          work,
          workOff,
          ldwork
        );

        // C2 := C2 - W
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (n - k + j - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      }
    }
  } else {
    // STOREV = ROWWISE
    if (direct === FORWARD) {
      // Let  V = ( V1  V2 )    (V1: first K columns)
      // where V1 is unit upper triangular.

      if (side === LEFT) {
        // Form  H * C  or  H**T * C  where  C = ( C1 )
        //                                       ( C2 )
        //
        // W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)

        // W := C1**T
        for (let j = 1; j <= k; j++) {
          dcopy(n, c, cOff + (j - 1), ldc, work, workOff + (j - 1) * ldwork, 1);
        }

        // W := W * V1**T
        dtrmm(
          RIGHT,
          UPPER,
          TRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        if (m > k) {
          // W := W + C2**T * V2**T
          dgemm(
            TRANS,
            TRANS,
            n,
            k,
            m - k,
            1.0,
            c,
            cOff + k,
            ldc,
            v,
            vOff + k * ldv,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T**T  or  W * T
        dtrmm(
          RIGHT,
          UPPER,
          transt,
          NONUNIT,
          n,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - V**T * W**T
        if (m > k) {
          // C2 := C2 - V2**T * W**T
          dgemm(
            TRANS,
            TRANS,
            m - k,
            n,
            k,
            -1.0,
            v,
            vOff + k * ldv,
            ldv,
            work,
            workOff,
            ldwork,
            1.0,
            c,
            cOff + k,
            ldc
          );
        }

        // W := W * V1
        dtrmm(
          RIGHT,
          UPPER,
          NOTRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        // C1 := C1 - W**T
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= n; i++) {
            c[cOff + (j - 1) + (i - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      } else {
        // side === RIGHT
        // Form  C * H  or  C * H**T  where  C = ( C1  C2 )
        //
        // W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)

        // W := C1
        for (let j = 1; j <= k; j++) {
          dcopy(
            m,
            c,
            cOff + (j - 1) * ldc,
            1,
            work,
            workOff + (j - 1) * ldwork,
            1
          );
        }

        // W := W * V1**T
        dtrmm(
          RIGHT,
          UPPER,
          TRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        if (n > k) {
          // W := W + C2 * V2**T
          dgemm(
            NOTRANS,
            TRANS,
            m,
            k,
            n - k,
            1.0,
            c,
            cOff + k * ldc,
            ldc,
            v,
            vOff + k * ldv,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T  or  W * T**T
        dtrmm(
          RIGHT,
          UPPER,
          trans,
          NONUNIT,
          m,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - W * V
        if (n > k) {
          // C2 := C2 - W * V2
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n - k,
            k,
            -1.0,
            work,
            workOff,
            ldwork,
            v,
            vOff + k * ldv,
            ldv,
            1.0,
            c,
            cOff + k * ldc,
            ldc
          );
        }

        // W := W * V1
        dtrmm(
          RIGHT,
          UPPER,
          NOTRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff,
          ldv,
          work,
          workOff,
          ldwork
        );

        // C1 := C1 - W
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (j - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      }
    } else {
      // DIRECT = BACKWARD
      // Let  V = ( V1  V2 )    (V2: last K columns)
      // where V2 is unit lower triangular.

      if (side === LEFT) {
        // Form  H * C  or  H**T * C  where  C = ( C1 )
        //                                       ( C2 )
        //
        // W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)

        // W := C2**T
        for (let j = 1; j <= k; j++) {
          dcopy(
            n,
            c,
            cOff + (m - k + j - 1),
            ldc,
            work,
            workOff + (j - 1) * ldwork,
            1
          );
        }

        // W := W * V2**T
        dtrmm(
          RIGHT,
          LOWER,
          TRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff + (m - k) * ldv,
          ldv,
          work,
          workOff,
          ldwork
        );

        if (m > k) {
          // W := W + C1**T * V1**T
          dgemm(
            TRANS,
            TRANS,
            n,
            k,
            m - k,
            1.0,
            c,
            cOff,
            ldc,
            v,
            vOff,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T**T  or  W * T
        dtrmm(
          RIGHT,
          LOWER,
          transt,
          NONUNIT,
          n,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - V**T * W**T
        if (m > k) {
          // C1 := C1 - V1**T * W**T
          dgemm(
            TRANS,
            TRANS,
            m - k,
            n,
            k,
            -1.0,
            v,
            vOff,
            ldv,
            work,
            workOff,
            ldwork,
            1.0,
            c,
            cOff,
            ldc
          );
        }

        // W := W * V2
        dtrmm(
          RIGHT,
          LOWER,
          NOTRANS,
          UNIT,
          n,
          k,
          1.0,
          v,
          vOff + (m - k) * ldv,
          ldv,
          work,
          workOff,
          ldwork
        );

        // C2 := C2 - W**T
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= n; i++) {
            c[cOff + (m - k + j - 1) + (i - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      } else {
        // side === RIGHT
        // Form  C * H  or  C * H**T  where  C = ( C1  C2 )
        //
        // W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)

        // W := C2
        for (let j = 1; j <= k; j++) {
          dcopy(
            m,
            c,
            cOff + (n - k + j - 1) * ldc,
            1,
            work,
            workOff + (j - 1) * ldwork,
            1
          );
        }

        // W := W * V2**T
        dtrmm(
          RIGHT,
          LOWER,
          TRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff + (n - k) * ldv,
          ldv,
          work,
          workOff,
          ldwork
        );

        if (n > k) {
          // W := W + C1 * V1**T
          dgemm(
            NOTRANS,
            TRANS,
            m,
            k,
            n - k,
            1.0,
            c,
            cOff,
            ldc,
            v,
            vOff,
            ldv,
            1.0,
            work,
            workOff,
            ldwork
          );
        }

        // W := W * T  or  W * T**T
        dtrmm(
          RIGHT,
          LOWER,
          trans,
          NONUNIT,
          m,
          k,
          1.0,
          t,
          tOff,
          ldt,
          work,
          workOff,
          ldwork
        );

        // C := C - W * V
        if (n > k) {
          // C1 := C1 - W * V1
          dgemm(
            NOTRANS,
            NOTRANS,
            m,
            n - k,
            k,
            -1.0,
            work,
            workOff,
            ldwork,
            v,
            vOff,
            ldv,
            1.0,
            c,
            cOff,
            ldc
          );
        }

        // W := W * V2
        dtrmm(
          RIGHT,
          LOWER,
          NOTRANS,
          UNIT,
          m,
          k,
          1.0,
          v,
          vOff + (n - k) * ldv,
          ldv,
          work,
          workOff,
          ldwork
        );

        // C2 := C2 - W
        for (let j = 1; j <= k; j++) {
          for (let i = 1; i <= m; i++) {
            c[cOff + (i - 1) + (n - k + j - 1) * ldc] -=
              work[workOff + (i - 1) + (j - 1) * ldwork];
          }
        }
      }
    }
  }
}
