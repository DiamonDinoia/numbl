// Translated from SRC/dlarf1f.f
// DLARF1F applies a real elementary reflector H to a real m by n matrix C,
// from either the left or the right, assuming v(1) = 1 (not referenced or
// modified).  H is represented in the form
//
//       H = I - tau * v * v**T
//
// where tau is a real scalar and v is a real vector with v(1) = 1.
// If tau = 0, then H is taken to be the unit matrix.
//
// Array indexing convention (matching Fortran column-major):
//   C(I,J)  =>  c[cOff + (I-1) + (J-1)*ldc]   (I,J are 1-based)
//   V(I)    =>  v[vOff + (I-1)*incv]            (I is 1-based)
//   WORK(I) =>  work[workOff + (I-1)]           (I is 1-based)

import { iladlr } from "../utils/iladlr.js";
import { iladlc } from "../utils/iladlc.js";
import { dgemv } from "../BLAS/dgemv.js";
import { dger } from "../BLAS/dger.js";
import { daxpy } from "../BLAS/daxpy.js";
import { dscal } from "../BLAS/dscal.js";
import { LEFT, NOTRANS, TRANS } from "../utils/constants.js";

export function dlarf1f(
  side: number,
  m: number,
  n: number,
  v: Float64Array,
  vOff: number,
  incv: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number,
  work: Float64Array,
  workOff: number
): void {
  const applyleft = side === LEFT;
  let lastv = 1;
  let lastc = 0;

  if (tau !== 0.0) {
    // Set up variables for scanning V. LASTV begins pointing to the end of V.
    // Since V(1) = 1 is not stored, we start scanning from the last element.
    if (applyleft) {
      lastv = m;
    } else {
      lastv = n;
    }

    let i: number;
    if (incv > 0) {
      i = 1 + (lastv - 1) * incv;
    } else {
      i = 1;
    }

    // Look for the last non-zero row in V.
    // Since V(1) = 1 is implicit, we don't access it; scan from LASTV down to 2.
    while (lastv > 1 && v[vOff + (i - 1)] === 0.0) {
      lastv--;
      i -= incv;
    }

    if (applyleft) {
      // Scan for the last non-zero column in C(1:lastv, :)
      lastc = iladlc(lastv, n, c, cOff, ldc);
    } else {
      // Scan for the last non-zero row in C(:, 1:lastv)
      lastc = iladlr(m, lastv, c, cOff, ldc);
    }
  }

  if (lastc === 0) return;

  if (applyleft) {
    // Form  H * C
    if (lastv === 1) {
      // v = [1], so H = I - tau * e1 * e1' applied to C(1,1:lastc)
      // C(1,1:lastc) := (1 - tau) * C(1,1:lastc)
      // C(1,:) is stored with stride ldc (row 1, each column)
      dscal(lastc, 1.0 - tau, c, cOff, ldc);
    } else {
      // w(1:lastc) := C(2:lastv, 1:lastc)**T * v(2:lastv)
      // C(2,1) is at cOff+1; C has ldc as leading dim; LASTV-1 rows, LASTC cols
      dgemv(
        TRANS,
        lastv - 1,
        lastc,
        1.0,
        c,
        cOff + 1,
        ldc,
        v,
        vOff + incv,
        incv,
        0.0,
        work,
        workOff,
        1
      );
      // w(1:lastc) += C(1,1:lastc)**T * v(1) = C(1,1:lastc)**T  (v(1)=1)
      // C(1,:) has stride ldc
      daxpy(lastc, 1.0, c, cOff, ldc, work, workOff, 1);

      // C(1,1:lastc) := C(1,...) - tau * v(1) * w(1:lastc)**T = C(1,...) - tau * w**T
      daxpy(lastc, -tau, work, workOff, 1, c, cOff, ldc);

      // C(2:lastv,1:lastc) := C(...) - tau * v(2:lastv) * w(1:lastc)**T
      dger(
        lastv - 1,
        lastc,
        -tau,
        v,
        vOff + incv,
        incv,
        work,
        workOff,
        1,
        c,
        cOff + 1,
        ldc
      );
    }
  } else {
    // Form  C * H
    if (lastv === 1) {
      // v = [1], so H = I - tau * e1 * e1' applied to C(1:lastc,1)
      // C(1:lastc,1) := (1 - tau) * C(1:lastc,1)
      // C(:,1) is stored with stride 1
      dscal(lastc, 1.0 - tau, c, cOff, 1);
    } else {
      // w(1:lastc) := C(1:lastc, 2:lastv) * v(2:lastv)
      // C(1,2) is at cOff+ldc; lastc rows, lastv-1 cols
      dgemv(
        NOTRANS,
        lastc,
        lastv - 1,
        1.0,
        c,
        cOff + ldc,
        ldc,
        v,
        vOff + incv,
        incv,
        0.0,
        work,
        workOff,
        1
      );
      // w(1:lastc) += C(1:lastc,1) * v(1) = C(1:lastc,1)  (v(1)=1)
      // C(:,1) has stride 1
      daxpy(lastc, 1.0, c, cOff, 1, work, workOff, 1);

      // C(1:lastc,1) := C(...) - tau * w(1:lastc) * v(1)**T = C(...) - tau * w
      daxpy(lastc, -tau, work, workOff, 1, c, cOff, 1);

      // C(1:lastc,2:lastv) := C(...) - tau * w(1:lastc) * v(2:lastv)**T
      dger(
        lastc,
        lastv - 1,
        -tau,
        work,
        workOff,
        1,
        v,
        vOff + incv,
        incv,
        c,
        cOff + ldc,
        ldc
      );
    }
  }
}
