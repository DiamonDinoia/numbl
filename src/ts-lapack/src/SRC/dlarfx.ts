// Translated from SRC/dlarfx.f
// DLARFX applies a real elementary reflector H to a real m by n
// matrix C, from either the left or the right. H is represented in the
// form
//
//       H = I - tau * v * v**T
//
// where tau is a real scalar and v is a real vector.
//
// If tau = 0, then H is taken to be the unit matrix.
//
// This version uses inline code if H has order < 11.
//
// Array indexing convention (column-major, matching Fortran):
//   C(I,J)  =>  c[cOff + (I-1) + (J-1)*ldc]   (I,J are 1-based)
//   V(I)    =>  v[vOff + (I-1)]                 (I is 1-based)

import { LEFT } from "../utils/constants.js";
import { dlarf } from "./dlarf.js";

/**
 * DLARFX applies a real elementary reflector H to a real m-by-n matrix C,
 * from either the left or the right, with loop unrolling for small sizes.
 *
 * @param side    - LEFT (0) to form H*C, RIGHT (1) to form C*H
 * @param m       - Number of rows of C
 * @param n       - Number of columns of C
 * @param v       - The vector v in H = I - tau*v*v'
 * @param vOff    - Offset into v for V(1)
 * @param tau     - The scalar tau
 * @param c       - The m-by-n matrix C (column-major)
 * @param cOff    - Offset into c for C(1,1)
 * @param ldc     - Leading dimension of c
 * @param work    - Workspace (not referenced if H has order < 11)
 * @param workOff - Offset into work
 */
export function dlarfx(
  side: number,
  m: number,
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number,
  work: Float64Array,
  workOff: number
): void {
  if (tau === 0.0) return;

  if (side === LEFT) {
    // Form  H * C, where H has order m.
    switch (m) {
      case 1:
        dlarfx_left_1(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 2:
        dlarfx_left_2(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 3:
        dlarfx_left_3(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 4:
        dlarfx_left_4(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 5:
        dlarfx_left_5(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 6:
        dlarfx_left_6(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 7:
        dlarfx_left_7(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 8:
        dlarfx_left_8(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 9:
        dlarfx_left_9(n, v, vOff, tau, c, cOff, ldc);
        return;
      case 10:
        dlarfx_left_10(n, v, vOff, tau, c, cOff, ldc);
        return;
      default:
        // Code for general M
        dlarf(side, m, n, v, vOff, 1, tau, c, cOff, ldc, work, workOff);
        return;
    }
  } else {
    // Form  C * H, where H has order n.
    switch (n) {
      case 1:
        dlarfx_right_1(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 2:
        dlarfx_right_2(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 3:
        dlarfx_right_3(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 4:
        dlarfx_right_4(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 5:
        dlarfx_right_5(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 6:
        dlarfx_right_6(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 7:
        dlarfx_right_7(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 8:
        dlarfx_right_8(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 9:
        dlarfx_right_9(m, v, vOff, tau, c, cOff, ldc);
        return;
      case 10:
        dlarfx_right_10(m, v, vOff, tau, c, cOff, ldc);
        return;
      default:
        // Code for general N
        dlarf(side, m, n, v, vOff, 1, tau, c, cOff, ldc, work, workOff);
        return;
    }
  }
}

// ============================================================
// Special-cased left applications: H * C  (H has order m)
// For left application, C(I,J) = c[cOff + (I-1) + (J-1)*ldc]
// ============================================================

function dlarfx_left_1(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const t1 = 1.0 - tau * v[vOff] * v[vOff];
  for (let j = 0; j < n; j++) {
    c[cOff + j * ldc] = t1 * c[cOff + j * ldc];
  }
}

function dlarfx_left_2(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff];
  const t1 = tau * v1;
  const v2 = v[vOff + 1];
  const t2 = tau * v2;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum = v1 * c[col] + v2 * c[col + 1];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
  }
}

function dlarfx_left_3(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum = v1 * c[col] + v2 * c[col + 1] + v3 * c[col + 2];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
  }
}

function dlarfx_left_4(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum =
      v1 * c[col] + v2 * c[col + 1] + v3 * c[col + 2] + v4 * c[col + 3];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
    c[col + 3] -= sum * t4;
  }
}

function dlarfx_left_5(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum =
      v1 * c[col] +
      v2 * c[col + 1] +
      v3 * c[col + 2] +
      v4 * c[col + 3] +
      v5 * c[col + 4];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
    c[col + 3] -= sum * t4;
    c[col + 4] -= sum * t5;
  }
}

function dlarfx_left_6(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum =
      v1 * c[col] +
      v2 * c[col + 1] +
      v3 * c[col + 2] +
      v4 * c[col + 3] +
      v5 * c[col + 4] +
      v6 * c[col + 5];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
    c[col + 3] -= sum * t4;
    c[col + 4] -= sum * t5;
    c[col + 5] -= sum * t6;
  }
}

function dlarfx_left_7(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum =
      v1 * c[col] +
      v2 * c[col + 1] +
      v3 * c[col + 2] +
      v4 * c[col + 3] +
      v5 * c[col + 4] +
      v6 * c[col + 5] +
      v7 * c[col + 6];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
    c[col + 3] -= sum * t4;
    c[col + 4] -= sum * t5;
    c[col + 5] -= sum * t6;
    c[col + 6] -= sum * t7;
  }
}

function dlarfx_left_8(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  const v8 = v[vOff + 7],
    t8 = tau * v8;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum =
      v1 * c[col] +
      v2 * c[col + 1] +
      v3 * c[col + 2] +
      v4 * c[col + 3] +
      v5 * c[col + 4] +
      v6 * c[col + 5] +
      v7 * c[col + 6] +
      v8 * c[col + 7];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
    c[col + 3] -= sum * t4;
    c[col + 4] -= sum * t5;
    c[col + 5] -= sum * t6;
    c[col + 6] -= sum * t7;
    c[col + 7] -= sum * t8;
  }
}

function dlarfx_left_9(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  const v8 = v[vOff + 7],
    t8 = tau * v8;
  const v9 = v[vOff + 8],
    t9 = tau * v9;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum =
      v1 * c[col] +
      v2 * c[col + 1] +
      v3 * c[col + 2] +
      v4 * c[col + 3] +
      v5 * c[col + 4] +
      v6 * c[col + 5] +
      v7 * c[col + 6] +
      v8 * c[col + 7] +
      v9 * c[col + 8];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
    c[col + 3] -= sum * t4;
    c[col + 4] -= sum * t5;
    c[col + 5] -= sum * t6;
    c[col + 6] -= sum * t7;
    c[col + 7] -= sum * t8;
    c[col + 8] -= sum * t9;
  }
}

function dlarfx_left_10(
  n: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  const v8 = v[vOff + 7],
    t8 = tau * v8;
  const v9 = v[vOff + 8],
    t9 = tau * v9;
  const v10 = v[vOff + 9],
    t10 = tau * v10;
  for (let j = 0; j < n; j++) {
    const col = cOff + j * ldc;
    const sum =
      v1 * c[col] +
      v2 * c[col + 1] +
      v3 * c[col + 2] +
      v4 * c[col + 3] +
      v5 * c[col + 4] +
      v6 * c[col + 5] +
      v7 * c[col + 6] +
      v8 * c[col + 7] +
      v9 * c[col + 8] +
      v10 * c[col + 9];
    c[col] -= sum * t1;
    c[col + 1] -= sum * t2;
    c[col + 2] -= sum * t3;
    c[col + 3] -= sum * t4;
    c[col + 4] -= sum * t5;
    c[col + 5] -= sum * t6;
    c[col + 6] -= sum * t7;
    c[col + 7] -= sum * t8;
    c[col + 8] -= sum * t9;
    c[col + 9] -= sum * t10;
  }
}

// ============================================================
// Special-cased right applications: C * H  (H has order n)
// For right application, C(J,I) = c[cOff + (J-1) + (I-1)*ldc]
// ============================================================

function dlarfx_right_1(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  _ldc: number // eslint-disable-line @typescript-eslint/no-unused-vars
): void {
  const t1 = 1.0 - tau * v[vOff] * v[vOff];
  for (let j = 0; j < m; j++) {
    c[cOff + j] = t1 * c[cOff + j];
  }
}

function dlarfx_right_2(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  for (let j = 0; j < m; j++) {
    const sum = v1 * c[cOff + j] + v2 * c[cOff + j + ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
  }
}

function dlarfx_right_3(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] + v2 * c[cOff + j + ldc] + v3 * c[cOff + j + 2 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
  }
}

function dlarfx_right_4(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] +
      v2 * c[cOff + j + ldc] +
      v3 * c[cOff + j + 2 * ldc] +
      v4 * c[cOff + j + 3 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
    c[cOff + j + 3 * ldc] -= sum * t4;
  }
}

function dlarfx_right_5(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] +
      v2 * c[cOff + j + ldc] +
      v3 * c[cOff + j + 2 * ldc] +
      v4 * c[cOff + j + 3 * ldc] +
      v5 * c[cOff + j + 4 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
    c[cOff + j + 3 * ldc] -= sum * t4;
    c[cOff + j + 4 * ldc] -= sum * t5;
  }
}

function dlarfx_right_6(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] +
      v2 * c[cOff + j + ldc] +
      v3 * c[cOff + j + 2 * ldc] +
      v4 * c[cOff + j + 3 * ldc] +
      v5 * c[cOff + j + 4 * ldc] +
      v6 * c[cOff + j + 5 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
    c[cOff + j + 3 * ldc] -= sum * t4;
    c[cOff + j + 4 * ldc] -= sum * t5;
    c[cOff + j + 5 * ldc] -= sum * t6;
  }
}

function dlarfx_right_7(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] +
      v2 * c[cOff + j + ldc] +
      v3 * c[cOff + j + 2 * ldc] +
      v4 * c[cOff + j + 3 * ldc] +
      v5 * c[cOff + j + 4 * ldc] +
      v6 * c[cOff + j + 5 * ldc] +
      v7 * c[cOff + j + 6 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
    c[cOff + j + 3 * ldc] -= sum * t4;
    c[cOff + j + 4 * ldc] -= sum * t5;
    c[cOff + j + 5 * ldc] -= sum * t6;
    c[cOff + j + 6 * ldc] -= sum * t7;
  }
}

function dlarfx_right_8(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  const v8 = v[vOff + 7],
    t8 = tau * v8;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] +
      v2 * c[cOff + j + ldc] +
      v3 * c[cOff + j + 2 * ldc] +
      v4 * c[cOff + j + 3 * ldc] +
      v5 * c[cOff + j + 4 * ldc] +
      v6 * c[cOff + j + 5 * ldc] +
      v7 * c[cOff + j + 6 * ldc] +
      v8 * c[cOff + j + 7 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
    c[cOff + j + 3 * ldc] -= sum * t4;
    c[cOff + j + 4 * ldc] -= sum * t5;
    c[cOff + j + 5 * ldc] -= sum * t6;
    c[cOff + j + 6 * ldc] -= sum * t7;
    c[cOff + j + 7 * ldc] -= sum * t8;
  }
}

function dlarfx_right_9(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  const v8 = v[vOff + 7],
    t8 = tau * v8;
  const v9 = v[vOff + 8],
    t9 = tau * v9;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] +
      v2 * c[cOff + j + ldc] +
      v3 * c[cOff + j + 2 * ldc] +
      v4 * c[cOff + j + 3 * ldc] +
      v5 * c[cOff + j + 4 * ldc] +
      v6 * c[cOff + j + 5 * ldc] +
      v7 * c[cOff + j + 6 * ldc] +
      v8 * c[cOff + j + 7 * ldc] +
      v9 * c[cOff + j + 8 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
    c[cOff + j + 3 * ldc] -= sum * t4;
    c[cOff + j + 4 * ldc] -= sum * t5;
    c[cOff + j + 5 * ldc] -= sum * t6;
    c[cOff + j + 6 * ldc] -= sum * t7;
    c[cOff + j + 7 * ldc] -= sum * t8;
    c[cOff + j + 8 * ldc] -= sum * t9;
  }
}

function dlarfx_right_10(
  m: number,
  v: Float64Array,
  vOff: number,
  tau: number,
  c: Float64Array,
  cOff: number,
  ldc: number
): void {
  const v1 = v[vOff],
    t1 = tau * v1;
  const v2 = v[vOff + 1],
    t2 = tau * v2;
  const v3 = v[vOff + 2],
    t3 = tau * v3;
  const v4 = v[vOff + 3],
    t4 = tau * v4;
  const v5 = v[vOff + 4],
    t5 = tau * v5;
  const v6 = v[vOff + 5],
    t6 = tau * v6;
  const v7 = v[vOff + 6],
    t7 = tau * v7;
  const v8 = v[vOff + 7],
    t8 = tau * v8;
  const v9 = v[vOff + 8],
    t9 = tau * v9;
  const v10 = v[vOff + 9],
    t10 = tau * v10;
  for (let j = 0; j < m; j++) {
    const sum =
      v1 * c[cOff + j] +
      v2 * c[cOff + j + ldc] +
      v3 * c[cOff + j + 2 * ldc] +
      v4 * c[cOff + j + 3 * ldc] +
      v5 * c[cOff + j + 4 * ldc] +
      v6 * c[cOff + j + 5 * ldc] +
      v7 * c[cOff + j + 6 * ldc] +
      v8 * c[cOff + j + 7 * ldc] +
      v9 * c[cOff + j + 8 * ldc] +
      v10 * c[cOff + j + 9 * ldc];
    c[cOff + j] -= sum * t1;
    c[cOff + j + ldc] -= sum * t2;
    c[cOff + j + 2 * ldc] -= sum * t3;
    c[cOff + j + 3 * ldc] -= sum * t4;
    c[cOff + j + 4 * ldc] -= sum * t5;
    c[cOff + j + 5 * ldc] -= sum * t6;
    c[cOff + j + 6 * ldc] -= sum * t7;
    c[cOff + j + 7 * ldc] -= sum * t8;
    c[cOff + j + 8 * ldc] -= sum * t9;
    c[cOff + j + 9 * ldc] -= sum * t10;
  }
}
