// Apply block reflector H = I - V * T * V' to matrix C from the left.
// C := H * C = C - V * T * (V' * C)
//
// V is m×k lower trapezoidal with unit diagonal.
// T is k×k upper triangular.
// C is m×n.
//
// Uses dgemm for the two matrix-matrix multiplications.

import { dgemm } from "../blas/dgemm.js";

// Work buffer reused across calls
let _work = new Float64Array(0);

export function dlarfb(
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
  ldc: number
): void {
  if (m === 0 || n === 0 || k === 0) return;

  // W is k×n workspace
  const wSize = k * n;
  if (_work.length < wSize) _work = new Float64Array(wSize);
  const W = _work;

  // Step 1: W = V' * C  (k×n = k×m * m×n)
  // V is lower trapezoidal: V(i,j)=0 for i<j, V(i,i)=1
  // Split: W = V(0:k-1, 0:k-1)' * C(0:k-1, :) + V(k:m-1, 0:k-1)' * C(k:m-1, :)
  // The first part uses the unit lower triangular block

  // W = C(0:k-1, :)' transposed into W, accounting for unit lower triangular V
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < k; i++) {
      W[i + j * k] = c[cOff + i + j * ldc];
    }
  }
  // W += lower-triangular part of V(0:k-1,:)' * C(0:k-1,:)
  // V(row, col) for row > col: add V(row,col) * C(row, j) to W(col, j)
  for (let j = 0; j < n; j++) {
    for (let col = 0; col < k; col++) {
      let s = 0;
      for (let row = col + 1; row < k; row++) {
        s += v[vOff + row + col * ldv] * c[cOff + row + j * ldc];
      }
      W[col + j * k] += s;
    }
  }

  // Add V(k:m-1, :)' * C(k:m-1, :) via dgemm if there are rows below k
  if (m > k) {
    // W += V2' * C2 where V2 = V(k:m-1, :), C2 = C(k:m-1, :)
    // This is k×n += (m-k)×k' * (m-k)×n, i.e. need transpose of V2
    // dgemm does A*B, we need V2'*C2. Transpose V2 into temp.
    const v2rows = m - k;
    const tmpSize = k * v2rows;
    // Use a section of work buffer or allocate
    const V2T = new Float64Array(tmpSize);
    for (let j = 0; j < k; j++)
      for (let i = 0; i < v2rows; i++)
        V2T[j + i * k] = v[vOff + (k + i) + j * ldv];

    dgemm(k, n, v2rows, 1.0, V2T, 0, k, c, cOff + k, ldc, 1.0, W, 0, k);
  }

  // Step 2: W = T * W  (k×n, T is k×k upper triangular)
  // In-place upper triangular multiply: row by row from top
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < k; i++) {
      let s = 0;
      for (let p = i; p < k; p++) {
        s += t[tOff + i + p * ldt] * W[p + j * k];
      }
      W[i + j * k] = s;
    }
  }

  // Step 3: C = C - V * W  (m×n -= m×k * k×n)
  // Upper part (0:k-1): unit lower triangular V
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < k; i++) {
      c[cOff + i + j * ldc] -= W[i + j * k]; // V(i,i)=1
    }
    for (let col = 0; col < k; col++) {
      const w = W[col + j * k];
      for (let row = col + 1; row < k; row++) {
        c[cOff + row + j * ldc] -= v[vOff + row + col * ldv] * w;
      }
    }
  }

  // Lower part (k:m-1): V2 * W via dgemm
  if (m > k) {
    dgemm(m - k, n, k, -1.0, v, vOff + k, ldv, W, 0, k, 1.0, c, cOff + k, ldc);
  }
}
