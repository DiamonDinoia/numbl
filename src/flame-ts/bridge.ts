// FLAME-TS bridge: assembles a LapackBridge using blocked BLIS/FLAME routines
// where they help (dgemm, LU, Cholesky) and falls back to ts-lapack for the rest.

import type { LapackBridge } from "../numbl-core/native/lapack-bridge.js";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";
import { matmul } from "./blas/dgemm.js";
import { dgetrf } from "./lapack/dgetrf.js";
import { dpotrf } from "./lapack/dpotrf.js";
import { dtrsmLUNN, dtrsmLLNUnit } from "./blas/dtrsm.js";

const tsBridge = getTsLapackBridge();

function inv(data: Float64Array, n: number): Float64Array {
  const a = new Float64Array(data);
  const ipiv = new Int32Array(n);
  const info = dgetrf(n, n, a, 0, n, ipiv);
  if (info > 0) throw new Error("inv: matrix is singular (dgetrf)");

  // Compute inverse by solving A * X = I column by column using LU factors
  const eye = new Float64Array(n * n);
  for (let i = 0; i < n; i++) eye[i + i * n] = 1;

  // Apply row permutations to identity
  for (let i = 0; i < n; i++) {
    const pi = ipiv[i] - 1;
    if (pi !== i) {
      for (let c = 0; c < n; c++) {
        const tmp = eye[i + c * n];
        eye[i + c * n] = eye[pi + c * n];
        eye[pi + c * n] = tmp;
      }
    }
  }

  // Solve L * Y = P*I
  dtrsmLLNUnit(n, n, 1.0, a, 0, n, eye, 0, n);
  // Solve U * X = Y
  dtrsmLUNN(n, n, 1.0, a, 0, n, eye, 0, n);

  return eye;
}

function lu(
  data: Float64Array,
  m: number,
  n: number
): { LU: Float64Array; ipiv: Int32Array } {
  const a = new Float64Array(data);
  const k = Math.min(m, n);
  const ipiv = new Int32Array(k);
  dgetrf(m, n, a, 0, m, ipiv);
  return { LU: a, ipiv };
}

function linsolve(
  A: Float64Array,
  m: number,
  n: number,
  B: Float64Array,
  nrhs: number
): Float64Array {
  if (m === n) {
    // Square: blocked LU + triangular solves
    const a = new Float64Array(A);
    const b = new Float64Array(B);
    const ipiv = new Int32Array(n);
    const info = dgetrf(n, n, a, 0, n, ipiv);
    if (info > 0) throw new Error("linsolve: matrix is singular");

    // Apply row permutations to B
    for (let i = 0; i < n; i++) {
      const pi = ipiv[i] - 1;
      if (pi !== i) {
        for (let c = 0; c < nrhs; c++) {
          const tmp = b[i + c * n];
          b[i + c * n] = b[pi + c * n];
          b[pi + c * n] = tmp;
        }
      }
    }

    // L * Y = P*B (unit diagonal), then U * X = Y
    dtrsmLLNUnit(n, nrhs, 1.0, a, 0, n, b, 0, n);
    dtrsmLUNN(n, nrhs, 1.0, a, 0, n, b, 0, n);
    return b;
  }
  // Non-square: fall back to ts-lapack (QR-based)
  return tsBridge.linsolve!(A, m, n, B, nrhs);
}

function chol(
  data: Float64Array,
  n: number,
  upper: boolean
): { R: Float64Array; info: number } {
  if (upper) {
    // For upper: A = R'*R. We compute lower L then transpose.
    // First transpose input to get lower triangle
    const a = new Float64Array(n * n);
    for (let j = 0; j < n; j++)
      for (let i = j; i < n; i++) a[i + j * n] = data[j + i * n];

    const info = dpotrf(n, a, 0, n);

    // Transpose L to get upper R, zero opposite triangle
    const R = new Float64Array(n * n);
    for (let j = 0; j < n; j++)
      for (let i = 0; i <= j; i++) R[i + j * n] = a[j + i * n]; // R(i,j) = L(j,i)

    return { R, info };
  } else {
    const a = new Float64Array(data);
    const info = dpotrf(n, a, 0, n);
    // Zero upper triangle
    for (let j = 0; j < n; j++) for (let i = 0; i < j; i++) a[i + j * n] = 0;
    return { R: a, info };
  }
}

const _bridge: LapackBridge = {
  matmul,
  inv,
  lu,
  linsolve,
  chol,
  // Fall back to ts-lapack for operations not yet FLAME-optimized
  qr: tsBridge.qr!.bind(tsBridge),
  eig: tsBridge.eig!.bind(tsBridge),
  svd: tsBridge.svd!.bind(tsBridge),
  linsolveComplex: tsBridge.linsolveComplex!.bind(tsBridge),
  cholComplex: tsBridge.cholComplex!.bind(tsBridge),
};

export function getFlameBridge(): LapackBridge {
  return _bridge;
}
