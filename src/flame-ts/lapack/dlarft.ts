// Form the triangular factor T of a block reflector H = I - V * T * V'
// V is m×k lower trapezoidal (unit diagonal), tau is length k.
// T is k×k upper triangular, stored in t at tOff with leading dimension ldt.
//
// Forward direction: H = H(1) * H(2) * ... * H(k)
// T(j,j) = tau(j)
// T(1:j-1, j) = -tau(j) * T(1:j-1, 1:j-1) * V(:, 1:j-1)' * v(:, j)

export function dlarft(
  m: number,
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
  for (let j = 0; j < k; j++) {
    if (tau[tauOff + j] === 0) {
      for (let i = 0; i <= j; i++) t[tOff + i + j * ldt] = 0;
      continue;
    }
    t[tOff + j + j * ldt] = tau[tauOff + j];
    if (j === 0) continue;

    // w = V(:, 0:j-1)' * v(:, j)
    // V has implicit 1 on diagonal: v(i,i) = 1, v(i<j, j) = 0 for lower trap
    for (let i = 0; i < j; i++) {
      let dot = 0;
      // v(i,i) = 1 so dot starts with v(i,j)
      // But V is lower trapezoidal: V(row, col) is stored, row >= col
      // V(:,i)' * V(:,j): sum over rows from j (since V(row<col,col)=0 except diagonal=1)
      dot = v[vOff + j + i * ldv]; // V(j,i) * V(j,j)=1
      for (let row = j + 1; row < m; row++) {
        dot += v[vOff + row + i * ldv] * v[vOff + row + j * ldv];
      }
      t[tOff + i + j * ldt] = dot;
    }

    // t(0:j-1, j) = -tau(j) * T(0:j-1, 0:j-1) * t(0:j-1, j)
    const tj = tau[tauOff + j];
    // First scale by -tau(j)
    for (let i = 0; i < j; i++) t[tOff + i + j * ldt] *= -tj;

    // Then multiply by T(0:j-1, 0:j-1) (upper triangular, in-place)
    // t(:,j) = T * t(:,j) where T is upper triangular j×j
    for (let i = j - 1; i >= 0; i--) {
      let s = t[tOff + i + i * ldt] * t[tOff + i + j * ldt];
      for (let p = i + 1; p < j; p++) {
        s += t[tOff + i + p * ldt] * t[tOff + p + j * ldt];
      }
      t[tOff + i + j * ldt] = s;
    }
  }
}
