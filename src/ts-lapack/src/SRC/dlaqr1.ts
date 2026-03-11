// Translated from SRC/dlaqr1.f
// DLAQR1 sets v to a scalar multiple of the first column of the product
//
//   K = (H - (sr1 + i*si1)*I) * (H - (sr2 + i*si2)*I)
//
// Given a 2-by-2 or 3-by-3 matrix H. Scaling avoids overflows and most
// underflows. It is assumed that either sr1 = sr2 and si1 = -si2, or
// si1 = si2 = 0.
//
// Indexing convention (matching Fortran column-major):
//   H(I,J) => h[hOff + (I-1) + (J-1)*ldh]   (1-based I,J)
//   V(I)   => v[vOff + (I-1)]                (1-based I)

export function dlaqr1(
  n: number,
  h: Float64Array,
  hOff: number,
  ldh: number,
  sr1: number,
  si1: number,
  sr2: number,
  si2: number,
  v: Float64Array,
  vOff: number
): void {
  // Quick return if possible
  if (n !== 2 && n !== 3) {
    return;
  }

  if (n === 2) {
    const s = Math.abs(h[hOff] - sr2) + Math.abs(si2) + Math.abs(h[hOff + 1]);
    if (s === 0.0) {
      v[vOff] = 0.0;
      v[vOff + 1] = 0.0;
    } else {
      const h21s = h[hOff + 1] / s; // H(2,1)/S
      v[vOff] =
        h21s * h[hOff + ldh] +
        (h[hOff] - sr1) * ((h[hOff] - sr2) / s) -
        si1 * (si2 / s);
      v[vOff + 1] = h21s * (h[hOff] + h[hOff + 1 + ldh] - sr1 - sr2);
    }
  } else {
    // n === 3
    const s =
      Math.abs(h[hOff] - sr2) +
      Math.abs(si2) +
      Math.abs(h[hOff + 1]) +
      Math.abs(h[hOff + 2]);
    if (s === 0.0) {
      v[vOff] = 0.0;
      v[vOff + 1] = 0.0;
      v[vOff + 2] = 0.0;
    } else {
      const h21s = h[hOff + 1] / s; // H(2,1)/S
      const h31s = h[hOff + 2] / s; // H(3,1)/S
      v[vOff] =
        (h[hOff] - sr1) * ((h[hOff] - sr2) / s) -
        si1 * (si2 / s) +
        h[hOff + ldh] * h21s +
        h[hOff + 2 * ldh] * h31s;
      v[vOff + 1] =
        h21s * (h[hOff] + h[hOff + 1 + ldh] - sr1 - sr2) +
        h[hOff + 1 + 2 * ldh] * h31s;
      v[vOff + 2] =
        h31s * (h[hOff] + h[hOff + 2 + 2 * ldh] - sr1 - sr2) +
        h21s * h[hOff + 2 + ldh];
    }
  }
}
