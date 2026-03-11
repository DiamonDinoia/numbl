// Translated from LAPACK/SRC/dlas2.f
// DLAS2 computes the singular values of a 2-by-2 triangular matrix
//    [  F   G  ]
//    [  0   H  ].
// On return, ssmin is the smaller singular value and ssmax is the
// larger singular value.

export function dlas2(
  f: number,
  g: number,
  h: number
): { ssmin: number; ssmax: number } {
  let ssmin: number;
  let ssmax: number;

  const fa = Math.abs(f);
  const ga = Math.abs(g);
  const ha = Math.abs(h);
  const fhmn = Math.min(fa, ha);
  const fhmx = Math.max(fa, ha);

  if (fhmn === 0.0) {
    ssmin = 0.0;
    if (fhmx === 0.0) {
      ssmax = ga;
    } else {
      ssmax =
        Math.max(fhmx, ga) *
        Math.sqrt(1.0 + (Math.min(fhmx, ga) / Math.max(fhmx, ga)) ** 2);
    }
  } else {
    if (ga < fhmx) {
      const as_ = 1.0 + fhmn / fhmx;
      const at = (fhmx - fhmn) / fhmx;
      const au = (ga / fhmx) ** 2;
      const c = 2.0 / (Math.sqrt(as_ * as_ + au) + Math.sqrt(at * at + au));
      ssmin = fhmn * c;
      ssmax = fhmx / c;
    } else {
      const au = fhmx / ga;
      if (au === 0.0) {
        // Avoid possible harmful underflow if exponent range
        // asymmetric (true ssmin may not underflow even if au underflows)
        ssmin = (fhmn * fhmx) / ga;
        ssmax = ga;
      } else {
        const as_ = 1.0 + fhmn / fhmx;
        const at = (fhmx - fhmn) / fhmx;
        const c =
          1.0 /
          (Math.sqrt(1.0 + (as_ * au) ** 2) + Math.sqrt(1.0 + (at * au) ** 2));
        ssmin = fhmn * c * au;
        ssmin = ssmin + ssmin;
        ssmax = ga / (c + c);
      }
    }
  }

  return { ssmin, ssmax };
}
