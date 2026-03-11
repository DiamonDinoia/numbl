// Translated from SRC/dlarfg.f
// DLARFG generates a real elementary reflector H of order n, such that
//
//       H * ( alpha ) = ( beta ),   H**T * H = I.
//           (   x   )   (   0  )
//
// where alpha and beta are scalars, and x is an (n-1)-element real vector.
// H is represented in the form
//
//       H = I - tau * ( 1 ) * ( 1 v**T ) ,
//                     ( v )
//
// where tau is a real scalar and v is a real (n-1)-element vector.
//
// If the elements of x are all zero, then tau = 0 and H is the unit matrix.
//
// Array indexing convention (matching Fortran):
//   X(I)  =>  x[xOff + (I-1)*incx]   (I is 1-based)
//
// Parameters:
//   n     - order of the reflector (>= 1)
//   alpha - scalar alpha (the pivot element)
//   x     - Float64Array; elements x(2:n) on entry; overwritten with v(2:n) on exit
//   xOff  - offset into x for X(1)
//   incx  - increment between elements of x (incx > 0)
//
// Returns { alpha: number (overwritten with beta), tau: number }

import { dlamch } from "./dlamch.js";
import { MACH_SFMIN, MACH_EPS } from "../utils/constants.js";
import { dlapy2 } from "./dlapy2.js";
import { dscal } from "../BLAS/dscal.js";
import { dnrm2 } from "../BLAS/dnrm2.js";

export function dlarfg(
  n: number,
  alpha: number,
  x: Float64Array,
  xOff: number,
  incx: number
): { alpha: number; tau: number } {
  if (n <= 1) {
    return { alpha, tau: 0.0 };
  }

  let xnorm = dnrm2(n - 1, x, xOff, incx);

  if (xnorm === 0.0) {
    // H = I
    return { alpha, tau: 0.0 };
  }

  // General case
  // Fortran: BETA = -SIGN(DLAPY2(ALPHA,XNORM), ALPHA)
  // Math.sign(0) returns 0 in JS but Fortran SIGN treats 0 as positive,
  // so we default to 1.0 when alpha is exactly 0.
  let beta = -(Math.sign(alpha) || 1.0) * dlapy2(alpha, xnorm);
  const safmin = dlamch(MACH_SFMIN) / dlamch(MACH_EPS);
  let knt = 0;

  if (Math.abs(beta) < safmin) {
    // XNORM, BETA may be inaccurate; scale X and recompute them
    const rsafmn = 1.0 / safmin;
    do {
      knt++;
      dscal(n - 1, rsafmn, x, xOff, incx);
      beta *= rsafmn;
      alpha *= rsafmn;
    } while (Math.abs(beta) < safmin && knt < 20);

    // New BETA is at most 1, at least SAFMIN
    xnorm = dnrm2(n - 1, x, xOff, incx);
    beta = -(Math.sign(alpha) || 1.0) * dlapy2(alpha, xnorm);
  }

  const tau = (beta - alpha) / beta;
  dscal(n - 1, 1.0 / (alpha - beta), x, xOff, incx);

  // If ALPHA is subnormal, it may lose relative accuracy
  for (let j = 1; j <= knt; j++) {
    beta *= safmin;
  }
  alpha = beta;

  return { alpha, tau };
}
