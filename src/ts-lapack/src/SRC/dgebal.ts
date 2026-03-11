// Translated from SRC/dgebal.f
// DGEBAL balances a general real matrix A. This involves, first,
// permuting A by a similarity transformation to isolate eigenvalues
// in the first 1 to ILO-1 and last IHI+1 to N elements on the
// diagonal; and second, applying a diagonal similarity transformation
// to rows and columns ILO to IHI to make the rows and columns as
// close in norm as possible. Both steps are optional.
//
// JOB parameter (integer encoding):
//   0 = 'N' (none), 1 = 'P' (permute only),
//   2 = 'S' (scale only), 3 = 'B' (both)
//
// Indexing convention (matching Fortran column-major):
//   A(I,J)    => a[aOff + (I-1) + (J-1)*lda]     (1-based I,J)
//   SCALE(I)  => scale[scaleOff + (I-1)]          (1-based I)

import { dlamch } from "./dlamch.js";
import { MACH_SFMIN, MACH_PREC } from "../utils/constants.js";
import { dscal } from "../BLAS/dscal.js";
import { dswap } from "../BLAS/dswap.js";
import { idamax } from "../BLAS/idamax.js";
import { dnrm2 } from "../BLAS/dnrm2.js";

const JOB_NONE = 0;
const JOB_PERMUTE = 1;
const JOB_SCALE = 2;
// const JOB_BOTH = 3;

const SCLFAC = 2.0;
const FACTOR = 0.95;

export function dgebal(
  job: number,
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  scale: Float64Array,
  scaleOff: number
): { ilo: number; ihi: number; info: number } {
  let info = 0;
  let ilo = 1;
  let ihi = n;

  // Test the input parameters
  if (job < 0 || job > 3) {
    info = -1;
    return { ilo, ihi, info };
  } else if (n < 0) {
    info = -2;
    return { ilo, ihi, info };
  } else if (lda < Math.max(1, n)) {
    info = -4;
    return { ilo, ihi, info };
  }

  // Quick return if possible
  if (n === 0) {
    ilo = 1;
    ihi = 0;
    return { ilo, ihi, info };
  }

  if (job === JOB_NONE) {
    for (let i = 1; i <= n; i++) {
      scale[scaleOff + (i - 1)] = 1.0;
    }
    ilo = 1;
    ihi = n;
    return { ilo, ihi, info };
  }

  // Permutation to isolate eigenvalues if possible
  let k = 1; // will become ILO
  let l = n; // will become IHI

  if (job !== JOB_SCALE) {
    // Search for rows isolating an eigenvalue and push them down
    let noconv = true;
    while (noconv) {
      noconv = false;
      for (let i = l; i >= 1; i--) {
        let canswap = true;
        for (let j = 1; j <= l; j++) {
          if (i !== j && a[aOff + (i - 1) + (j - 1) * lda] !== 0.0) {
            canswap = false;
            break;
          }
        }

        if (canswap) {
          scale[scaleOff + (l - 1)] = i;
          if (i !== l) {
            // DSWAP(L, A(1,I), 1, A(1,L), 1) — swap columns I and L, rows 1..L
            dswap(l, a, aOff + (i - 1) * lda, 1, a, aOff + (l - 1) * lda, 1);
            // DSWAP(N-K+1, A(I,K), LDA, A(L,K), LDA) — swap rows I and L, cols K..N
            dswap(
              n - k + 1,
              a,
              aOff + (i - 1) + (k - 1) * lda,
              lda,
              a,
              aOff + (l - 1) + (k - 1) * lda,
              lda
            );
          }
          noconv = true;

          if (l === 1) {
            ilo = 1;
            ihi = 1;
            return { ilo, ihi, info };
          }

          l = l - 1;
        }
      }
    }

    // Search for columns isolating an eigenvalue and push them left
    noconv = true;
    while (noconv) {
      noconv = false;
      for (let j = k; j <= l; j++) {
        let canswap = true;
        for (let i = k; i <= l; i++) {
          if (i !== j && a[aOff + (i - 1) + (j - 1) * lda] !== 0.0) {
            canswap = false;
            break;
          }
        }

        if (canswap) {
          scale[scaleOff + (k - 1)] = j;
          if (j !== k) {
            // DSWAP(L, A(1,J), 1, A(1,K), 1) — swap columns J and K, rows 1..L
            dswap(l, a, aOff + (j - 1) * lda, 1, a, aOff + (k - 1) * lda, 1);
            // DSWAP(N-K+1, A(J,K), LDA, A(K,K), LDA) — swap rows J and K, cols K..N
            dswap(
              n - k + 1,
              a,
              aOff + (j - 1) + (k - 1) * lda,
              lda,
              a,
              aOff + (k - 1) + (k - 1) * lda,
              lda
            );
          }
          noconv = true;
          k = k + 1;
        }
      }
    }
  }

  // Initialize SCALE for non-permuted submatrix
  for (let i = k; i <= l; i++) {
    scale[scaleOff + (i - 1)] = 1.0;
  }

  // If we only had to permute, we are done
  if (job === JOB_PERMUTE) {
    ilo = k;
    ihi = l;
    return { ilo, ihi, info };
  }

  // Balance the submatrix in rows K to L
  // Iterative loop for norm reduction

  const sfmin1 = dlamch(MACH_SFMIN) / dlamch(MACH_PREC);
  const sfmax1 = 1.0 / sfmin1;
  const sfmin2 = sfmin1 * SCLFAC;
  const sfmax2 = 1.0 / sfmin2;

  let noconv = true;
  while (noconv) {
    noconv = false;

    for (let i = k; i <= l; i++) {
      // C = column norm of A(K:L, I)
      let c = dnrm2(l - k + 1, a, aOff + (k - 1) + (i - 1) * lda, 1);
      // R = row norm of A(I, K:L)
      let r = dnrm2(l - k + 1, a, aOff + (i - 1) + (k - 1) * lda, lda);

      // ICA = IDAMAX(L, A(1,I), 1) — index of max abs in column I, rows 1..L
      const ica = idamax(l, a, aOff + (i - 1) * lda, 1);
      let ca = Math.abs(a[aOff + (ica - 1) + (i - 1) * lda]);

      // IRA = IDAMAX(N-K+1, A(I,K), LDA) — index of max abs in row I, cols K..N
      const ira = idamax(n - k + 1, a, aOff + (i - 1) + (k - 1) * lda, lda);
      let ra = Math.abs(a[aOff + (i - 1) + (ira + k - 2) * lda]);

      // Guard against zero C or R due to underflow
      if (c === 0.0 || r === 0.0) continue;

      // Exit if NaN to avoid infinite loop
      if (isNaN(c + ca + r + ra)) {
        info = -3;
        return { ilo, ihi, info };
      }

      let g = r / SCLFAC;
      let f = 1.0;
      const s = c + r;

      while (
        c < g &&
        Math.max(f, c, ca) < sfmax2 &&
        Math.min(r, g, ra) > sfmin2
      ) {
        f = f * SCLFAC;
        c = c * SCLFAC;
        ca = ca * SCLFAC;
        r = r / SCLFAC;
        g = g / SCLFAC;
        ra = ra / SCLFAC;
      }

      g = c / SCLFAC;

      while (
        g >= r &&
        Math.max(r, ra) < sfmax2 &&
        Math.min(f, c, g, ca) > sfmin2
      ) {
        f = f / SCLFAC;
        c = c / SCLFAC;
        g = g / SCLFAC;
        ca = ca / SCLFAC;
        r = r * SCLFAC;
        ra = ra * SCLFAC;
      }

      // Now balance
      if (c + r >= FACTOR * s) continue;
      if (f < 1.0 && scale[scaleOff + (i - 1)] < 1.0) {
        if (f * scale[scaleOff + (i - 1)] <= sfmin1) continue;
      }
      if (f > 1.0 && scale[scaleOff + (i - 1)] > 1.0) {
        if (scale[scaleOff + (i - 1)] >= sfmax1 / f) continue;
      }

      g = 1.0 / f;
      scale[scaleOff + (i - 1)] = scale[scaleOff + (i - 1)] * f;
      noconv = true;

      // DSCAL(N-K+1, G, A(I,K), LDA) — scale row I, cols K..N
      dscal(n - k + 1, g, a, aOff + (i - 1) + (k - 1) * lda, lda);
      // DSCAL(L, F, A(1,I), 1) — scale column I, rows 1..L
      dscal(l, f, a, aOff + (i - 1) * lda, 1);
    }
  }

  ilo = k;
  ihi = l;

  return { ilo, ihi, info };
}
