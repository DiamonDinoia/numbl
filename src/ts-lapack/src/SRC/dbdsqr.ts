// Translated from LAPACK/SRC/dbdsqr.f
//
// DBDSQR computes the singular values and, optionally, the right and/or
// left singular vectors from the singular value decomposition (SVD) of
// a real N-by-N (upper or lower) bidiagonal matrix B using the implicit
// zero-shift QR algorithm.
//
// NOTE: The DLASQ1 fast path (when NCVT=NRU=NCC=0) is intentionally
// skipped. We always fall through to the QR iteration path.

import { UPPER, LOWER } from "../utils/constants.js";
import { dlamch } from "./dlamch.js";
import { dlartg } from "./dlartg.js";
import { dlas2 } from "./dlas2.js";
import { dlasv2 } from "./dlasv2.js";
import { dlasr } from "./dlasr.js";
import { drot } from "../BLAS/drot.js";
import { dscal } from "../BLAS/dscal.js";
import { dswap } from "../BLAS/dswap.js";
import {
  PIVOT_V,
  // PIVOT_B and PIVOT_T imported for potential future use
  // PIVOT_B,
  // PIVOT_T,
  DIRECT_F,
  DIRECT_B,
  LEFT,
  RIGHT,
  MACH_EPS,
  MACH_SFMIN,
} from "../utils/constants.js";
import { xerbla } from "../utils/xerbla.js";

const ZERO = 0.0;
const ONE = 1.0;
const NEGONE = -1.0;
const HNDRTH = 0.01;
const TEN = 10.0;
const HNDRD = 100.0;
const MEIGTH = -0.125;
const MAXITR = 6;

// Fortran SIGN(a,b): returns |a| * sign(b)
function fsign(a: number, b: number): number {
  return b >= 0 ? Math.abs(a) : -Math.abs(a);
}

export function dbdsqr(
  uplo: number, // UPPER=0 or LOWER=1
  n: number,
  ncvt: number, // number of columns of VT
  nru: number, // number of rows of U
  ncc: number, // number of columns of C
  d: Float64Array,
  dOff: number, // diagonal (length n)
  e: Float64Array,
  eOff: number, // off-diagonal (length n-1)
  vt: Float64Array,
  vtOff: number,
  ldvt: number, // right singular vectors
  u: Float64Array,
  uOff: number,
  ldu: number, // left singular vectors
  c: Float64Array,
  cOff: number,
  ldc: number, // additional matrix
  work: Float64Array,
  workOff: number // workspace 4*n
): number {
  // returns info
  let info = 0;

  // Test the input parameters
  const lower = uplo === LOWER;
  if (uplo !== UPPER && !lower) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ncvt < 0) {
    info = -3;
  } else if (nru < 0) {
    info = -4;
  } else if (ncc < 0) {
    info = -5;
  } else if ((ncvt === 0 && ldvt < 1) || (ncvt > 0 && ldvt < Math.max(1, n))) {
    info = -9;
  } else if (ldu < Math.max(1, nru)) {
    info = -11;
  } else if ((ncc === 0 && ldc < 1) || (ncc > 0 && ldc < Math.max(1, n))) {
    info = -13;
  }
  if (info !== 0) {
    xerbla("DBDSQR", -info);
    return info;
  }
  if (n === 0) return 0;

  // Helper: 0-based access to Fortran 1-based arrays
  // D(i) in Fortran => d[dOff + i - 1]
  // E(i) in Fortran => e[eOff + i - 1]
  // WORK(i) in Fortran => work[workOff + i - 1]
  // VT(i,j) in Fortran => vt[vtOff + (i-1) + (j-1)*ldvt]
  // U(i,j) in Fortran => u[uOff + (i-1) + (j-1)*ldu]
  // C(i,j) in Fortran => c[cOff + (i-1) + (j-1)*ldc]

  const D = (i: number) => d[dOff + i - 1];
  const setD = (i: number, v: number) => {
    d[dOff + i - 1] = v;
  };
  const E = (i: number) => e[eOff + i - 1];
  const setE = (i: number, v: number) => {
    e[eOff + i - 1] = v;
  };
  // const _W = (i: number) => work[workOff + i - 1];
  const setW = (i: number, v: number) => {
    work[workOff + i - 1] = v;
  };

  // VT(i,1) offset for drot/dlasr: pointer to row i of VT
  // VT is column-major: VT(i,j) = vt[vtOff + (i-1) + (j-1)*ldvt]
  // VT(i,1) = vtOff + (i-1), stride = ldvt
  const vtRow = (i: number) => vtOff + (i - 1);
  // U(1,j) offset: pointer to column j of U
  // U(i,j) = u[uOff + (i-1) + (j-1)*ldu]
  // U(1,j) = uOff + (j-1)*ldu, stride = 1
  const uCol = (j: number) => uOff + (j - 1) * ldu;
  // C(i,1) offset: pointer to row i of C
  const cRow = (i: number) => cOff + (i - 1);

  if (n === 1) {
    // Jump to label 160 — make singular values positive
    // (handled at end, but for n=1 we go directly there)
    if (D(1) === ZERO) {
      setD(1, ZERO); // avoid -0
    }
    if (D(1) < ZERO) {
      setD(1, -D(1));
      if (ncvt > 0) {
        dscal(ncvt, NEGONE, vt, vtRow(1), ldvt);
      }
    }
    return 0;
  }

  // ROTATE is true if any singular vectors desired
  // const _rotate = ncvt > 0 || nru > 0 || ncc > 0;

  // SKIP the DLASQ1 fast path — always fall through to QR iteration
  // (Original Fortran calls DLASQ1 when !rotate, we skip that.)

  const nm1 = n - 1;
  const nm12 = nm1 + nm1;
  const nm13 = nm12 + nm1;
  let idir = 0;

  // Get machine constants
  const eps = dlamch(MACH_EPS);
  const unfl = dlamch(MACH_SFMIN);

  // If matrix lower bidiagonal, rotate to be upper bidiagonal
  if (lower) {
    for (let i = 1; i <= n - 1; i++) {
      const res = dlartg(D(i), E(i));
      setD(i, res.r);
      setE(i, res.sn * D(i + 1));
      setD(i + 1, res.cs * D(i + 1));
      setW(i, res.cs);
      setW(nm1 + i, res.sn);
    }

    // Update singular vectors if desired
    if (nru > 0) {
      dlasr(
        RIGHT,
        PIVOT_V,
        DIRECT_F,
        nru,
        n,
        work,
        workOff,
        work,
        workOff + nm1,
        u,
        uOff,
        ldu
      );
    }
    if (ncc > 0) {
      dlasr(
        LEFT,
        PIVOT_V,
        DIRECT_F,
        n,
        ncc,
        work,
        workOff,
        work,
        workOff + nm1,
        c,
        cOff,
        ldc
      );
    }
  }

  // Compute singular values to relative accuracy TOL
  const tolmul = Math.max(TEN, Math.min(HNDRD, Math.pow(eps, MEIGTH)));
  const tol = tolmul * eps;

  // Compute approximate maximum, minimum singular values
  let smax = ZERO;
  for (let i = 1; i <= n; i++) {
    smax = Math.max(smax, Math.abs(D(i)));
  }
  for (let i = 1; i <= n - 1; i++) {
    smax = Math.max(smax, Math.abs(E(i)));
  }

  let smin = ZERO;
  let sminoa: number;
  let thresh: number;
  let mu: number;

  if (tol >= ZERO) {
    // Relative accuracy desired
    sminoa = Math.abs(D(1));
    if (sminoa !== ZERO) {
      mu = sminoa;
      for (let i = 2; i <= n; i++) {
        mu = Math.abs(D(i)) * (mu / (mu + Math.abs(E(i - 1))));
        sminoa = Math.min(sminoa, mu);
        if (sminoa === ZERO) break;
      }
    }
    sminoa = sminoa / Math.sqrt(n);
    thresh = Math.max(tol * sminoa, MAXITR * (n * (n * unfl)));
  } else {
    // Absolute accuracy desired
    thresh = Math.max(Math.abs(tol) * smax, MAXITR * (n * (n * unfl)));
  }

  // Prepare for main iteration loop
  const maxitdivn = MAXITR * n;
  let iterdivn = 0;
  let iter = -1;
  let oldll = -1;
  let oldm = -1;

  // M points to last element of unconverged part of matrix
  let m = n;

  // Main iteration loop (label 60)
  mainLoop: while (true) {
    // Check for convergence or exceeding iteration count
    if (m <= 1) break mainLoop; // goto 160

    if (iter >= n) {
      iter = iter - n;
      iterdivn = iterdivn + 1;
      if (iterdivn >= maxitdivn) {
        // goto 200 — failure to converge
        info = 0;
        for (let i = 1; i <= n - 1; i++) {
          if (E(i) !== ZERO) info++;
        }
        return info;
      }
    }

    // Find diagonal block of matrix to work on (label 60 scan)
    if (tol < ZERO && Math.abs(D(m)) <= thresh) {
      setD(m, ZERO);
    }
    smax = Math.abs(D(m));

    let ll = 0;
    let foundSplit = false;

    for (let lll = 1; lll <= m - 1; lll++) {
      ll = m - lll;
      const abss = Math.abs(D(ll));
      const abse = Math.abs(E(ll));
      if (tol < ZERO && abss <= thresh) {
        setD(ll, ZERO);
      }
      if (abse <= thresh) {
        // goto 80
        foundSplit = true;
        break;
      }
      smax = Math.max(smax, abss, abse);
    }

    if (!foundSplit) {
      ll = 0;
      // goto 90 (fall through)
    } else {
      // label 80
      setE(ll, ZERO);

      // Matrix splits since E(LL) = 0
      if (ll === m - 1) {
        // Convergence of bottom singular value, return to top of loop
        m = m - 1;
        continue mainLoop; // goto 60
      }
    }

    // label 90
    ll = ll + 1;

    // E(LL) through E(M-1) are nonzero, E(LL-1) is zero

    if (ll === m - 1) {
      // 2 by 2 block, handle separately
      const res = dlasv2(D(m - 1), E(m - 1), D(m));
      setD(m - 1, res.ssmax);
      setE(m - 1, ZERO);
      setD(m, res.ssmin);

      // Compute singular vectors, if desired
      if (ncvt > 0) {
        drot(
          ncvt,
          vt,
          vtRow(m - 1),
          ldvt,
          vt,
          vtRow(m),
          ldvt,
          res.csr,
          res.snr
        );
      }
      if (nru > 0) {
        drot(nru, u, uCol(m - 1), 1, u, uCol(m), 1, res.csl, res.snl);
      }
      if (ncc > 0) {
        drot(ncc, c, cRow(m - 1), ldc, c, cRow(m), ldc, res.csl, res.snl);
      }
      m = m - 2;
      continue mainLoop; // goto 60
    }

    // If working on new submatrix, choose shift direction
    if (ll > oldm || m < oldll) {
      if (Math.abs(D(ll)) >= Math.abs(D(m))) {
        idir = 1; // Chase bulge from top to bottom
      } else {
        idir = 2; // Chase bulge from bottom to top
      }
    }

    // Apply convergence tests
    if (idir === 1) {
      // Run convergence test in forward direction
      // First apply standard test to bottom of matrix
      if (
        Math.abs(E(m - 1)) <= Math.abs(tol) * Math.abs(D(m)) ||
        (tol < ZERO && Math.abs(E(m - 1)) <= thresh)
      ) {
        setE(m - 1, ZERO);
        continue mainLoop; // goto 60
      }

      if (tol >= ZERO) {
        // Relative accuracy — apply convergence criterion forward
        mu = Math.abs(D(ll));
        smin = mu;
        let converged = false;
        for (let lll = ll; lll <= m - 1; lll++) {
          if (Math.abs(E(lll)) <= tol * mu) {
            setE(lll, ZERO);
            converged = true;
            break; // goto 60
          }
          mu = Math.abs(D(lll + 1)) * (mu / (mu + Math.abs(E(lll))));
          smin = Math.min(smin, mu);
        }
        if (converged) continue mainLoop;
      }
    } else {
      // Run convergence test in backward direction
      // First apply standard test to top of matrix
      if (
        Math.abs(E(ll)) <= Math.abs(tol) * Math.abs(D(ll)) ||
        (tol < ZERO && Math.abs(E(ll)) <= thresh)
      ) {
        setE(ll, ZERO);
        continue mainLoop; // goto 60
      }

      if (tol >= ZERO) {
        // Relative accuracy — apply convergence criterion backward
        mu = Math.abs(D(m));
        smin = mu;
        let converged = false;
        for (let lll = m - 1; lll >= ll; lll--) {
          if (Math.abs(E(lll)) <= tol * mu) {
            setE(lll, ZERO);
            converged = true;
            break; // goto 60
          }
          mu = Math.abs(D(lll)) * (mu / (mu + Math.abs(E(lll))));
          smin = Math.min(smin, mu);
        }
        if (converged) continue mainLoop;
      }
    }

    oldll = ll;
    oldm = m;

    // Compute shift
    let shift: number;
    let sll: number;

    if (tol >= ZERO && n * tol * (smin / smax) <= Math.max(eps, HNDRTH * tol)) {
      // Use a zero shift to avoid loss of relative accuracy
      shift = ZERO;
    } else {
      // Compute the shift from 2-by-2 block at end of matrix
      if (idir === 1) {
        sll = Math.abs(D(ll));
        const res = dlas2(D(m - 1), E(m - 1), D(m));
        shift = res.ssmin;
      } else {
        sll = Math.abs(D(m));
        const res = dlas2(D(ll), E(ll), D(ll + 1));
        shift = res.ssmin;
      }

      // Test if shift negligible
      if (sll > ZERO) {
        if ((shift / sll) * (shift / sll) < eps) {
          shift = ZERO;
        }
      }
    }

    // Increment iteration count
    iter = iter + m - ll;

    // QR step
    if (shift === ZERO) {
      // Simplified QR iteration (zero shift)
      if (idir === 1) {
        // Chase bulge from top to bottom
        let cs = ONE;
        let oldcs = ONE;
        let oldsn = ZERO;
        let sn: number;
        let r: number;

        for (let i = ll; i <= m - 1; i++) {
          const res1 = dlartg(D(i) * cs, E(i));
          cs = res1.cs;
          sn = res1.sn;
          r = res1.r;
          if (i > ll) {
            setE(i - 1, oldsn * r);
          }
          const res2 = dlartg(oldcs * r, D(i + 1) * sn);
          oldcs = res2.cs;
          oldsn = res2.sn;
          setD(i, res2.r);
          setW(i - ll + 1, cs);
          setW(i - ll + 1 + nm1, sn);
          setW(i - ll + 1 + nm12, oldcs);
          setW(i - ll + 1 + nm13, oldsn);
        }
        const h = D(m) * cs;
        setD(m, h * oldcs);
        setE(m - 1, h * oldsn);

        // Update singular vectors
        if (ncvt > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_F,
            m - ll + 1,
            ncvt,
            work,
            workOff,
            work,
            workOff + nm1,
            vt,
            vtRow(ll),
            ldvt
          );
        }
        if (nru > 0) {
          dlasr(
            RIGHT,
            PIVOT_V,
            DIRECT_F,
            nru,
            m - ll + 1,
            work,
            workOff + nm12,
            work,
            workOff + nm13,
            u,
            uCol(ll),
            ldu
          );
        }
        if (ncc > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_F,
            m - ll + 1,
            ncc,
            work,
            workOff + nm12,
            work,
            workOff + nm13,
            c,
            cRow(ll),
            ldc
          );
        }

        // Test convergence
        if (Math.abs(E(m - 1)) <= thresh) {
          setE(m - 1, ZERO);
        }
      } else {
        // Chase bulge from bottom to top
        let cs = ONE;
        let oldcs = ONE;
        let oldsn = ZERO;
        let sn: number;
        let r: number;

        for (let i = m; i >= ll + 1; i--) {
          const res1 = dlartg(D(i) * cs, E(i - 1));
          cs = res1.cs;
          sn = res1.sn;
          r = res1.r;
          if (i < m) {
            setE(i, oldsn * r);
          }
          const res2 = dlartg(oldcs * r, D(i - 1) * sn);
          oldcs = res2.cs;
          oldsn = res2.sn;
          setD(i, res2.r);
          setW(i - ll, cs);
          setW(i - ll + nm1, -sn);
          setW(i - ll + nm12, oldcs);
          setW(i - ll + nm13, -oldsn);
        }
        const h = D(ll) * cs;
        setD(ll, h * oldcs);
        setE(ll, h * oldsn);

        // Update singular vectors
        if (ncvt > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_B,
            m - ll + 1,
            ncvt,
            work,
            workOff + nm12,
            work,
            workOff + nm13,
            vt,
            vtRow(ll),
            ldvt
          );
        }
        if (nru > 0) {
          dlasr(
            RIGHT,
            PIVOT_V,
            DIRECT_B,
            nru,
            m - ll + 1,
            work,
            workOff,
            work,
            workOff + nm1,
            u,
            uCol(ll),
            ldu
          );
        }
        if (ncc > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_B,
            m - ll + 1,
            ncc,
            work,
            workOff,
            work,
            workOff + nm1,
            c,
            cRow(ll),
            ldc
          );
        }

        // Test convergence
        if (Math.abs(E(ll)) <= thresh) {
          setE(ll, ZERO);
        }
      }
    } else {
      // Use nonzero shift
      if (idir === 1) {
        // Chase bulge from top to bottom
        let f = (Math.abs(D(ll)) - shift) * (fsign(ONE, D(ll)) + shift / D(ll));
        let g = E(ll);

        let cosr: number, sinr: number, cosl: number, sinl: number, r: number;

        for (let i = ll; i <= m - 1; i++) {
          const res1 = dlartg(f, g);
          cosr = res1.cs;
          sinr = res1.sn;
          r = res1.r;
          if (i > ll) {
            setE(i - 1, r);
          }
          f = cosr * D(i) + sinr * E(i);
          setE(i, cosr * E(i) - sinr * D(i));
          g = sinr * D(i + 1);
          setD(i + 1, cosr * D(i + 1));

          const res2 = dlartg(f, g);
          cosl = res2.cs;
          sinl = res2.sn;
          r = res2.r;
          setD(i, r);
          f = cosl * E(i) + sinl * D(i + 1);
          setD(i + 1, cosl * D(i + 1) - sinl * E(i));
          if (i < m - 1) {
            g = sinl * E(i + 1);
            setE(i + 1, cosl * E(i + 1));
          }
          setW(i - ll + 1, cosr);
          setW(i - ll + 1 + nm1, sinr);
          setW(i - ll + 1 + nm12, cosl);
          setW(i - ll + 1 + nm13, sinl);
        }
        setE(m - 1, f);

        // Update singular vectors
        if (ncvt > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_F,
            m - ll + 1,
            ncvt,
            work,
            workOff,
            work,
            workOff + nm1,
            vt,
            vtRow(ll),
            ldvt
          );
        }
        if (nru > 0) {
          dlasr(
            RIGHT,
            PIVOT_V,
            DIRECT_F,
            nru,
            m - ll + 1,
            work,
            workOff + nm12,
            work,
            workOff + nm13,
            u,
            uCol(ll),
            ldu
          );
        }
        if (ncc > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_F,
            m - ll + 1,
            ncc,
            work,
            workOff + nm12,
            work,
            workOff + nm13,
            c,
            cRow(ll),
            ldc
          );
        }

        // Test convergence
        if (Math.abs(E(m - 1)) <= thresh) {
          setE(m - 1, ZERO);
        }
      } else {
        // Chase bulge from bottom to top
        let f = (Math.abs(D(m)) - shift) * (fsign(ONE, D(m)) + shift / D(m));
        let g = E(m - 1);

        let cosr: number, sinr: number, cosl: number, sinl: number, r: number;

        for (let i = m; i >= ll + 1; i--) {
          const res1 = dlartg(f, g);
          cosr = res1.cs;
          sinr = res1.sn;
          r = res1.r;
          if (i < m) {
            setE(i, r);
          }
          f = cosr * D(i) + sinr * E(i - 1);
          setE(i - 1, cosr * E(i - 1) - sinr * D(i));
          g = sinr * D(i - 1);
          setD(i - 1, cosr * D(i - 1));

          const res2 = dlartg(f, g);
          cosl = res2.cs;
          sinl = res2.sn;
          r = res2.r;
          setD(i, r);
          f = cosl * E(i - 1) + sinl * D(i - 1);
          setD(i - 1, cosl * D(i - 1) - sinl * E(i - 1));
          if (i > ll + 1) {
            g = sinl * E(i - 2);
            setE(i - 2, cosl * E(i - 2));
          }
          setW(i - ll, cosr);
          setW(i - ll + nm1, -sinr);
          setW(i - ll + nm12, cosl);
          setW(i - ll + nm13, -sinl);
        }
        setE(ll, f);

        // Test convergence
        if (Math.abs(E(ll)) <= thresh) {
          setE(ll, ZERO);
        }

        // Update singular vectors if desired
        if (ncvt > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_B,
            m - ll + 1,
            ncvt,
            work,
            workOff + nm12,
            work,
            workOff + nm13,
            vt,
            vtRow(ll),
            ldvt
          );
        }
        if (nru > 0) {
          dlasr(
            RIGHT,
            PIVOT_V,
            DIRECT_B,
            nru,
            m - ll + 1,
            work,
            workOff,
            work,
            workOff + nm1,
            u,
            uCol(ll),
            ldu
          );
        }
        if (ncc > 0) {
          dlasr(
            LEFT,
            PIVOT_V,
            DIRECT_B,
            m - ll + 1,
            ncc,
            work,
            workOff,
            work,
            workOff + nm1,
            c,
            cRow(ll),
            ldc
          );
        }
      }
    }

    // QR iteration finished, go back and check convergence (goto 60)
    continue mainLoop;
  }

  // label 160 — All singular values converged, make them positive
  for (let i = 1; i <= n; i++) {
    if (D(i) === ZERO) {
      setD(i, ZERO); // avoid -0
    }
    if (D(i) < ZERO) {
      setD(i, -D(i));
      // Change sign of singular vectors, if desired
      if (ncvt > 0) {
        dscal(ncvt, NEGONE, vt, vtRow(i), ldvt);
      }
    }
  }

  // Sort the singular values into decreasing order (insertion sort on
  // singular values, but only one transposition per singular vector)
  for (let i = 1; i <= n - 1; i++) {
    // Scan for smallest D(I)
    let isub = 1;
    smin = D(1);
    for (let j = 2; j <= n + 1 - i; j++) {
      if (D(j) <= smin) {
        isub = j;
        smin = D(j);
      }
    }
    if (isub !== n + 1 - i) {
      // Swap singular values and vectors
      setD(isub, D(n + 1 - i));
      setD(n + 1 - i, smin);
      if (ncvt > 0) {
        dswap(ncvt, vt, vtRow(isub), ldvt, vt, vtRow(n + 1 - i), ldvt);
      }
      if (nru > 0) {
        dswap(nru, u, uCol(isub), 1, u, uCol(n + 1 - i), 1);
      }
      if (ncc > 0) {
        dswap(ncc, c, cRow(isub), ldc, c, cRow(n + 1 - i), ldc);
      }
    }
  }

  // goto 220
  return 0;
}
