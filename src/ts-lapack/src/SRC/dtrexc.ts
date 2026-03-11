// Translated from SRC/dtrexc.f
// DTREXC reorders the real Schur factorization of a real matrix
// A = Q*T*Q**T, so that the diagonal block of T with row index IFST is
// moved to row ILST.
//
// The real Schur form T is reordered by an orthogonal similarity
// transformation Z**T*T*Z, and optionally the matrix Q of Schur vectors
// is updated by postmultiplying it with Z.
//
// T must be in Schur canonical form (as returned by DHSEQR), that is,
// block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
// 2-by-2 diagonal block has its diagonal elements equal and its
// off-diagonal elements of opposite sign.
//
// Array indexing convention (column-major, matching Fortran):
//   T(I,J) => t[tOff + (I-1) + (J-1)*ldt]   (I,J are 1-based)
//   Q(I,J) => q[qOff + (I-1) + (J-1)*ldq]   (I,J are 1-based)

import { dlaexc } from "./dlaexc.js";

// COMPQ constants
// 0 = 'N' — do not update Q
// 1 = 'V' — update the matrix Q of Schur vectors

/**
 * DTREXC reorders the real Schur factorization of a real matrix
 * A = Q*T*Q**T, so that the diagonal block of T with row index IFST is
 * moved to row ILST.
 *
 * @param compq - 0 ('N'): do not update Q; 1 ('V'): update Q
 * @param n - order of the matrix T (>= 0)
 * @param t - upper quasi-triangular matrix in Schur form, dimension (ldt, n)
 * @param tOff - offset into t
 * @param ldt - leading dimension of t
 * @param q - Schur vectors, dimension (ldq, n), updated if compq=1
 * @param qOff - offset into q
 * @param ldq - leading dimension of q
 * @param ifst - index of the first row of the block to move (1-based, in/out)
 * @param ilst - target index for the block (1-based, in/out)
 * @param work - workspace, dimension (n)
 * @param workOff - offset into work
 * @returns INFO: 0 = success, < 0 = illegal argument, 1 = swap failed
 */
export function dtrexc(
  compq: number,
  n: number,
  t: Float64Array,
  tOff: number,
  ldt: number,
  q: Float64Array,
  qOff: number,
  ldq: number,
  ifst: { val: number },
  ilst: { val: number },
  work: Float64Array,
  workOff: number
): number {
  const ZERO = 0.0;

  // Helper for column-major indexing (1-based)
  const T_ = (i: number, j: number) => tOff + (i - 1) + (j - 1) * ldt;

  let info = 0;
  const wantq = compq === 1;

  // Decode and test the input arguments.
  if (!wantq && compq !== 0) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  } else if (ldt < Math.max(1, n)) {
    info = -4;
  } else if (ldq < 1 || (wantq && ldq < Math.max(1, n))) {
    info = -6;
  } else if (n > 0 && (ifst.val < 1 || ifst.val > n)) {
    info = -7;
  } else if (n > 0 && (ilst.val < 1 || ilst.val > n)) {
    info = -8;
  }
  if (info !== 0) {
    return info;
  }

  // Quick return if possible
  if (n <= 1) return 0;

  // Determine the first row of specified block
  // and find out it is 1 by 1 or 2 by 2.
  if (ifst.val > 1) {
    if (t[T_(ifst.val, ifst.val - 1)] !== ZERO) {
      ifst.val = ifst.val - 1;
    }
  }
  let nbf = 1;
  if (ifst.val < n) {
    if (t[T_(ifst.val + 1, ifst.val)] !== ZERO) {
      nbf = 2;
    }
  }

  // Determine the first row of the final block
  // and find out it is 1 by 1 or 2 by 2.
  if (ilst.val > 1) {
    if (t[T_(ilst.val, ilst.val - 1)] !== ZERO) {
      ilst.val = ilst.val - 1;
    }
  }
  let nbl = 1;
  if (ilst.val < n) {
    if (t[T_(ilst.val + 1, ilst.val)] !== ZERO) {
      nbl = 2;
    }
  }

  if (ifst.val === ilst.val) return 0;

  let here: number;

  if (ifst.val < ilst.val) {
    // Update ILST
    if (nbf === 2 && nbl === 1) ilst.val = ilst.val - 1;
    if (nbf === 1 && nbl === 2) ilst.val = ilst.val + 1;

    here = ifst.val;

    // label 10 loop: swap block with next one below
    while (true) {
      if (nbf === 1 || nbf === 2) {
        // Current block either 1 by 1 or 2 by 2
        let nbnext = 1;
        if (here + nbf + 1 <= n) {
          if (t[T_(here + nbf + 1, here + nbf)] !== ZERO) {
            nbnext = 2;
          }
        }
        info = dlaexc(
          wantq,
          n,
          t,
          tOff,
          ldt,
          q,
          qOff,
          ldq,
          here,
          nbf,
          nbnext,
          work,
          workOff
        );
        if (info !== 0) {
          ilst.val = here;
          return info;
        }
        here = here + nbnext;

        // Test if 2 by 2 block breaks into two 1 by 1 blocks
        if (nbf === 2) {
          if (t[T_(here + 1, here)] === ZERO) {
            nbf = 3;
          }
        }
      } else {
        // Current block consists of two 1 by 1 blocks each of which
        // must be swapped individually
        let nbnext = 1;
        if (here + 3 <= n) {
          if (t[T_(here + 3, here + 2)] !== ZERO) {
            nbnext = 2;
          }
        }
        info = dlaexc(
          wantq,
          n,
          t,
          tOff,
          ldt,
          q,
          qOff,
          ldq,
          here + 1,
          1,
          nbnext,
          work,
          workOff
        );
        if (info !== 0) {
          ilst.val = here;
          return info;
        }
        if (nbnext === 1) {
          // Swap two 1 by 1 blocks, no problems possible
          info = dlaexc(
            wantq,
            n,
            t,
            tOff,
            ldt,
            q,
            qOff,
            ldq,
            here,
            1,
            nbnext,
            work,
            workOff
          );
          here = here + 1;
        } else {
          // Recompute NBNEXT in case 2 by 2 split
          if (t[T_(here + 2, here + 1)] === ZERO) {
            nbnext = 1;
          }
          if (nbnext === 2) {
            // 2 by 2 Block did not split
            info = dlaexc(
              wantq,
              n,
              t,
              tOff,
              ldt,
              q,
              qOff,
              ldq,
              here,
              1,
              nbnext,
              work,
              workOff
            );
            if (info !== 0) {
              ilst.val = here;
              return info;
            }
            here = here + 2;
          } else {
            // 2 by 2 Block did split
            dlaexc(
              wantq,
              n,
              t,
              tOff,
              ldt,
              q,
              qOff,
              ldq,
              here,
              1,
              1,
              work,
              workOff
            );
            dlaexc(
              wantq,
              n,
              t,
              tOff,
              ldt,
              q,
              qOff,
              ldq,
              here + 1,
              1,
              1,
              work,
              workOff
            );
            here = here + 2;
          }
        }
      }
      if (here < ilst.val) {
        continue; // goto 10
      }
      break;
    }
  } else {
    // IFST > ILST: swap block upward
    here = ifst.val;

    // label 20 loop: swap block with next one above
    while (true) {
      if (nbf === 1 || nbf === 2) {
        // Current block either 1 by 1 or 2 by 2
        let nbnext = 1;
        if (here >= 3) {
          if (t[T_(here - 1, here - 2)] !== ZERO) {
            nbnext = 2;
          }
        }
        info = dlaexc(
          wantq,
          n,
          t,
          tOff,
          ldt,
          q,
          qOff,
          ldq,
          here - nbnext,
          nbnext,
          nbf,
          work,
          workOff
        );
        if (info !== 0) {
          ilst.val = here;
          return info;
        }
        here = here - nbnext;

        // Test if 2 by 2 block breaks into two 1 by 1 blocks
        if (nbf === 2) {
          if (t[T_(here + 1, here)] === ZERO) {
            nbf = 3;
          }
        }
      } else {
        // Current block consists of two 1 by 1 blocks each of which
        // must be swapped individually
        let nbnext = 1;
        if (here >= 3) {
          if (t[T_(here - 1, here - 2)] !== ZERO) {
            nbnext = 2;
          }
        }
        info = dlaexc(
          wantq,
          n,
          t,
          tOff,
          ldt,
          q,
          qOff,
          ldq,
          here - nbnext,
          nbnext,
          1,
          work,
          workOff
        );
        if (info !== 0) {
          ilst.val = here;
          return info;
        }
        if (nbnext === 1) {
          // Swap two 1 by 1 blocks, no problems possible
          info = dlaexc(
            wantq,
            n,
            t,
            tOff,
            ldt,
            q,
            qOff,
            ldq,
            here,
            nbnext,
            1,
            work,
            workOff
          );
          here = here - 1;
        } else {
          // Recompute NBNEXT in case 2 by 2 split
          if (t[T_(here, here - 1)] === ZERO) {
            nbnext = 1;
          }
          if (nbnext === 2) {
            // 2 by 2 Block did not split
            info = dlaexc(
              wantq,
              n,
              t,
              tOff,
              ldt,
              q,
              qOff,
              ldq,
              here - 1,
              2,
              1,
              work,
              workOff
            );
            if (info !== 0) {
              ilst.val = here;
              return info;
            }
            here = here - 2;
          } else {
            // 2 by 2 Block did split
            dlaexc(
              wantq,
              n,
              t,
              tOff,
              ldt,
              q,
              qOff,
              ldq,
              here,
              1,
              1,
              work,
              workOff
            );
            dlaexc(
              wantq,
              n,
              t,
              tOff,
              ldt,
              q,
              qOff,
              ldq,
              here - 1,
              1,
              1,
              work,
              workOff
            );
            here = here - 2;
          }
        }
      }
      if (here > ilst.val) {
        continue; // goto 20
      }
      break;
    }
  }
  ilst.val = here;

  return 0;
}
