// Translated from SRC/dgehd2.f
// DGEHD2 reduces a real general matrix A to upper Hessenberg form H by
// an orthogonal similarity transformation:  Q**T * A * Q = H .
//
// This is the unblocked algorithm.
//
// Array indexing convention (column-major, matching Fortran):
//   A(I,J)  =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   TAU(I)  =>  tau[tauOff + (I-1)]              (I is 1-based)
//
// Parameters:
//   n      - order of the matrix A (n >= 0)
//   ilo    - 1 <= ilo <= ihi <= max(1,n)
//   ihi    - assumed that A is already upper triangular in rows/columns
//            1:ilo-1 and ihi+1:n
//   a      - Float64Array; on entry the n-by-n general matrix; on exit upper
//            Hessenberg form with reflector vectors below the first subdiagonal
//   aOff   - offset into a for A(1,1)
//   lda    - leading dimension of a (>= max(1,n))
//   tau    - Float64Array of length n-1; scalar factors of reflectors
//   tauOff - offset into tau for TAU(1)
//   work   - Float64Array workspace of length n
//   workOff - offset into work
//
// Returns INFO (0 = success, < 0 = illegal argument)

import { dlarfg } from "./dlarfg.js";
import { dlarf1f } from "./dlarf1f.js";
import { LEFT, RIGHT } from "../utils/constants.js";

export function dgehd2(
  n: number,
  ilo: number,
  ihi: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  tau: Float64Array,
  tauOff: number,
  work: Float64Array,
  workOff: number
): number {
  // Test the input parameters
  let info = 0;
  if (n < 0) {
    info = -1;
  } else if (ilo < 1 || ilo > Math.max(1, n)) {
    info = -2;
  } else if (ihi < Math.min(ilo, n) || ihi > n) {
    info = -3;
  } else if (lda < Math.max(1, n)) {
    info = -5;
  }
  if (info !== 0) {
    return info;
  }

  for (let i = ilo; i <= ihi - 1; i++) {
    // Compute elementary reflector H(i) to annihilate A(i+2:ihi,i)
    //
    // Fortran: CALL DLARFG( IHI-I, A(I+1,I), A(MIN(I+2,N),I), 1, TAU(I) )
    // A(I+1,I) => a[aOff + (I+1-1) + (I-1)*lda] = a[aOff + I + (I-1)*lda]
    // A(MIN(I+2,N),I) => a[aOff + (MIN(I+2,N)-1) + (I-1)*lda]
    const alphaIdx = aOff + i + (i - 1) * lda;
    const xIdx = aOff + (Math.min(i + 2, n) - 1) + (i - 1) * lda;
    const result = dlarfg(ihi - i, a[alphaIdx], a, xIdx, 1);
    a[alphaIdx] = result.alpha;
    tau[tauOff + (i - 1)] = result.tau;

    // Apply H(i) to A(1:ihi,i+1:ihi) from the right
    //
    // Fortran: CALL DLARF1F( 'Right', IHI, IHI-I, A(I+1,I), 1, TAU(I),
    //                        A(1,I+1), LDA, WORK )
    dlarf1f(
      RIGHT,
      ihi,
      ihi - i,
      a,
      aOff + i + (i - 1) * lda, // A(I+1,I)
      1,
      tau[tauOff + (i - 1)],
      a,
      aOff + i * lda, // A(1,I+1)
      lda,
      work,
      workOff
    );

    // Apply H(i) to A(i+1:ihi,i+1:n) from the left
    //
    // Fortran: CALL DLARF1F( 'Left', IHI-I, N-I, A(I+1,I), 1, TAU(I),
    //                        A(I+1,I+1), LDA, WORK )
    dlarf1f(
      LEFT,
      ihi - i,
      n - i,
      a,
      aOff + i + (i - 1) * lda, // A(I+1,I)
      1,
      tau[tauOff + (i - 1)],
      a,
      aOff + i + i * lda, // A(I+1,I+1)
      lda,
      work,
      workOff
    );
  }

  return 0;
}
