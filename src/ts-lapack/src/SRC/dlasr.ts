// Translated from LAPACK/SRC/dlasr.f
// DLASR applies a sequence of plane rotations to a real matrix A,
// from either the left or the right.
//
// When side = LEFT, the transformation takes the form A := P*A
// When side = RIGHT, the transformation takes the form A := A*P**T
//
// Array indexing convention (column-major):
//   A(I,J) => a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)

import {
  LEFT,
  RIGHT,
  PIVOT_V,
  PIVOT_T,
  PIVOT_B,
  DIRECT_F,
  DIRECT_B,
} from "../utils/constants.js";

export function dlasr(
  side: number,
  pivot: number,
  direct: number,
  m: number,
  n: number,
  c: Float64Array,
  cOff: number,
  s: Float64Array,
  sOff: number,
  a: Float64Array,
  aOff: number,
  lda: number
): void {
  const ONE = 1.0;
  const ZERO = 0.0;

  // Test the input parameters
  let info = 0;
  if (side !== LEFT && side !== RIGHT) {
    info = 1;
  } else if (pivot !== PIVOT_V && pivot !== PIVOT_T && pivot !== PIVOT_B) {
    info = 2;
  } else if (direct !== DIRECT_F && direct !== DIRECT_B) {
    info = 3;
  } else if (m < 0) {
    info = 4;
  } else if (n < 0) {
    info = 5;
  } else if (lda < Math.max(1, m)) {
    info = 9;
  }
  if (info !== 0) {
    throw new Error(`DLASR: illegal argument ${info}`);
  }

  // Quick return if possible
  if (m === 0 || n === 0) return;

  let ctemp: number;
  let stemp: number;
  let temp: number;
  let i: number;
  let j: number;

  if (side === LEFT) {
    // Form P * A
    if (pivot === PIVOT_V) {
      if (direct === DIRECT_F) {
        for (j = 1; j <= m - 1; j++) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= n; i++) {
              temp = a[aOff + j + (i - 1) * lda];
              a[aOff + j + (i - 1) * lda] =
                ctemp * temp - stemp * a[aOff + (j - 1) + (i - 1) * lda];
              a[aOff + (j - 1) + (i - 1) * lda] =
                stemp * temp + ctemp * a[aOff + (j - 1) + (i - 1) * lda];
            }
          }
        }
      } else if (direct === DIRECT_B) {
        for (j = m - 1; j >= 1; j--) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= n; i++) {
              temp = a[aOff + j + (i - 1) * lda];
              a[aOff + j + (i - 1) * lda] =
                ctemp * temp - stemp * a[aOff + (j - 1) + (i - 1) * lda];
              a[aOff + (j - 1) + (i - 1) * lda] =
                stemp * temp + ctemp * a[aOff + (j - 1) + (i - 1) * lda];
            }
          }
        }
      }
    } else if (pivot === PIVOT_T) {
      if (direct === DIRECT_F) {
        for (j = 2; j <= m; j++) {
          ctemp = c[cOff + (j - 2)];
          stemp = s[sOff + (j - 2)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= n; i++) {
              temp = a[aOff + (j - 1) + (i - 1) * lda];
              a[aOff + (j - 1) + (i - 1) * lda] =
                ctemp * temp - stemp * a[aOff + 0 + (i - 1) * lda];
              a[aOff + 0 + (i - 1) * lda] =
                stemp * temp + ctemp * a[aOff + 0 + (i - 1) * lda];
            }
          }
        }
      } else if (direct === DIRECT_B) {
        for (j = m; j >= 2; j--) {
          ctemp = c[cOff + (j - 2)];
          stemp = s[sOff + (j - 2)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= n; i++) {
              temp = a[aOff + (j - 1) + (i - 1) * lda];
              a[aOff + (j - 1) + (i - 1) * lda] =
                ctemp * temp - stemp * a[aOff + 0 + (i - 1) * lda];
              a[aOff + 0 + (i - 1) * lda] =
                stemp * temp + ctemp * a[aOff + 0 + (i - 1) * lda];
            }
          }
        }
      }
    } else if (pivot === PIVOT_B) {
      if (direct === DIRECT_F) {
        for (j = 1; j <= m - 1; j++) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= n; i++) {
              temp = a[aOff + (j - 1) + (i - 1) * lda];
              a[aOff + (j - 1) + (i - 1) * lda] =
                stemp * a[aOff + (m - 1) + (i - 1) * lda] + ctemp * temp;
              a[aOff + (m - 1) + (i - 1) * lda] =
                ctemp * a[aOff + (m - 1) + (i - 1) * lda] - stemp * temp;
            }
          }
        }
      } else if (direct === DIRECT_B) {
        for (j = m - 1; j >= 1; j--) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= n; i++) {
              temp = a[aOff + (j - 1) + (i - 1) * lda];
              a[aOff + (j - 1) + (i - 1) * lda] =
                stemp * a[aOff + (m - 1) + (i - 1) * lda] + ctemp * temp;
              a[aOff + (m - 1) + (i - 1) * lda] =
                ctemp * a[aOff + (m - 1) + (i - 1) * lda] - stemp * temp;
            }
          }
        }
      }
    }
  } else if (side === RIGHT) {
    // Form A * P**T
    if (pivot === PIVOT_V) {
      if (direct === DIRECT_F) {
        for (j = 1; j <= n - 1; j++) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= m; i++) {
              temp = a[aOff + (i - 1) + j * lda];
              a[aOff + (i - 1) + j * lda] =
                ctemp * temp - stemp * a[aOff + (i - 1) + (j - 1) * lda];
              a[aOff + (i - 1) + (j - 1) * lda] =
                stemp * temp + ctemp * a[aOff + (i - 1) + (j - 1) * lda];
            }
          }
        }
      } else if (direct === DIRECT_B) {
        for (j = n - 1; j >= 1; j--) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= m; i++) {
              temp = a[aOff + (i - 1) + j * lda];
              a[aOff + (i - 1) + j * lda] =
                ctemp * temp - stemp * a[aOff + (i - 1) + (j - 1) * lda];
              a[aOff + (i - 1) + (j - 1) * lda] =
                stemp * temp + ctemp * a[aOff + (i - 1) + (j - 1) * lda];
            }
          }
        }
      }
    } else if (pivot === PIVOT_T) {
      if (direct === DIRECT_F) {
        for (j = 2; j <= n; j++) {
          ctemp = c[cOff + (j - 2)];
          stemp = s[sOff + (j - 2)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= m; i++) {
              temp = a[aOff + (i - 1) + (j - 1) * lda];
              a[aOff + (i - 1) + (j - 1) * lda] =
                ctemp * temp - stemp * a[aOff + (i - 1) + 0 * lda];
              a[aOff + (i - 1) + 0 * lda] =
                stemp * temp + ctemp * a[aOff + (i - 1) + 0 * lda];
            }
          }
        }
      } else if (direct === DIRECT_B) {
        for (j = n; j >= 2; j--) {
          ctemp = c[cOff + (j - 2)];
          stemp = s[sOff + (j - 2)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= m; i++) {
              temp = a[aOff + (i - 1) + (j - 1) * lda];
              a[aOff + (i - 1) + (j - 1) * lda] =
                ctemp * temp - stemp * a[aOff + (i - 1) + 0 * lda];
              a[aOff + (i - 1) + 0 * lda] =
                stemp * temp + ctemp * a[aOff + (i - 1) + 0 * lda];
            }
          }
        }
      }
    } else if (pivot === PIVOT_B) {
      if (direct === DIRECT_F) {
        for (j = 1; j <= n - 1; j++) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= m; i++) {
              temp = a[aOff + (i - 1) + (j - 1) * lda];
              a[aOff + (i - 1) + (j - 1) * lda] =
                stemp * a[aOff + (i - 1) + (n - 1) * lda] + ctemp * temp;
              a[aOff + (i - 1) + (n - 1) * lda] =
                ctemp * a[aOff + (i - 1) + (n - 1) * lda] - stemp * temp;
            }
          }
        }
      } else if (direct === DIRECT_B) {
        for (j = n - 1; j >= 1; j--) {
          ctemp = c[cOff + (j - 1)];
          stemp = s[sOff + (j - 1)];
          if (ctemp !== ONE || stemp !== ZERO) {
            for (i = 1; i <= m; i++) {
              temp = a[aOff + (i - 1) + (j - 1) * lda];
              a[aOff + (i - 1) + (j - 1) * lda] =
                stemp * a[aOff + (i - 1) + (n - 1) * lda] + ctemp * temp;
              a[aOff + (i - 1) + (n - 1) * lda] =
                ctemp * a[aOff + (i - 1) + (n - 1) * lda] - stemp * temp;
            }
          }
        }
      }
    }
  }
}
