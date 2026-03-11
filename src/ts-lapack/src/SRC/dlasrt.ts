// Translated from LAPACK/SRC/dlasrt.f
// DLASRT sorts the numbers in D in increasing order (if id = SORT_INC)
// or in decreasing order (if id = SORT_DEC).
//
// Uses Quick Sort, reverting to Insertion sort on arrays of size <= 20.
// Dimension of STACK limits N to about 2**32.
//
// Returns INFO: 0 = success, <0 = -i means the i-th argument had an illegal value.

import { SORT_INC, SORT_DEC } from "../utils/constants.js";

export function dlasrt(
  id: number,
  n: number,
  d: Float64Array,
  dOff: number
): number {
  const SELECT = 20;

  // Test the input parameters.
  let info = 0;
  let dir = -1;
  if (id === SORT_DEC) {
    dir = 0;
  } else if (id === SORT_INC) {
    dir = 1;
  }
  if (dir === -1) {
    info = -1;
  } else if (n < 0) {
    info = -2;
  }
  if (info !== 0) {
    return info;
  }

  // Quick return if possible
  if (n <= 1) {
    return 0;
  }

  // Manual stack for quicksort (STACK(2, 32) in Fortran, 1-indexed)
  // stack1[k] = start, stack2[k] = end, for k = 1..stkpnt
  const stack1 = new Int32Array(33);
  const stack2 = new Int32Array(33);

  let stkpnt = 1;
  stack1[1] = 1;
  stack2[1] = n;

  while (stkpnt > 0) {
    const start = stack1[stkpnt];
    const endd = stack2[stkpnt];
    stkpnt--;

    if (endd - start <= SELECT && endd - start > 0) {
      // Insertion sort on D(START:ENDD)
      if (dir === 0) {
        // Sort into decreasing order
        for (let i = start + 1; i <= endd; i++) {
          for (let j = i; j >= start + 1; j--) {
            if (d[dOff + (j - 1)] > d[dOff + (j - 2)]) {
              const dmnmx = d[dOff + (j - 1)];
              d[dOff + (j - 1)] = d[dOff + (j - 2)];
              d[dOff + (j - 2)] = dmnmx;
            } else {
              break;
            }
          }
        }
      } else {
        // Sort into increasing order
        for (let i = start + 1; i <= endd; i++) {
          for (let j = i; j >= start + 1; j--) {
            if (d[dOff + (j - 1)] < d[dOff + (j - 2)]) {
              const dmnmx = d[dOff + (j - 1)];
              d[dOff + (j - 1)] = d[dOff + (j - 2)];
              d[dOff + (j - 2)] = dmnmx;
            } else {
              break;
            }
          }
        }
      }
    } else if (endd - start > SELECT) {
      // Partition D(START:ENDD) and stack parts, largest one first
      // Choose partition entry as median of 3
      const d1 = d[dOff + (start - 1)];
      const d2 = d[dOff + (endd - 1)];
      const ii = Math.trunc((start + endd) / 2);
      const d3 = d[dOff + (ii - 1)];
      let dmnmx: number;
      if (d1 < d2) {
        if (d3 < d1) {
          dmnmx = d1;
        } else if (d3 < d2) {
          dmnmx = d3;
        } else {
          dmnmx = d2;
        }
      } else {
        if (d3 < d2) {
          dmnmx = d2;
        } else if (d3 < d1) {
          dmnmx = d3;
        } else {
          dmnmx = d1;
        }
      }

      let i: number;
      let j: number;

      if (dir === 0) {
        // Sort into decreasing order
        i = start - 1;
        j = endd + 1;
        while (true) {
          do {
            j--;
          } while (d[dOff + (j - 1)] < dmnmx);
          do {
            i++;
          } while (d[dOff + (i - 1)] > dmnmx);
          if (i < j) {
            const tmp = d[dOff + (i - 1)];
            d[dOff + (i - 1)] = d[dOff + (j - 1)];
            d[dOff + (j - 1)] = tmp;
          } else {
            break;
          }
        }
        if (j - start > endd - j - 1) {
          stkpnt++;
          stack1[stkpnt] = start;
          stack2[stkpnt] = j;
          stkpnt++;
          stack1[stkpnt] = j + 1;
          stack2[stkpnt] = endd;
        } else {
          stkpnt++;
          stack1[stkpnt] = j + 1;
          stack2[stkpnt] = endd;
          stkpnt++;
          stack1[stkpnt] = start;
          stack2[stkpnt] = j;
        }
      } else {
        // Sort into increasing order
        i = start - 1;
        j = endd + 1;
        while (true) {
          do {
            j--;
          } while (d[dOff + (j - 1)] > dmnmx);
          do {
            i++;
          } while (d[dOff + (i - 1)] < dmnmx);
          if (i < j) {
            const tmp = d[dOff + (i - 1)];
            d[dOff + (i - 1)] = d[dOff + (j - 1)];
            d[dOff + (j - 1)] = tmp;
          } else {
            break;
          }
        }
        if (j - start > endd - j - 1) {
          stkpnt++;
          stack1[stkpnt] = start;
          stack2[stkpnt] = j;
          stkpnt++;
          stack1[stkpnt] = j + 1;
          stack2[stkpnt] = endd;
        } else {
          stkpnt++;
          stack1[stkpnt] = j + 1;
          stack2[stkpnt] = endd;
          stkpnt++;
          stack1[stkpnt] = start;
          stack2[stkpnt] = j;
        }
      }
    }
  }

  return 0;
}
