// Translated from SRC/dlaswp.f
// DLASWP performs a series of row interchanges on the matrix A.
// One row interchange is initiated for each of rows K1 through K2 of A.
//
// Array indexing convention (matching Fortran column-major):
//   A(I,J)     =>  a[aOff + (I-1) + (J-1)*lda]   (I,J are 1-based)
//   IPIV(IX)   =>  ipiv[ipivOff + (IX-1)]          (IX is 1-based)

export function dlaswp(
  n: number,
  a: Float64Array,
  aOff: number,
  lda: number,
  k1: number,
  k2: number,
  ipiv: Int32Array,
  ipivOff: number,
  incx: number
): void {
  let ix0: number, i1: number, i2: number, inc: number;

  if (incx > 0) {
    ix0 = k1;
    i1 = k1;
    i2 = k2;
    inc = 1;
  } else if (incx < 0) {
    ix0 = k1 + (k1 - k2) * incx;
    i1 = k2;
    i2 = k1;
    inc = -1;
  } else {
    return;
  }

  const n32 = Math.floor(n / 32) * 32;

  if (n32 !== 0) {
    for (let j = 1; j <= n32; j += 32) {
      let ix = ix0;
      for (let i = i1; inc > 0 ? i <= i2 : i >= i2; i += inc) {
        const ip = ipiv[ipivOff + (ix - 1)]; // IPIV(IX), 1-based
        if (ip !== i) {
          for (let kk = j; kk <= j + 31; kk++) {
            const tmp = a[aOff + (i - 1) + (kk - 1) * lda];
            a[aOff + (i - 1) + (kk - 1) * lda] =
              a[aOff + (ip - 1) + (kk - 1) * lda];
            a[aOff + (ip - 1) + (kk - 1) * lda] = tmp;
          }
        }
        ix += incx;
      }
    }
  }

  if (n32 !== n) {
    const n32start = n32 + 1;
    let ix = ix0;
    for (let i = i1; inc > 0 ? i <= i2 : i >= i2; i += inc) {
      const ip = ipiv[ipivOff + (ix - 1)];
      if (ip !== i) {
        for (let kk = n32start; kk <= n; kk++) {
          const tmp = a[aOff + (i - 1) + (kk - 1) * lda];
          a[aOff + (i - 1) + (kk - 1) * lda] =
            a[aOff + (ip - 1) + (kk - 1) * lda];
          a[aOff + (ip - 1) + (kk - 1) * lda] = tmp;
        }
      }
      ix += incx;
    }
  }
}
