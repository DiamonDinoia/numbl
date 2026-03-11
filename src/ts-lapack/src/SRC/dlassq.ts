// Translated from SRC/dlassq.f90
// DLASSQ updates a scaled sum of squares representation.
//
// Returns scale_out and sumsq_out such that
//   (scale_out**2)*sumsq_out = x(1)**2 +...+ x(n)**2 + (scale_in**2)*sumsq_in
//
// Uses Blue's scaling constants for safe accumulation.
// Reference: Anderson E. (2017), Algorithm 978: Safe Scaling in the
// Level 1 BLAS, ACM Trans Math Softw 44:1--28

// Blue's scaling constants for float64 (double precision)
// radix = 2, minexponent = -1021, maxexponent = 1024, digits = 53
//
// tsml = 2^ceil((-1021 - 1) * 0.5) = 2^(-511)
const TSML = 1.4916681462400413e-154;
// tbig = 2^floor((1024 - 53 + 1) * 0.5) = 2^486
const TBIG = 1.997919072202235e146;
// ssml = 2^(-floor((-1021 - 53) * 0.5)) = 2^537
const SSML = 4.4989137945431964e161;
// sbig = 2^(-ceil((1024 + 53 - 1) * 0.5)) = 2^(-538)
const SBIG = 1.1113793747425387e-162;

/**
 * DLASSQ updates a scaled sum of squares.
 *
 * Given scl and sumsq, DLASSQ updates them such that
 *   (scl_out**2)*sumsq_out = x(1)**2 +...+ x(n)**2 + (scl_in**2)*sumsq_in
 *
 * @param n - Number of elements in the vector x.
 * @param x - The vector, dimension (1+(n-1)*|incx|).
 * @param xOff - Offset into array x.
 * @param incx - The increment between successive values of x.
 * @param scl - In/out scaling factor. Wrapped in { val } for pass-by-reference.
 * @param sumsq - In/out sum of squares. Wrapped in { val } for pass-by-reference.
 */
export function dlassq(
  n: number,
  x: Float64Array,
  xOff: number,
  incx: number,
  scl: { val: number },
  sumsq: { val: number }
): void {
  // Quick return if possible
  if (Number.isNaN(scl.val) || Number.isNaN(sumsq.val)) return;
  if (sumsq.val === 0.0) scl.val = 1.0;
  if (scl.val === 0.0) {
    scl.val = 1.0;
    sumsq.val = 0.0;
  }
  if (n <= 0) {
    return;
  }

  // Compute the sum of squares in 3 accumulators:
  //   abig -- sums of squares scaled down to avoid overflow
  //   asml -- sums of squares scaled up to avoid underflow
  //   amed -- sums of squares that do not require scaling
  let notbig = true;
  let asml = 0.0;
  let amed = 0.0;
  let abig = 0.0;
  let ix = 0; // 0-based starting index into x (relative to xOff)
  if (incx < 0) ix = -(n - 1) * incx;

  for (let i = 0; i < n; i++) {
    const ax = Math.abs(x[xOff + ix]);
    if (ax > TBIG) {
      abig = abig + ax * SBIG * (ax * SBIG);
      notbig = false;
    } else if (ax < TSML) {
      if (notbig) asml = asml + ax * SSML * (ax * SSML);
    } else {
      amed = amed + ax * ax;
    }
    ix = ix + incx;
  }

  // Put the existing sum of squares into one of the accumulators
  if (sumsq.val > 0.0) {
    const ax = scl.val * Math.sqrt(sumsq.val);
    if (ax > TBIG) {
      if (scl.val > 1.0) {
        scl.val = scl.val * SBIG;
        abig = abig + scl.val * (scl.val * sumsq.val);
      } else {
        // sumsq > tbig^2 => (sbig * (sbig * sumsq)) is representable
        abig = abig + scl.val * (scl.val * (SBIG * (SBIG * sumsq.val)));
      }
    } else if (ax < TSML) {
      if (notbig) {
        if (scl.val < 1.0) {
          scl.val = scl.val * SSML;
          asml = asml + scl.val * (scl.val * sumsq.val);
        } else {
          // sumsq < tsml^2 => (ssml * (ssml * sumsq)) is representable
          asml = asml + scl.val * (scl.val * (SSML * (SSML * sumsq.val)));
        }
      }
    } else {
      amed = amed + scl.val * (scl.val * sumsq.val);
    }
  }

  // Combine abig and amed or amed and asml if more than one
  // accumulator was used.
  if (abig > 0.0) {
    // Combine abig and amed if abig > 0.
    if (amed > 0.0 || Number.isNaN(amed)) {
      abig = abig + amed * SBIG * SBIG;
    }
    scl.val = 1.0 / SBIG;
    sumsq.val = abig;
  } else if (asml > 0.0) {
    // Combine amed and asml if asml > 0.
    if (amed > 0.0 || Number.isNaN(amed)) {
      const amedSqrt = Math.sqrt(amed);
      const asmlScaled = Math.sqrt(asml) / SSML;
      let ymin: number, ymax: number;
      if (asmlScaled > amedSqrt) {
        ymin = amedSqrt;
        ymax = asmlScaled;
      } else {
        ymin = asmlScaled;
        ymax = amedSqrt;
      }
      scl.val = 1.0;
      sumsq.val = ymax * ymax * (1.0 + (ymin / ymax) * (ymin / ymax));
    } else {
      scl.val = 1.0 / SSML;
      sumsq.val = asml;
    }
  } else {
    // Otherwise all values are mid-range or zero
    scl.val = 1.0;
    sumsq.val = amed;
  }
}
