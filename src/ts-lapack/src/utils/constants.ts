// Integer flag constants replacing LAPACK/BLAS string arguments

// trans / transa / transb
export const NOTRANS = 0; // 'N' — no transpose
export const TRANS = 1; // 'T' — transpose
export const CONJTRANS = 2; // 'C' — conjugate transpose

// uplo
export const UPPER = 0; // 'U' — upper triangular
export const LOWER = 1; // 'L' — lower triangular

// diag
export const NONUNIT = 0; // 'N' — non-unit diagonal
export const UNIT = 1; // 'U' — unit diagonal

// side
export const LEFT = 0; // 'L' — left side
export const RIGHT = 1; // 'R' — right side

// cmach codes for dlamch
export const MACH_EPS = 0; // 'E' — relative machine precision (unit roundoff)
export const MACH_SFMIN = 1; // 'S' — safe minimum: 1/sfmin does not overflow
export const MACH_BASE = 2; // 'B' — base of the machine (2 for IEEE 754)
export const MACH_PREC = 3; // 'P' — eps * base
export const MACH_T = 4; // 'N' — number of mantissa digits (53 for float64)
export const MACH_RND = 5; // 'R' — 1.0 (rounding mode; round-to-nearest)
export const MACH_EMIN = 6; // 'M' — minimum exponent before underflow (-1021)
export const MACH_RMIN = 7; // 'U' — underflow threshold (smallest positive normal)
export const MACH_EMAX = 8; // 'L' — largest exponent before overflow (1024)
export const MACH_RMAX = 9; // 'O' — overflow threshold

// dlasr pivot type
export const PIVOT_V = 0; // 'V' — variable pivot
export const PIVOT_T = 1; // 'T' — top pivot
export const PIVOT_B = 2; // 'B' — bottom pivot

// dlasr direction
export const DIRECT_F = 0; // 'F' — forward
export const DIRECT_B = 1; // 'B' — backward

// dlasrt sort order
export const SORT_INC = 0; // 'I' — increasing
export const SORT_DEC = 1; // 'D' — decreasing

// dorgbr / dormbr vect type
export const VECT_Q = 0; // 'Q'
export const VECT_P = 1; // 'P'
