// Translated from LAPACK/SRC/xerbla.f
// XERBLA is an error handler for the LAPACK routines.
// It is called by a LAPACK routine if an input parameter has an invalid value.

export function xerbla(srname: string, info: number): void {
  throw new Error(
    `** On entry to '${srname}' parameter number ${info} had an illegal value`
  );
}
