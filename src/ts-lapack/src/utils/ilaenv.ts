/* eslint-disable @typescript-eslint/no-unused-vars */
// Translated from LAPACK/SRC/ilaenv.f
// ILAENV returns problem-dependent parameters (e.g., block sizes) used by
// the LAPACK routines. Only the subset needed by translated routines is
// implemented here; everything else returns 1.
//
// ispec=1: optimal block size for the named routine

export function ilaenv(
  ispec: number,
  name: string,
  _opts: string,
  _n1: number,
  _n2: number,
  _n3: number,
  _n4: number
): number {
  if (ispec === 1) {
    const uname = name.toUpperCase().trim();
    switch (uname) {
      case "DGETRF":
        return 64;
      case "DGETRI":
        return 64;
      case "DTRTRI":
        return 64;
      case "DGEHRD":
        return 64;
      case "DORGHR":
        return 64;
      case "DGEBRD":
        return 64;
      case "DGESVD":
        return 64;
      case "DORGLQ":
        return 64;
      case "DORMLQ":
        return 64;
      case "DORGBR":
        return 64;
      case "DORMBR":
        return 64;
      case "DGELQF":
        return 64;
      default:
        return 1;
    }
  }

  // ISPECs 12-16 are used by DLAQR0/DLAQR4 for tuning parameters.
  // Values taken from the reference LAPACK ilaenv.f defaults.
  if (ispec === 12) {
    // NMIN: crossover point for multishift QR vs DLAHQR.
    // Matrices of order < NMIN use DLAHQR.
    return 75;
  }
  if (ispec === 13) {
    // NWR: recommended deflation window size.
    return 13;
  }
  if (ispec === 14) {
    // NIBBLE: nibble crossover point; controls when to skip QR sweep.
    return 14;
  }
  if (ispec === 15) {
    // NSR: recommended number of simultaneous shifts.
    return 14;
  }
  if (ispec === 16) {
    // KACC22: specifies whether to accumulate reflections and use
    // 2-by-2 block structure during matrix-matrix multiply.
    // 0 = do not accumulate, 1 = accumulate, 2 = use block structure.
    return 0;
  }

  // ISPEC=6: crossover point for SVD (when to use QR/LQ preprocessing)
  if (ispec === 6) {
    return Math.max(Math.floor((5.0 / 3.0) * Math.min(_n1, _n2)), 1);
  }

  return 1;
}
