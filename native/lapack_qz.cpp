/**
 * qz() and qzComplex() — Generalized Schur (QZ) factorization via LAPACK.
 *
 *   qz(dataA, dataB, n, computeEigvecs) — real case via dgges/dtgevc
 *   qzComplex(ARe, AIm, BRe, BIm, n, computeEigvecs) — complex case via zgges/ztgevc
 *
 *   Computes the generalized Schur decomposition of (A, B):
 *     Q*A*Z = AA, Q*B*Z = BB
 *   where Q and Z are unitary (orthogonal for real case).
 */

#include "lapack_common.h"

Napi::Value Qz(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 4
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    Napi::TypeError::New(env,
      "qz: expected (Float64Array dataA, Float64Array dataB, number n, "
      "boolean computeEigvecs)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrA = info[0].As<Napi::TypedArray>();
  auto arrB = info[1].As<Napi::TypedArray>();
  if (arrA.TypedArrayType() != napi_float64_array ||
      arrB.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "qz: data must be Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[2].As<Napi::Number>().Int32Value();
  bool computeEigvecs = info[3].As<Napi::Boolean>().Value();

  if (n <= 0
      || static_cast<int>(arrA.ElementLength()) != n * n
      || static_cast<int>(arrB.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "qz: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto f64A = info[0].As<Napi::Float64Array>();
  auto f64B = info[1].As<Napi::Float64Array>();

  // Copy inputs (dgges overwrites A and B)
  std::vector<double> a(n * n), b(n * n);
  std::memcpy(a.data(), f64A.Data(), n * n * sizeof(double));
  std::memcpy(b.data(), f64B.Data(), n * n * sizeof(double));

  // Output arrays
  std::vector<double> alphar(n), alphai(n), beta(n);
  std::vector<double> vsl(n * n), vsr(n * n);

  char jobvsl = 'V';
  char jobvsr = 'V';
  char sort = 'N';  // no eigenvalue sorting
  int sdim = 0;
  int info_val = 0;

  // BWORK not referenced when sort='N', but allocate anyway
  std::vector<int> bwork(n);

  // Workspace query
  int lwork = -1;
  double work_query = 0.0;
  dgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alphar.data(), alphai.data(), beta.data(),
         vsl.data(), &n, vsr.data(), &n,
         &work_query, &lwork, bwork.data(), &info_val);

  lwork = static_cast<int>(work_query);
  if (lwork < 1) lwork = std::max(1, 8 * n + 16);
  std::vector<double> work(lwork);

  // Reset a and b (workspace query may have modified them)
  std::memcpy(a.data(), f64A.Data(), n * n * sizeof(double));
  std::memcpy(b.data(), f64B.Data(), n * n * sizeof(double));

  // Compute generalized Schur decomposition
  dgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alphar.data(), alphai.data(), beta.data(),
         vsl.data(), &n, vsr.data(), &n,
         work.data(), &lwork, bwork.data(), &info_val);

  if (info_val != 0) {
    if (info_val > 0) {
      Napi::Error::New(env, "qz: QZ iteration failed to converge in dgges")
          .ThrowAsJavaScriptException();
    } else {
      Napi::Error::New(env, "qz: illegal argument in dgges")
          .ThrowAsJavaScriptException();
    }
    return env.Null();
  }

  // Build result object
  auto result = Napi::Object::New(env);

  // AA = a (overwritten by dgges with the quasi-triangular Schur form of A)
  auto aa_arr = Napi::Float64Array::New(env, static_cast<size_t>(n * n));
  std::memcpy(aa_arr.Data(), a.data(), n * n * sizeof(double));
  result.Set("AA", aa_arr);

  // BB = b (overwritten by dgges with the triangular Schur form of B)
  auto bb_arr = Napi::Float64Array::New(env, static_cast<size_t>(n * n));
  std::memcpy(bb_arr.Data(), b.data(), n * n * sizeof(double));
  result.Set("BB", bb_arr);

  // Q = VSL^T  (MATLAB convention: Q*A*Z = AA, so Q = VSL')
  auto q_arr = Napi::Float64Array::New(env, static_cast<size_t>(n * n));
  {
    double* q = q_arr.Data();
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        q[i + j * n] = vsl[j + i * n];  // Q(i,j) = VSL(j,i)
  }
  result.Set("Q", q_arr);

  // Z = VSR (right Schur vectors, no transpose needed)
  auto z_arr = Napi::Float64Array::New(env, static_cast<size_t>(n * n));
  std::memcpy(z_arr.Data(), vsr.data(), n * n * sizeof(double));
  result.Set("Z", z_arr);

  // Generalized eigenvalue components
  auto alphar_arr = Napi::Float64Array::New(env, static_cast<size_t>(n));
  std::memcpy(alphar_arr.Data(), alphar.data(), n * sizeof(double));
  result.Set("alphar", alphar_arr);

  auto alphai_arr = Napi::Float64Array::New(env, static_cast<size_t>(n));
  std::memcpy(alphai_arr.Data(), alphai.data(), n * sizeof(double));
  result.Set("alphai", alphai_arr);

  auto beta_arr = Napi::Float64Array::New(env, static_cast<size_t>(n));
  std::memcpy(beta_arr.Data(), beta.data(), n * sizeof(double));
  result.Set("beta", beta_arr);

  // Optionally compute generalized eigenvectors via dtgevc
  if (computeEigvecs) {
    // Copy Schur vectors for backtransform
    std::vector<double> vr(vsr);   // right: start with VSR
    std::vector<double> vl(vsl);   // left: start with VSL

    char side = 'B';     // both left and right
    char howmny = 'B';   // backtransform using input VL/VR
    std::vector<int> select(n);  // not referenced when howmny='B'
    int mm = n;
    int m_out = 0;
    std::vector<double> work_tgevc(6 * n);

    dtgevc_(&side, &howmny, select.data(), &n,
            a.data(), &n, b.data(), &n,
            vl.data(), &n, vr.data(), &n,
            &mm, &m_out, work_tgevc.data(), &info_val);

    if (info_val != 0) {
      Napi::Error::New(env, "qz: dtgevc failed to compute eigenvectors")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    // V = right generalized eigenvectors (packed format, like dgeev)
    auto v_arr = Napi::Float64Array::New(env, static_cast<size_t>(n * n));
    std::memcpy(v_arr.Data(), vr.data(), n * n * sizeof(double));
    result.Set("V", v_arr);

    // W = left generalized eigenvectors (packed format, like dgeev)
    auto w_arr = Napi::Float64Array::New(env, static_cast<size_t>(n * n));
    std::memcpy(w_arr.Data(), vl.data(), n * n * sizeof(double));
    result.Set("W", w_arr);
  }

  return result;
}

// ── qzComplex() ──────────────────────────────────────────────────────────────

Napi::Value QzComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 6
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsTypedArray()
      || !info[3].IsTypedArray()
      || !info[4].IsNumber()
      || !info[5].IsBoolean()) {
    Napi::TypeError::New(env,
      "qzComplex: expected (Float64Array ARe, Float64Array AIm, "
      "Float64Array BRe, Float64Array BIm, number n, boolean computeEigvecs)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int n = info[4].As<Napi::Number>().Int32Value();
  bool computeEigvecs = info[5].As<Napi::Boolean>().Value();

  auto f64ARe = info[0].As<Napi::Float64Array>();
  auto f64AIm = info[1].As<Napi::Float64Array>();
  auto f64BRe = info[2].As<Napi::Float64Array>();
  auto f64BIm = info[3].As<Napi::Float64Array>();

  if (n <= 0
      || static_cast<int>(f64ARe.ElementLength()) != n * n
      || static_cast<int>(f64AIm.ElementLength()) != n * n
      || static_cast<int>(f64BRe.ElementLength()) != n * n
      || static_cast<int>(f64BIm.ElementLength()) != n * n) {
    Napi::RangeError::New(env, "qzComplex: data.length must equal n*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // Convert to interleaved complex format
  std::vector<lapack_complex_double> a(n * n), b(n * n);
  for (int i = 0; i < n * n; ++i) {
    a[i].real = f64ARe[i];
    a[i].imag = f64AIm[i];
    b[i].real = f64BRe[i];
    b[i].imag = f64BIm[i];
  }

  // Output arrays (for complex zgges, both alpha and beta are complex)
  std::vector<lapack_complex_double> alpha(n);
  std::vector<lapack_complex_double> beta_c(n);
  std::vector<lapack_complex_double> vsl(n * n), vsr(n * n);

  char jobvsl = 'V';
  char jobvsr = 'V';
  char sort = 'N';
  int sdim = 0;
  int info_val = 0;
  std::vector<int> bwork(n);

  // Workspace query
  int lwork = -1;
  lapack_complex_double work_query;
  std::vector<double> rwork(8 * n);
  zgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alpha.data(), beta_c.data(),
         vsl.data(), &n, vsr.data(), &n,
         &work_query, &lwork, rwork.data(), bwork.data(), &info_val);

  lwork = static_cast<int>(work_query.real);
  if (lwork < 1) lwork = std::max(1, 2 * n);
  std::vector<lapack_complex_double> work(lwork);

  // Reset a and b
  for (int i = 0; i < n * n; ++i) {
    a[i].real = f64ARe[i];
    a[i].imag = f64AIm[i];
    b[i].real = f64BRe[i];
    b[i].imag = f64BIm[i];
  }

  // Compute
  zgges_(&jobvsl, &jobvsr, &sort, nullptr,
         &n, a.data(), &n, b.data(), &n,
         &sdim, alpha.data(), beta_c.data(),
         vsl.data(), &n, vsr.data(), &n,
         work.data(), &lwork, rwork.data(), bwork.data(), &info_val);

  if (info_val != 0) {
    if (info_val > 0) {
      Napi::Error::New(env, "qzComplex: QZ iteration failed to converge in zgges")
          .ThrowAsJavaScriptException();
    } else {
      Napi::Error::New(env, "qzComplex: illegal argument in zgges")
          .ThrowAsJavaScriptException();
    }
    return env.Null();
  }

  auto result = Napi::Object::New(env);
  size_t nn = static_cast<size_t>(n * n);
  size_t sn = static_cast<size_t>(n);

  // AA (complex) = a after zgges
  auto aaRe = Napi::Float64Array::New(env, nn);
  auto aaIm = Napi::Float64Array::New(env, nn);
  for (int i = 0; i < n * n; ++i) {
    aaRe[i] = a[i].real;
    aaIm[i] = a[i].imag;
  }
  result.Set("AARe", aaRe);
  result.Set("AAIm", aaIm);

  // BB (complex) = b after zgges
  auto bbRe = Napi::Float64Array::New(env, nn);
  auto bbIm = Napi::Float64Array::New(env, nn);
  for (int i = 0; i < n * n; ++i) {
    bbRe[i] = b[i].real;
    bbIm[i] = b[i].imag;
  }
  result.Set("BBRe", bbRe);
  result.Set("BBIm", bbIm);

  // Q = VSL^H (conjugate transpose) for MATLAB convention Q*A*Z = AA
  auto qRe = Napi::Float64Array::New(env, nn);
  auto qIm = Napi::Float64Array::New(env, nn);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      // Q(i,j) = conj(VSL(j,i))
      qRe[i + j * n] =  vsl[j + i * n].real;
      qIm[i + j * n] = -vsl[j + i * n].imag;
    }
  }
  result.Set("QRe", qRe);
  result.Set("QIm", qIm);

  // Z = VSR (no transpose)
  auto zRe = Napi::Float64Array::New(env, nn);
  auto zIm = Napi::Float64Array::New(env, nn);
  for (int i = 0; i < n * n; ++i) {
    zRe[i] = vsr[i].real;
    zIm[i] = vsr[i].imag;
  }
  result.Set("ZRe", zRe);
  result.Set("ZIm", zIm);

  // Alpha (complex) and beta (complex)
  auto alphaRe = Napi::Float64Array::New(env, sn);
  auto alphaIm = Napi::Float64Array::New(env, sn);
  auto betaRe = Napi::Float64Array::New(env, sn);
  auto betaIm = Napi::Float64Array::New(env, sn);
  for (int i = 0; i < n; ++i) {
    alphaRe[i] = alpha[i].real;
    alphaIm[i] = alpha[i].imag;
    betaRe[i] = beta_c[i].real;
    betaIm[i] = beta_c[i].imag;
  }
  result.Set("alphaRe", alphaRe);
  result.Set("alphaIm", alphaIm);
  result.Set("betaRe", betaRe);
  result.Set("betaIm", betaIm);

  // Optionally compute generalized eigenvectors via ztgevc
  if (computeEigvecs) {
    std::vector<lapack_complex_double> vr(vsr);  // start with VSR
    std::vector<lapack_complex_double> vl(vsl);  // start with VSL

    char side = 'B';
    char howmny = 'B';
    std::vector<int> select(n);
    int mm = n;
    int m_out = 0;
    std::vector<lapack_complex_double> work_tgevc(2 * n);
    std::vector<double> rwork_tgevc(2 * n);

    ztgevc_(&side, &howmny, select.data(), &n,
            a.data(), &n, b.data(), &n,
            vl.data(), &n, vr.data(), &n,
            &mm, &m_out, work_tgevc.data(), rwork_tgevc.data(), &info_val);

    if (info_val != 0) {
      Napi::Error::New(env, "qzComplex: ztgevc failed")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    auto vRe = Napi::Float64Array::New(env, nn);
    auto vIm = Napi::Float64Array::New(env, nn);
    for (int i = 0; i < n * n; ++i) {
      vRe[i] = vr[i].real;
      vIm[i] = vr[i].imag;
    }
    result.Set("VRe", vRe);
    result.Set("VIm", vIm);

    auto wRe = Napi::Float64Array::New(env, nn);
    auto wIm = Napi::Float64Array::New(env, nn);
    for (int i = 0; i < n * n; ++i) {
      wRe[i] = vl[i].real;
      wIm[i] = vl[i].imag;
    }
    result.Set("WRe", wRe);
    result.Set("WIm", wIm);
  }

  return result;
}
