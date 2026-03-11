/**
 * svd() — Singular Value Decomposition via LAPACK dgesdd (divide-and-conquer).
 *
 *   svd(data: Float64Array, m: number, n: number, econ: boolean,
 *       computeUV: boolean): {U?: Float64Array, S: Float64Array, V?: Float64Array}
 *
 *     SVD of an m×n matrix stored in column-major order.
 *     econ=true,  computeUV=true:  economy SVD — U is m×k, S is k, V is n×k
 *     econ=false, computeUV=true:  full SVD    — U is m×m, S is k, V is n×n
 *     computeUV=false:             singular values only — S is k
 *     (k = min(m, n))
 *     Returns object with S (always) and optionally U and V as Float64Arrays
 *     in column-major order.
 */

#include "lapack_common.h"
#include <string>

// ── svd() ─────────────────────────────────────────────────────────────────────

Napi::Value Svd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()
      || !info[4].IsBoolean()) {
    Napi::TypeError::New(env,
      "svd: expected (Float64Array data, number m, number n, boolean econ, boolean computeUV)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "svd: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m          = info[1].As<Napi::Number>().Int32Value();
  int n          = info[2].As<Napi::Number>().Int32Value();
  bool econ      = info[3].As<Napi::Boolean>().Value();
  bool computeUV = info[4].As<Napi::Boolean>().Value();

  if (m <= 0 || n <= 0 || static_cast<int>(arr.ElementLength()) != m * n) {
    Napi::RangeError::New(env, "svd: data.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();
  int k = m < n ? m : n; // k = min(m, n)

  // ── Step 1: SVD computation (dgesdd) ──────────────────────────────────────
  // Copy input into working buffer (dgesdd overwrites A in-place)
  std::vector<double> a(m * n);
  std::memcpy(a.data(), float64arr.Data(), m * n * sizeof(double));

  std::vector<double> s(k); // singular values
  int info_val = 0;

  // Determine jobz parameter based on what we need to compute
  char jobz;
  if (!computeUV) {
    jobz = 'N'; // compute singular values only
  } else if (econ) {
    jobz = 'S'; // economy-size: U is m×k, VT is k×n
  } else {
    jobz = 'A'; // full-size: U is m×m, VT is n×n
  }

  // Allocate U and VT based on jobz
  int ldu, ldvt;
  std::vector<double> u_vec, vt_vec;

  if (jobz == 'N') {
    ldu  = m;
    ldvt = n;
    // U and VT are not referenced
  } else if (jobz == 'S') {
    ldu  = m;
    ldvt = k;
    u_vec.resize(m * k);
    vt_vec.resize(k * n);
  } else { // jobz == 'A'
    ldu  = m;
    ldvt = n;
    u_vec.resize(m * m);
    vt_vec.resize(n * n);
  }

  double* u_ptr  = (jobz == 'N') ? nullptr : u_vec.data();
  double* vt_ptr = (jobz == 'N') ? nullptr : vt_vec.data();

  // Query optimal workspace for dgesdd
  int lwork = -1;
  double work_query = 0.0;
  std::vector<int> iwork(8 * k);
  dgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          &work_query, &lwork, iwork.data(), &info_val);

  lwork = static_cast<int>(work_query);
  if (lwork < 1) lwork = 3 * k + std::max(m, n); // conservative fallback

  std::vector<double> work(lwork);
  dgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          work.data(), &lwork, iwork.data(), &info_val);

  if (info_val != 0) {
    Napi::Error::New(env, "svd: dgesdd failed")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // ── Step 2: Build result object ────────────────────────────────────────────
  auto result = Napi::Object::New(env);

  // Always return S (singular values)
  auto S_arr = Napi::Float64Array::New(env, static_cast<size_t>(k));
  std::memcpy(S_arr.Data(), s.data(), k * sizeof(double));
  result.Set("S", S_arr);

  if (computeUV) {
    // Return U
    int u_size = (jobz == 'S') ? m * k : m * m;
    auto U_arr = Napi::Float64Array::New(env, static_cast<size_t>(u_size));
    std::memcpy(U_arr.Data(), u_vec.data(), u_size * sizeof(double));
    result.Set("U", U_arr);

    // Return V (transpose of VT from LAPACK)
    // LAPACK returns VT (V^T); we return V = VT^T
    int vt_rows = (jobz == 'S') ? k : n;
    int vt_cols = n;
    std::vector<double> v_vec(vt_rows * vt_cols);
    for (int i = 0; i < vt_rows; i++) {
      for (int j = 0; j < vt_cols; j++) {
        // V(j, i) = VT(i, j) — column-major transpose
        v_vec[j + i * vt_cols] = vt_vec[i + j * vt_rows];
      }
    }
    auto V_arr = Napi::Float64Array::New(env, static_cast<size_t>(vt_rows * vt_cols));
    std::memcpy(V_arr.Data(), v_vec.data(), vt_rows * vt_cols * sizeof(double));
    result.Set("V", V_arr);
  }

  return result;
}

// ── svdComplex() ─────────────────────────────────────────────────────────────

Napi::Value SvdComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 6
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsNumber()
      || !info[4].IsBoolean()
      || !info[5].IsBoolean()) {
    Napi::TypeError::New(env,
      "svdComplex: expected (Float64Array dataRe, Float64Array dataIm, number m, number n, boolean econ, boolean computeUV)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();
  if (arrRe.TypedArrayType() != napi_float64_array || arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "svdComplex: data must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m          = info[2].As<Napi::Number>().Int32Value();
  int n          = info[3].As<Napi::Number>().Int32Value();
  bool econ      = info[4].As<Napi::Boolean>().Value();
  bool computeUV = info[5].As<Napi::Boolean>().Value();

  if (m <= 0 || n <= 0
      || static_cast<int>(arrRe.ElementLength()) != m * n
      || static_cast<int>(arrIm.ElementLength()) != m * n) {
    Napi::RangeError::New(env, "svdComplex: data.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64Re = info[0].As<Napi::Float64Array>();
  auto float64Im = info[1].As<Napi::Float64Array>();
  int k = m < n ? m : n;

  // Convert split real/imag to interleaved complex format
  std::vector<lapack_complex_double> a(m * n);
  for (int i = 0; i < m * n; ++i) {
    a[i].real = float64Re[i];
    a[i].imag = float64Im[i];
  }

  std::vector<double> s(k);
  int info_val = 0;

  char jobz;
  if (!computeUV) {
    jobz = 'N';
  } else if (econ) {
    jobz = 'S';
  } else {
    jobz = 'A';
  }

  int ldu, ldvt;
  std::vector<lapack_complex_double> u_vec, vt_vec;

  if (jobz == 'N') {
    ldu  = m;
    ldvt = n;
  } else if (jobz == 'S') {
    ldu  = m;
    ldvt = k;
    u_vec.resize(m * k);
    vt_vec.resize(k * n);
  } else { // 'A'
    ldu  = m;
    ldvt = n;
    u_vec.resize(m * m);
    vt_vec.resize(n * n);
  }

  lapack_complex_double* u_ptr  = (jobz == 'N') ? nullptr : u_vec.data();
  lapack_complex_double* vt_ptr = (jobz == 'N') ? nullptr : vt_vec.data();

  // Compute rwork size for zgesdd
  // For jobz='N': 7*min(m,n) (some implementations need more than the documented 5)
  // For jobz='S' or 'A': min(m,n) * max(5*min(m,n)+7, 2*max(m,n)+2*min(m,n)+1)
  int rwork_size;
  if (jobz == 'N') {
    rwork_size = 7 * k;
  } else {
    int t1 = 5 * k + 7;
    int t2 = 2 * std::max(m, n) + 2 * k + 1;
    rwork_size = k * std::max(t1, t2);
  }
  std::vector<double> rwork(rwork_size);
  std::vector<int> iwork(8 * k);

  // Keep a copy of the input for potential zgesvd fallback
  std::vector<lapack_complex_double> a_backup(a);

  // Workspace query
  int lwork = -1;
  lapack_complex_double work_query;
  zgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          &work_query, &lwork, rwork.data(), iwork.data(), &info_val);

  lwork = static_cast<int>(work_query.real);
  if (lwork < 1) lwork = 3 * k + std::max(m, n);

  std::vector<lapack_complex_double> work(lwork);
  zgesdd_(&jobz, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu, vt_ptr, &ldvt,
          work.data(), &lwork, rwork.data(), iwork.data(), &info_val);

  // If zgesdd fails, fall back to zgesvd (standard algorithm, more robust)
  if (info_val != 0) {
    // Restore the input matrix (zgesdd overwrites it)
    a = a_backup;
    info_val = 0;

    // Map jobz to jobu/jobvt for zgesvd
    char jobu, jobvt;
    if (jobz == 'N') {
      jobu = 'N'; jobvt = 'N';
    } else if (jobz == 'S') {
      jobu = 'S'; jobvt = 'S';
    } else {
      jobu = 'A'; jobvt = 'A';
    }

    // rwork for zgesvd: 5*min(m,n)
    std::vector<double> rwork_svd(5 * k);

    // Workspace query for zgesvd
    lwork = -1;
    zgesvd_(&jobu, &jobvt, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu,
            vt_ptr, &ldvt, &work_query, &lwork, rwork_svd.data(), &info_val);

    lwork = static_cast<int>(work_query.real);
    if (lwork < 1) lwork = 2 * k + std::max(m, n);

    work.resize(lwork);
    info_val = 0;
    zgesvd_(&jobu, &jobvt, &m, &n, a.data(), &m, s.data(), u_ptr, &ldu,
            vt_ptr, &ldvt, work.data(), &lwork, rwork_svd.data(), &info_val);

    if (info_val != 0) {
      std::string msg = "svdComplex: zgesvd failed (info=" + std::to_string(info_val)
                      + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ")";
      Napi::Error::New(env, msg).ThrowAsJavaScriptException();
      return env.Null();
    }
  }

  // Build result
  auto result = Napi::Object::New(env);

  // S is always real
  auto S_arr = Napi::Float64Array::New(env, static_cast<size_t>(k));
  std::memcpy(S_arr.Data(), s.data(), k * sizeof(double));
  result.Set("S", S_arr);

  if (computeUV) {
    // U: convert from interleaved to split real/imag
    int u_size = (jobz == 'S') ? m * k : m * m;
    auto URe = Napi::Float64Array::New(env, static_cast<size_t>(u_size));
    auto UIm = Napi::Float64Array::New(env, static_cast<size_t>(u_size));
    for (int i = 0; i < u_size; i++) {
      URe[i] = u_vec[i].real;
      UIm[i] = u_vec[i].imag;
    }
    result.Set("URe", URe);
    result.Set("UIm", UIm);

    // V = conj(VT^T): conjugate transpose of VT
    int vt_rows = (jobz == 'S') ? k : n;
    int vt_cols = n;
    int v_size = vt_rows * vt_cols;
    auto VRe = Napi::Float64Array::New(env, static_cast<size_t>(v_size));
    auto VIm = Napi::Float64Array::New(env, static_cast<size_t>(v_size));
    for (int i = 0; i < vt_rows; i++) {
      for (int j = 0; j < vt_cols; j++) {
        // V(j, i) = conj(VT(i, j)) — column-major conjugate transpose
        int v_idx = j + i * vt_cols;
        int vt_idx = i + j * vt_rows;
        VRe[v_idx] = vt_vec[vt_idx].real;
        VIm[v_idx] = -vt_vec[vt_idx].imag;  // conjugate
      }
    }
    result.Set("VRe", VRe);
    result.Set("VIm", VIm);
  }

  return result;
}
