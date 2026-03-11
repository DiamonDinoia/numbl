/**
 * qr() and qrComplex() — QR decomposition via LAPACK.
 *
 *   qr(data: Float64Array, m: number, n: number, econ: boolean,
 *      wantQ: boolean): {Q: Float64Array | undefined, R: Float64Array}
 *
 *   qrComplex(dataRe: Float64Array, dataIm: Float64Array, m: number, n: number,
 *             econ: boolean, wantQ: boolean):
 *             {QRe?: Float64Array, QIm?: Float64Array, RRe: Float64Array, RIm: Float64Array}
 *
 *     QR decomposition of an m×n matrix stored in column-major order.
 *     econ=true:   economy/thin QR — Q is m×k, R is k×n  (k = min(m,n))
 *     econ=false:  full QR         — Q is m×m, R is m×n
 *     wantQ=false: skips Q generation; Q properties are absent from the returned object.
 */

#include "lapack_common.h"

// ── qr() ─────────────────────────────────────────────────────────────────────

Napi::Value Qr(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 5
      || !info[0].IsTypedArray()
      || !info[1].IsNumber()
      || !info[2].IsNumber()
      || !info[3].IsBoolean()
      || !info[4].IsBoolean()) {
    Napi::TypeError::New(env,
      "qr: expected (Float64Array data, number m, number n, boolean econ, boolean wantQ)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arr = info[0].As<Napi::TypedArray>();
  if (arr.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "qr: data must be a Float64Array")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m      = info[1].As<Napi::Number>().Int32Value();
  int n      = info[2].As<Napi::Number>().Int32Value();
  bool econ  = info[3].As<Napi::Boolean>().Value();
  bool wantQ = info[4].As<Napi::Boolean>().Value();

  if (m <= 0 || n <= 0 || static_cast<int>(arr.ElementLength()) != m * n) {
    Napi::RangeError::New(env, "qr: data.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arr = info[0].As<Napi::Float64Array>();
  int k = m < n ? m : n; // k = min(m, n)

  // ── Step 1: QR factorisation (dgeqrf) ─────────────────────────────────────
  // Copy input into working buffer (dgeqrf overwrites A in-place)
  std::vector<double> a(m * n);
  std::memcpy(a.data(), float64arr.Data(), m * n * sizeof(double));

  std::vector<double> tau(k);
  int info_val = 0;

  // Query optimal workspace for dgeqrf
  int lwork = -1;
  double work_query = 0.0;
  dgeqrf_(&m, &n, a.data(), &m, tau.data(), &work_query, &lwork, &info_val);
  lwork = static_cast<int>(work_query);
  if (lwork < 1) lwork = k;

  std::vector<double> work(lwork);
  dgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info_val);

  if (info_val != 0) {
    Napi::Error::New(env, "qr: dgeqrf failed")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // ── Step 2: Extract R from the upper triangle of the factored matrix ───────
  // Economy R: k×n   Full R: m×n (rows k..m-1 are zero)
  int r_rows = econ ? k : m;
  std::vector<double> R(r_rows * n, 0.0);
  for (int j = 0; j < n; j++) {
    int ilim = j < k ? j : k - 1; // min(j, k-1)
    for (int i = 0; i <= ilim; i++) {
      R[i + j * r_rows] = a[i + j * m];
    }
  }

  // ── Step 3: Generate Q via dorgqr (only if wantQ) ─────────────────────────
  // Economy Q: m×k   Full Q: m×m
  int q_cols = econ ? k : m;

  auto result = Napi::Object::New(env);

  if (wantQ) {
    // Build a buffer of size m×q_cols containing the first q_cols columns of
    // the factored matrix (remaining columns are zero — dorgqr extends Q).
    std::vector<double> q_buf(m * q_cols, 0.0);
    int cols_to_copy = n < q_cols ? n : q_cols; // min(n, q_cols)
    for (int j = 0; j < cols_to_copy; j++) {
      for (int i = 0; i < m; i++) {
        q_buf[i + j * m] = a[i + j * m];
      }
    }

    // Query optimal workspace for dorgqr
    lwork = -1;
    dorgqr_(&m, &q_cols, &k, q_buf.data(), &m, tau.data(),
            &work_query, &lwork, &info_val);
    lwork = static_cast<int>(work_query);
    if (lwork < 1) lwork = q_cols;

    work.assign(lwork, 0.0);
    dorgqr_(&m, &q_cols, &k, q_buf.data(), &m, tau.data(),
            work.data(), &lwork, &info_val);

    if (info_val != 0) {
      Napi::Error::New(env, "qr: dorgqr failed")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    auto Q_arr = Napi::Float64Array::New(env, static_cast<size_t>(m * q_cols));
    std::memcpy(Q_arr.Data(), q_buf.data(), m * q_cols * sizeof(double));
    result.Set("Q", Q_arr);
  }

  auto R_arr = Napi::Float64Array::New(env, static_cast<size_t>(r_rows * n));
  std::memcpy(R_arr.Data(), R.data(), r_rows * n * sizeof(double));
  result.Set("R", R_arr);

  return result;
}

// ── qrComplex() ──────────────────────────────────────────────────────────────

Napi::Value QrComplex(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 6
      || !info[0].IsTypedArray()
      || !info[1].IsTypedArray()
      || !info[2].IsNumber()
      || !info[3].IsNumber()
      || !info[4].IsBoolean()
      || !info[5].IsBoolean()) {
    Napi::TypeError::New(env,
      "qrComplex: expected (Float64Array dataRe, Float64Array dataIm, "
      "number m, number n, boolean econ, boolean wantQ)")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto arrRe = info[0].As<Napi::TypedArray>();
  auto arrIm = info[1].As<Napi::TypedArray>();

  if (arrRe.TypedArrayType() != napi_float64_array ||
      arrIm.TypedArrayType() != napi_float64_array) {
    Napi::TypeError::New(env, "qrComplex: dataRe and dataIm must be Float64Arrays")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  int m      = info[2].As<Napi::Number>().Int32Value();
  int n      = info[3].As<Napi::Number>().Int32Value();
  bool econ  = info[4].As<Napi::Boolean>().Value();
  bool wantQ = info[5].As<Napi::Boolean>().Value();

  if (m <= 0 || n <= 0 ||
      static_cast<int>(arrRe.ElementLength()) != m * n ||
      static_cast<int>(arrIm.ElementLength()) != m * n) {
    Napi::RangeError::New(env,
      "qrComplex: dataRe.length and dataIm.length must equal m*n")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  auto float64arrRe = info[0].As<Napi::Float64Array>();
  auto float64arrIm = info[1].As<Napi::Float64Array>();
  int k = m < n ? m : n;

  // ── Step 1: Convert to interleaved complex format ──────────────────────────
  std::vector<lapack_complex_double> a(m * n);
  for (int i = 0; i < m * n; ++i) {
    a[i].real = float64arrRe[i];
    a[i].imag = float64arrIm[i];
  }

  std::vector<lapack_complex_double> tau(k);
  int info_val = 0;

  // ── Step 2: QR factorisation (zgeqrf) ──────────────────────────────────────
  int lwork = -1;
  lapack_complex_double work_query;
  zgeqrf_(&m, &n, a.data(), &m, tau.data(), &work_query, &lwork, &info_val);
  lwork = static_cast<int>(work_query.real);
  if (lwork < 1) lwork = k;

  std::vector<lapack_complex_double> work(lwork);
  zgeqrf_(&m, &n, a.data(), &m, tau.data(), work.data(), &lwork, &info_val);

  if (info_val != 0) {
    Napi::Error::New(env, "qrComplex: zgeqrf failed")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  // ── Step 3: Extract R from the upper triangle ──────────────────────────────
  int r_rows = econ ? k : m;
  std::vector<double> R_re(r_rows * n, 0.0);
  std::vector<double> R_im(r_rows * n, 0.0);
  for (int j = 0; j < n; j++) {
    int ilim = j < k ? j : k - 1;
    for (int i = 0; i <= ilim; i++) {
      R_re[i + j * r_rows] = a[i + j * m].real;
      R_im[i + j * r_rows] = a[i + j * m].imag;
    }
  }

  // ── Step 4: Generate Q via zungqr (only if wantQ) ─────────────────────────
  int q_cols = econ ? k : m;

  auto result = Napi::Object::New(env);

  if (wantQ) {
    std::vector<lapack_complex_double> q_buf(m * q_cols, {0.0, 0.0});
    int cols_to_copy = n < q_cols ? n : q_cols;
    for (int j = 0; j < cols_to_copy; j++) {
      for (int i = 0; i < m; i++) {
        q_buf[i + j * m] = a[i + j * m];
      }
    }

    lwork = -1;
    zungqr_(&m, &q_cols, &k, q_buf.data(), &m, tau.data(),
            &work_query, &lwork, &info_val);
    lwork = static_cast<int>(work_query.real);
    if (lwork < 1) lwork = q_cols;

    work.assign(lwork, {0.0, 0.0});
    zungqr_(&m, &q_cols, &k, q_buf.data(), &m, tau.data(),
            work.data(), &lwork, &info_val);

    if (info_val != 0) {
      Napi::Error::New(env, "qrComplex: zungqr failed")
          .ThrowAsJavaScriptException();
      return env.Null();
    }

    auto QRe_arr = Napi::Float64Array::New(env, static_cast<size_t>(m * q_cols));
    auto QIm_arr = Napi::Float64Array::New(env, static_cast<size_t>(m * q_cols));
    for (int i = 0; i < m * q_cols; ++i) {
      QRe_arr[i] = q_buf[i].real;
      QIm_arr[i] = q_buf[i].imag;
    }
    result.Set("QRe", QRe_arr);
    result.Set("QIm", QIm_arr);
  }

  auto RRe_arr = Napi::Float64Array::New(env, static_cast<size_t>(r_rows * n));
  auto RIm_arr = Napi::Float64Array::New(env, static_cast<size_t>(r_rows * n));
  std::memcpy(RRe_arr.Data(), R_re.data(), r_rows * n * sizeof(double));
  std::memcpy(RIm_arr.Data(), R_im.data(), r_rows * n * sizeof(double));
  result.Set("RRe", RRe_arr);
  result.Set("RIm", RIm_arr);

  return result;
}
